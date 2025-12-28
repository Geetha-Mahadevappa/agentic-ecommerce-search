#!/usr/bin/env python3

"""
Embedding pipeline for the e‑commerce dataset.

This script loads configuration, ensures raw data is available, prepares
canonical product text, chunks it, generates embeddings, and writes out
all related metadata. The goal is to keep the workflow simple, predictable,
and easy to rerun.
"""

import json
import time
import logging
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float

from embeddings_pipeline.download_datasets import download_and_copy

from logging_config import setup_logging

# Use the shared logging configuration so logs go to embeddings.log
setup_logging("logs/embeddings.log")
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(self, config_path="configs/config_embedding.yaml"):
        """Runs the full embedding workflow."""
        self.cfg = self._load_config(config_path)
        self.paths = self.cfg["paths"]
        self.model_cfg = self.cfg["model"]
        self.embed_cfg = self.cfg["embedding"]

    def _load_config(self, path):
        """
        Read and parse the YAML config file.

        Parameters
        ----------
        path : str
            Path to the config file.

        Returns
        -------
        dict
            Parsed configuration dictionary.
        """
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)

    def ensure_raw_data(self):
        """
        Make sure raw CSV files exist. Download them if missing.

        Returns
        -------
        tuple[Path, Path]
            Paths to purchase CSV and review CSV.
        """
        raw_dir = Path(self.paths["raw_dir"])
        raw_dir.mkdir(parents=True, exist_ok=True)

        purchase_csv = raw_dir / "customer_purchase_data.csv"
        reviews_csv = raw_dir / "customer_reviews_data.csv"

        if purchase_csv.exists() and reviews_csv.exists():
            logger.info(f"Raw data found in {raw_dir}")
            return purchase_csv, reviews_csv

        logger.info("Raw data missing. Downloading dataset...")
        download_and_copy(raw_dir)

        if not purchase_csv.exists() or not reviews_csv.exists():
            raise FileNotFoundError("Dataset download finished but CSV files missing.")
        return purchase_csv, reviews_csv

    def chunk_text(self, text):
        """
        Split text into overlapping word chunks.

        Parameters
        ----------
        text : str
            Input text to be chunked.

        Returns
        -------
        list[str]
            List of text chunks.
        """
        max_words = self.embed_cfg["chunk_words"]
        overlap = self.embed_cfg["chunk_overlap"]

        if not text:
            return []

        words = text.split()
        if len(words) <= max_words:
            return [" ".join(words)]

        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i + max_words]
            chunks.append(" ".join(chunk))
            i += max_words - overlap
        return chunks

    def manifest_matches(self, manifest_path):
        """
        Check if existing embeddings match the current model version.

        Parameters
        ----------
        manifest_path : Path
            Path to the manifest JSON file.

        Returns
        -------
        bool
            True if versions match, False otherwise.
        """
        if not manifest_path.exists():
            return False
        with open(manifest_path) as f:
            data = json.load(f)
        return data.get("model_version") == self.model_cfg["version"]

    def init_metadata_db(self, path):
        """
        Create the SQLite metadata table if it doesn't exist.

        Parameters
        ----------
        path : Path
            Path to the SQLite database file.

        Returns
        -------
        tuple
            (engine, table) SQLAlchemy engine and table object.
        """
        engine = create_engine(f"sqlite:///{path}")
        meta = MetaData()

        table = Table(
            "chunks", meta,
            Column("chunk_id", String, primary_key=True),
            Column("product_id", String, index=True),
            Column("chunk_index", Integer),
            Column("text", String),
            Column("product_name", String),
            Column("category", String),
            Column("price", Float),
            Column("embed_model", String),
            Column("created_at", Float),
        )

        meta.create_all(engine)
        return engine, table

    def write_metadata(self, engine, table, rows):
        """
        Insert chunk metadata into the SQLite database.

        Parameters
        ----------
        engine : sqlalchemy.Engine
            Database engine.

        table : sqlalchemy.Table
            Metadata table.

        rows : list[dict]
            Metadata rows to insert.
        """
        with engine.begin() as conn:
            for r in rows:
                conn.execute(table.insert().prefix_with("OR REPLACE"), r)

    def validate_embeddings(self, embeddings, chunk_texts):
        """
        Validate the embedding matrix before saving or building a FAISS index.

        This check acts as a safety gate. If something went wrong during encoding
        (e.g., partial batches, NaNs, wrong shape), it's better to stop the pipeline
        here than to silently produce a corrupted index. The user can simply rerun
        the pipeline after fixing the issue.

        Parameters
        ----------
        embeddings : np.ndarray
            The full 2D embedding matrix produced by the model. Each row should
            correspond to one text chunk.

        chunk_texts : list[str]
            The list of text chunks that were fed into the embedding model. Used
            to verify that the number of embeddings matches the number of chunks.
        """

        logger.info("Validating embedding matrix...")

        # The embedding matrix must be 2D: (num_chunks, embedding_dim)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}")

        # Ensure we have one embedding per chunk
        if embeddings.shape[0] != len(chunk_texts):
            raise ValueError(f"Embedding count mismatch: {embeddings.shape[0]} embeddings for {len(chunk_texts)} chunks")

        # Check for NaNs or infinite values — both indicate a broken batch
        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain non‑finite values")

        logger.info(f"Embedding validation passed. Shape={embeddings.shape}, dtype={embeddings.dtype}")

    def run(self):
        """Execute the full embedding pipeline."""
        logger.info("Starting embedding pipeline...")

        embed_dir = Path(self.paths["embed_dir"])
        embed_dir.mkdir(parents=True, exist_ok=True)

        embeddings_npz = Path(self.paths["embeddings_npz"])
        mapping_parquet = Path(self.paths["mapping_parquet"])
        metadata_db = Path(self.paths["metadata_db"])
        manifest_json = Path(self.paths["manifest_json"])

        # Ensure raw data exists
        purchase_csv, reviews_csv = self.ensure_raw_data()

        # Skip if embeddings already exist and version matches
        if embeddings_npz.exists() and self.manifest_matches(manifest_json):
            logger.info("Embeddings already exist and version matches. Skipping.")
            return

        # Load data
        logger.info("Loading purchase data...")
        purchases = pd.read_csv(purchase_csv).fillna("")

        logger.info("Loading review data...")
        reviews = pd.read_csv(reviews_csv).fillna("")

        review_texts = (
            reviews.groupby("ProductID")["ReviewText"]
            .apply(lambda x: " ".join(x))
            .reset_index()
            .rename(columns={"ReviewText": "AllReviews"})
        )

        df = purchases.merge(review_texts, on="ProductID", how="left")
        df["AllReviews"] = df["AllReviews"].fillna("")

        # Drop products with no reviews at all
        df = df[df["AllReviews"].str.strip() != ""]
        logger.info(f"Filtered out products with no reviews. Remaining products: {len(df)}")

        # Cap overly long review text to keep chunking and embedding efficient
        max_chars = self.embed_cfg["max_review_chars"]
        df["AllReviews"] = df["AllReviews"].str[:max_chars]
        logger.info("Capped review text to max_review_chars=%d", max_chars)

        df["canonical_text"] = (
            df["ProductName"].astype(str)
            + " | "
            + df["ProductCategory"].astype(str)
            + " | "
            + df["AllReviews"].astype(str)
        )

        # Chunking
        logger.info("Chunking product text...")
        chunk_texts = []
        mapping = []
        meta_rows = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            pid = str(row["ProductID"])
            pname = row["ProductName"]
            cat = row["ProductCategory"]
            price = float(row["PurchasePrice"])

            chunks = self.chunk_text(row["canonical_text"])

            for idx, chunk in enumerate(chunks):
                cid = f"{pid}__{idx}"
                chunk_texts.append(chunk)
                mapping.append({"chunk_id": cid, "product_id": pid})

                meta_rows.append({
                    "chunk_id": cid,
                    "product_id": pid,
                    "chunk_index": idx,
                    "text": chunk,
                    "product_name": pname,
                    "category": cat,
                    "price": price,
                    "embed_model": self.model_cfg["version"],
                    "created_at": time.time(),
                })

        # Embeddings
        logger.info("Loading embedding model: %s", self.model_cfg["name"])
        model = SentenceTransformer(self.model_cfg["name"])

        logger.info(f"Encoding {len(chunk_texts)} chunks...")
        all_embs = []
        bs = self.embed_cfg["batch_size"]

        for i in tqdm(range(0, len(chunk_texts), bs)):
            batch = chunk_texts[i:i + bs]
            embs = model.encode(batch, batch_size=64, show_progress_bar=False)
            all_embs.append(np.asarray(embs, dtype=np.float32))

        embeddings = np.vstack(all_embs)

        # Validate embeddings before saving.
        self.validate_embeddings(embeddings, chunk_texts)

        # Save outputs
        tmp = embed_dir / f"tmp_{int(time.time())}.npz"
        np.savez_compressed(tmp, embeddings=embeddings)
        tmp.replace(embeddings_npz)

        pd.DataFrame(mapping).to_parquet(mapping_parquet, index=False)

        engine, table = self.init_metadata_db(metadata_db)
        self.write_metadata(engine, table, meta_rows)

        manifest = {
            "model_name": self.model_cfg["name"],
            "model_version": self.model_cfg["version"],
            "num_chunks": len(chunk_texts),
            "embedding_dim": embeddings.shape[1],
            "created_at": time.time(),
        }
        with open(manifest_json, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Embedding pipeline completed successfully.")

if __name__ == "__main__":
    try:
        EmbeddingPipeline().run()
    except Exception:
        logger.exception("Embedding pipeline failed")
