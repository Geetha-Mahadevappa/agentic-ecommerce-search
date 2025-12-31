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
from typing import Tuple

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, Engine

from embeddings_pipeline.download_datasets import download_and_copy

from logging_config import setup_logging

# Use the shared logging configuration so logs go to embeddings.log
setup_logging("logs/embeddings.log")
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(self, config_path: Path = Path("configs/config_embedding.yaml")):
        """Runs the full embedding workflow."""
        self.cfg = self._load_config(config_path)
        self.paths = self.cfg["paths"]
        self.model_cfg = self.cfg["model"]
        self.embed_cfg = self.cfg["embedding"]

    def _load_config(self, path: str) -> dict:
        """Load the embedding configuration YAML file."""
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)

    def ensure_raw_data(self) -> tuple[Path, Path]:
        """Make sure raw product CSV files exist. Download them if missing."""
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

    def manifest_matches(self, manifest_path: Path) -> bool:
        """Check if existing embeddings match the current model version."""
        if not manifest_path.exists():
            return False
        with open(manifest_path) as f:
            data = json.load(f)
        return data.get("model_version") == self.model_cfg["version"]

    def init_metadata_db(self, path: Path) -> Tuple[Engine, Table]:
        """Create the SQLite metadata table if it doesn't exist."""
        engine = create_engine(f"sqlite:///{path}")
        meta = MetaData()

        table = Table(
            "chunks", meta,   # keeping your table name
            Column("variant_id", String, primary_key=True),
            Column("product_id", String, index=True),
            Column("text", String),
            Column("product_name", String),
            Column("category", String),
            Column("price", Float),
            Column("embed_model", String),
            Column("created_at", Float),
        )

        meta.create_all(engine)
        return engine, table

    def validate_embeddings(self, embeddings: np.ndarray, texts: list[str]) -> None:
        """
        Validate the embedding matrix before saving or building a FAISS index.
        """

        logger.info("Validating embedding matrix...")

        if embeddings.ndim != 2:
            raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}")

        if embeddings.shape[0] != len(texts):
            raise ValueError(f"Embedding count mismatch: {embeddings.shape[0]} embeddings for {len(texts)} texts")

        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain non‑finite values")

        logger.info(f"Embedding validation passed. Shape={embeddings.shape}, dtype={embeddings.dtype}")

    def write_metadata(self, engine: Engine, table: Table, rows: list[dict]) -> None:
        """Insert chunk metadata into the SQLite database."""
        with engine.begin() as conn:
            for r in rows:
                conn.execute(table.insert().prefix_with("OR REPLACE"), r)

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

        # FIX: KEEP ALL VARIANTS — assign unique variant_id
        df["variant_id"] = df.index.astype(str)
        logger.info(f"Assigned unique variant_id to all {len(df)} variants")

        # Cap overly long review text
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

        # FIX: one embedding per VARIANT
        logger.info("Building variant-level text (no chunking)...")
        variant_texts = df["canonical_text"].tolist()

        # Embeddings
        logger.info("Loading embedding model: %s", self.model_cfg["name"])
        model = SentenceTransformer(self.model_cfg["name"])

        logger.info(f"Encoding {len(variant_texts)} variants...")
        embeddings = model.encode(
            variant_texts,
            batch_size=self.embed_cfg["batch_size"],
            show_progress_bar=False
        ).astype(np.float32)

        # Validate embeddings
        self.validate_embeddings(embeddings, variant_texts)

        # Save embeddings
        tmp = embed_dir / f"tmp_{int(time.time())}.npz"
        np.savez_compressed(tmp, embeddings=embeddings)
        tmp.replace(embeddings_npz)

        # Mapping: variant_id → embedding index
        mapping = df[["variant_id", "ProductID"]].copy()
        mapping = mapping.rename(columns={"ProductID": "product_id"})
        mapping["embedding_index"] = range(len(mapping))
        mapping.to_parquet(mapping_parquet, index=False)

        # Metadata DB: one row per variant
        engine, table = self.init_metadata_db(metadata_db)
        meta_rows = []

        for _, row in df.iterrows():
            # Compute actual price instead of the product
            unit_price = float(row["PurchasePrice"]) / float(row["PurchaseQuantity"])

            meta_rows.append({
                "variant_id": row["variant_id"],
                "product_id": str(row["ProductID"]),
                "text": row["canonical_text"],
                "product_name": row["ProductName"],
                "category": row["ProductCategory"],
                "price": round(unit_price, 2),
                "embed_model": self.model_cfg["version"],
                "created_at": time.time(),
            })

        self.write_metadata(engine, table, meta_rows)

        manifest = {
            "model_name": self.model_cfg["name"],
            "model_version": self.model_cfg["version"],
            "num_variants": len(variant_texts),
            "embedding_dim": embeddings.shape[1],
            "created_at": time.time(),
        }
        with open(manifest_json, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Embedding pipeline completed successfully.")


if __name__ == "__main__":
    try:
        EmbeddingPipeline(Path("configs/config_embedding.yaml")).run()
    except Exception:
        logger.exception("Embedding pipeline failed")
