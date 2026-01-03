#!/usr/bin/env python3

"""
Embedding pipeline for the e‑commerce dataset.

This script loads configuration, ensures raw data is available, prepares
canonical product text, generates embeddings, and writes out all related
metadata. The goal is to keep the workflow simple, predictable, and easy
to rerun.
"""

import json
import time
import logging
from pathlib import Path
from typing import Tuple

import yaml
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Engine

from embeddings_pipeline.download_datasets import download_and_copy
from logging_config import setup_logging

# Use the shared logging configuration so logs go to embeddings.log
setup_logging("logs/embeddings.log")
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(self, config_path: Path = Path("configs/config_embedding.yaml")):
        """Initialize the embedding pipeline with configuration."""
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
        """Ensure raw product CSV files exist. Download them if missing."""
        raw_dir = Path(self.paths["raw_dir"])
        raw_dir.mkdir(parents=True, exist_ok=True)

        purchase_csv = raw_dir / self.paths["purchase_csv"]
        reviews_csv = raw_dir / self.paths["reviews_csv"]

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
            "chunks", meta,
            Column("variant_id", String, primary_key=True),
            Column("product_id", String, index=True),
            Column("text", String),
            Column("product_name", String),
            Column("category", String),
            Column("country", String),
            Column("price_level", String),
            Column("price", Float),
            Column("embed_model", String),
            Column("created_at", Float),
        )

        meta.create_all(engine)
        return engine, table

    def normalize_price_by_country(self, purchases: pd.DataFrame) -> pd.DataFrame:
        """Normalize unit price within each country and assign price_level buckets."""
        logger.info("Normalizing price by country...")

        purchases["unit_price"] = (
            purchases["PurchasePrice"] / purchases["PurchaseQuantity"]
        )

        def _normalize(group: pd.DataFrame) -> pd.DataFrame:
            p = group["unit_price"]
            p_min, p_max = p.min(), p.max()
            if p_max == p_min:
                group["unit_price_norm"] = 0.5
            else:
                group["unit_price_norm"] = (p - p_min) / (p_max - p_min)
            return group

        purchases = purchases.groupby("Country", group_keys=False).apply(_normalize)

        def bucket(x: float) -> str:
            if x <= 0.33:
                return "Low"
            elif x <= 0.66:
                return "Mid"
            return "High"

        purchases["price_level"] = purchases["unit_price_norm"].apply(bucket)
        return purchases

    def prepare_reviews(self, reviews: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate reviews by ProductID using unique reviews, top-N selection,
        and max_chars truncation. This keeps review text compact and avoids
        overweighting repeated content.
        """
        logger.info("Preparing aggregated review text...")

        max_reviews = self.embed_cfg.get("max_reviews", 5)
        max_chars = self.embed_cfg["max_review_chars"]

        def combine_reviews(series):
            unique_reviews = series.unique()
            selected = unique_reviews[:max_reviews]
            combined = " ".join(selected)
            return combined[:max_chars]

        review_texts = (
            reviews.groupby("ProductID")["ReviewText"]
            .apply(combine_reviews)
            .reset_index()
            .rename(columns={"ReviewText": "AllReviews"})
        )
        return review_texts

    def validate_embeddings(self, embeddings: np.ndarray, texts: list[str]) -> None:
        """Validate the embedding matrix before saving or building a FAISS index."""
        logger.info("Validating embedding matrix...")

        # Structural checks
        if embeddings.ndim != 2:
            raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}")

        if embeddings.shape[0] != len(texts):
            raise ValueError(f"Embedding count mismatch: {embeddings.shape[0]} embeddings for {len(texts)} texts")

        # Numerical integrity checks
        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain non‑finite values")

        # Zero-vector detection
        if (np.linalg.norm(embeddings, axis=1) == 0).any():
            raise ValueError("Zero-vector embeddings detected.")

        # Abnormal norm detection
        norms = np.linalg.norm(embeddings, axis=1)
        if (norms < 1e-6).any() or (norms > 1000).any():
            raise ValueError("Abnormal embedding norms detected.")
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

        purchase_csv, reviews_csv = self.ensure_raw_data()

        if embeddings_npz.exists() and self.manifest_matches(manifest_json):
            logger.info("Embeddings already exist and version matches. Skipping.")
            return

        logger.info("Loading purchase data...")
        purchases = pd.read_csv(purchase_csv).fillna("")
        purchases = self.normalize_price_by_country(purchases)

        logger.info("Loading review data...")
        reviews = pd.read_csv(reviews_csv).fillna("")

        # Prepare aggregated review text
        review_texts = self.prepare_reviews(reviews)

        # Join general reviews back to all purchase rows (variants)
        df = purchases.merge(review_texts, on="ProductID", how="left")
        df["AllReviews"] = df["AllReviews"].fillna("")

        # Drop products with no reviews
        df = df[df["AllReviews"].str.strip() != ""]
        logger.info(f"Filtered out products with no reviews. Remaining products: {len(df)}")

        # VirtualID = ProductID + ProductName + Country
        df["variant_id"] = (
            df["ProductID"].astype(str)
            + "_"
            + df["ProductName"].astype(str).str.replace(" ", "_")
            + "_"
            + df["Country"].astype(str)
        )

        logger.info(f"Assigned VirtualID-based variant_id to all {len(df)} variants")

        # Canonical text
        df["canonical_text"] = (
            df["ProductName"].astype(str)
            + " | "
            + df["ProductCategory"].astype(str)
            + " | country: "
            + df["Country"].astype(str)
            + " | price_level: "
            + df["price_level"].astype(str)
            + " | General reviews for ProductID "
            + df["ProductID"].astype(str)
            + ": "
            + df["AllReviews"].astype(str)
        )

        variant_texts = df["canonical_text"].tolist()

        logger.info("Loading embedding model: %s", self.model_cfg["name"])
        model = SentenceTransformer(self.model_cfg["name"])

        logger.info(f"Encoding {len(variant_texts)} variants...")
        embeddings = model.encode(
            variant_texts,
            batch_size=self.embed_cfg["batch_size"],
            show_progress_bar=False
        ).astype(np.float32)

        self.validate_embeddings(embeddings, variant_texts)

        tmp = embed_dir / f"tmp_{int(time.time())}.npz"
        np.savez_compressed(tmp, embeddings=embeddings)
        tmp.replace(embeddings_npz)

        mapping = df[["variant_id", "ProductID"]].copy()
        mapping = mapping.rename(columns={"ProductID": "product_id"})
        mapping["embedding_index"] = range(len(mapping))
        mapping.to_parquet(mapping_parquet, index=False)

        engine, table = self.init_metadata_db(metadata_db)
        meta_rows: list[dict] = []

        for _, row in df.iterrows():
            unit_price = float(row["PurchasePrice"]) / float(row["PurchaseQuantity"])

            meta_rows.append({
                "variant_id": row["variant_id"],
                "product_id": str(row["ProductID"]),
                "text": row["canonical_text"],
                "product_name": row["ProductName"],
                "category": row["ProductCategory"],
                "country": row["Country"],
                "price_level": row["price_level"],
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
