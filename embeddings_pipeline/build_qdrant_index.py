#!/usr/bin/env python3
"""Build a local Qdrant collection from existing embeddings and metadata."""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import sqlite3
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from logging_config import setup_logging

setup_logging("logs/embeddings.log")
logger = logging.getLogger(__name__)


class QdrantIndexBuilder:
    def __init__(self, config_path: Path = Path("configs/config_agents.yaml")):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.paths = self.cfg["paths"]

    def _load_embeddings(self) -> np.ndarray:
        npz_path = Path("configs/config_embedding.yaml")
        with open(npz_path, "r", encoding="utf-8") as f:
            emb_cfg = yaml.safe_load(f)
        emb_path = Path(emb_cfg["paths"]["embeddings_npz"])
        arr = np.load(emb_path)
        emb = np.asarray(arr["embeddings"], dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] == 0:
            raise ValueError("Embeddings are missing or invalid")
        return emb

    def _load_mapping(self) -> pd.DataFrame:
        mapping = pd.read_parquet(self.paths["mapping_parquet"])
        mapping = mapping.sort_values("embedding_index").reset_index(drop=True)
        mapping["variant_id"] = mapping["variant_id"].astype(str)
        mapping["product_id"] = mapping["product_id"].astype(str)
        return mapping

    def _load_metadata(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.paths["metadata_db"])
        try:
            df = pd.read_sql(
                """
                SELECT variant_id, product_id, text, product_name, category, country, price_level, price
                FROM chunks
                """,
                conn,
            )
        finally:
            conn.close()
        df["variant_id"] = df["variant_id"].astype(str)
        return df.set_index("variant_id", drop=False)

    def run(self) -> None:
        embeddings = self._load_embeddings()
        mapping = self._load_mapping()
        metadata = self._load_metadata()

        if len(mapping) != embeddings.shape[0]:
            raise ValueError("Mapping and embedding counts do not match")

        qdrant_path = Path(self.paths["qdrant_path"])
        qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(qdrant_path))
        collection = self.paths["qdrant_collection"]

        if client.collection_exists(collection):
            client.delete_collection(collection)

        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=int(embeddings.shape[1]), distance=Distance.COSINE),
        )

        points = []
        for idx, row in mapping.iterrows():
            vid = str(row["variant_id"])
            meta = metadata.loc[vid]
            payload = {
                "variant_id": vid,
                "product_id": str(meta["product_id"]),
                "text": str(meta["text"]),
                "product_name": str(meta["product_name"]),
                "category": str(meta["category"]),
                "country": str(meta["country"]),
                "price_level": str(meta["price_level"]),
                "price": float(meta["price"]),
            }
            points.append(PointStruct(id=idx, vector=embeddings[idx].tolist(), payload=payload))

        client.upsert(collection_name=collection, points=points)
        logger.info("Qdrant collection '%s' built with %d points", collection, len(points))


if __name__ == "__main__":
    QdrantIndexBuilder(Path("configs/config_agents.yaml")).run()
