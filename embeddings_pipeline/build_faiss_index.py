#!/usr/bin/env python3

"""
FAISS Index Builder

This module builds a FAISS index from a precomputed embedding matrix.
It is designed to run after the embedding pipeline has generated
`embeddings.npz` and written it to disk.

The resulting index is saved to the path defined in the agent config.
"""

from pathlib import Path

import faiss
import numpy as np
import yaml
import logging
import pandas as pd

from logging_config import setup_logging
setup_logging("logs/embeddings.log")
logger = logging.getLogger(__name__)

class FaissIndexBuilder:
    def __init__(self, config_path: Path = Path("configs/config_agents.yaml")):
        """
        Build a FAISS index from a precomputed embedding matrix.

        This class loads configuration, reads the embedding file produced by the
        embedding pipeline, constructs an HNSW index, and writes it to disk.
        """
        self.config = self._load_config(config_path)
        self.paths = self.config["paths"]

    def _load_config(self, path: Path) -> dict:
        """Load the agent configuration YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def load_embeddings(self) -> np.ndarray:
        """Load the embedding matrix from .npz file."""
        npz_path = Path(self.paths["embeddings_npz"])
        if not npz_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {npz_path}")

        logger.info(f"Loading embeddings from {npz_path}")
        arr = np.load(npz_path)

        # Ensure embeddings are float32 and contiguous for FAISS
        embeddings = np.asarray(arr["embeddings"], dtype=np.float32)
        embeddings = np.ascontiguousarray(embeddings)

        if embeddings.shape[0] == 0:
            raise ValueError("Embedding file contains zero vectors — cannot build FAISS index.")

        return embeddings

    def load_mapping(self) -> pd.DataFrame:
        """Load and normalize the product mapping used by retrieval."""
        mapping_path = Path(self.paths["mapping_parquet"])
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

        mapping = pd.read_parquet(mapping_path)

        # variant_id is now the primary key for FAISS alignment
        mapping["variant_id"] = mapping["variant_id"].astype(str).str.strip()
        mapping["product_id"] = mapping["product_id"].astype(str).str.strip()

        return mapping

    def align_embeddings_with_mapping(self, embeddings: np.ndarray, mapping: pd.DataFrame):
        """
        Ensure embeddings and mapping are aligned row‑by‑row.

        FAISS requires that embeddings[i] corresponds to mapping.iloc[i].
        If mapping was reordered after embeddings were created, we must
        reorder embeddings to match mapping.
        """
        logger.info("Aligning embeddings with mapping using embedding_index...")

        if "embedding_index" not in mapping.columns:
            raise ValueError("Mapping is missing 'embedding_index' column required for alignment.")

        # sort mapping by embedding_index to restore original order
        mapping_sorted = mapping.sort_values("embedding_index").reset_index(drop=True)

        if embeddings.shape[0] != mapping_sorted.shape[0]:
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match mapping count ({mapping_sorted.shape[0]})"
            )

        # embedding_index should be 0..N-1 after sorting
        expected = np.arange(len(mapping_sorted))
        actual = mapping_sorted["embedding_index"].to_numpy()
        if not np.array_equal(expected, actual):
            raise ValueError(
                "embedding_index column is not a contiguous 0..N-1 sequence after sorting; "
                "embedding/mapping alignment is invalid."
            )

        # embeddings are already in the correct order; we align mapping to them
        embeddings_aligned = embeddings
        logger.info("Alignment complete. Embeddings and mapping now share identical ordering.")
        return embeddings_aligned, mapping_sorted

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build an HNSW FAISS index from the embedding matrix."""
        dim = embeddings.shape[1]
        logger.info("Building FAISS index: dim=%d, n_vectors=%d", dim, embeddings.shape[0])

        # HNSW index (L2 distance)
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200

        index.add(embeddings)
        return index

    def verify_index(self, index: faiss.Index, embeddings: np.ndarray) -> None:
        """
        Verify that the FAISS index is valid before saving it.
        We confirm that dimensions match, the index contains the expected
        number of vectors, and that a basic nearest‑neighbor search works.
        """
        logger.info("Verifying FAISS index...")

        # Check that index dimension matches embedding dimension
        if index.d != embeddings.shape[1]:
            raise ValueError("Index dimension does not match embedding dimension")

        # Check that index contains all vectors
        if index.ntotal != embeddings.shape[0]:
            raise ValueError("Index size does not match number of embeddings")

        # Run a simple nearest‑neighbor query to confirm the index responds
        distances, ids = index.search(embeddings[0:1], 5)
        if ids is None or ids.size == 0:
            raise RuntimeError("FAISS search returned no results")

        logger.info("FAISS index verification passed")

    def write_index(self, index: faiss.Index) -> None:
        """Write the FAISS index to disk."""
        index_path = Path(self.paths["faiss_index"])
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(index_path))
        logger.info(f"FAISS index written to {index_path}")

    def run(self):
        """Execute the full FAISS index build process."""
        embeddings = self.load_embeddings()
        mapping = self.load_mapping()

        embeddings, mapping_sorted = self.align_embeddings_with_mapping(embeddings, mapping)

        mapping_path = Path(self.paths["mapping_parquet"])
        mapping_sorted.to_parquet(mapping_path, index=False)
        logger.info("Mapping saved in aligned order for retrieval pipeline")

        # Build and save FAISS index
        index = self.build_index(embeddings)
        self.verify_index(index, embeddings)
        self.write_index(index)


if __name__ == "__main__":
    builder = FaissIndexBuilder(Path("configs/config_agents.yaml"))
    builder.run()
