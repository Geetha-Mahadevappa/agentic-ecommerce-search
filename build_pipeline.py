#!/usr/bin/env python3

"""
Runs the two offline preprocessing steps in order:
1. Generate product embeddings.
2. Build the FAISS index.

If the embedding step fails, the process stops immediately.
"""

import logging
from pathlib import Path
from embeddings_pipeline.embed_products import EmbeddingPipeline
from embeddings_pipeline.build_faiss_index import FaissIndexBuilder

from logging_config import setup_logging
setup_logging("logs/embeddings.log")
logger = logging.getLogger(__name__)

def main(embedding_config, faiss_config):
    # Embeddings
    try:
        EmbeddingPipeline(Path(embedding_config)).run()
    except Exception as e:
        logger.error(f"Embedding pipeline failed: {e}")
        raise RuntimeError(f"Embedding pipeline failed {e}")

    # FAISS index
    try:
        FaissIndexBuilder(Path(faiss_config)).run()
    except Exception as e:
        logger.error(f"FAISS index build failed: {e}")
        raise RuntimeError(f"FAISS index build failed: {e}")


if __name__ == "__main__":
    embedding_config = "configs/config_embedding.yaml"
    faiss_config = "configs/config_agents.yaml"
    main(embedding_config, faiss_config)
