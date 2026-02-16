#!/usr/bin/env python3

"""
Runs the two offline preprocessing steps in order:
1. Generate product embeddings.
2. Build the Qdrant collection.
"""

import logging
from pathlib import Path
from embeddings_pipeline.embed_products import EmbeddingPipeline
from embeddings_pipeline.build_qdrant_index import QdrantIndexBuilder

from logging_config import setup_logging
setup_logging("logs/embeddings.log")
logger = logging.getLogger(__name__)


def main(embedding_config: str, retrieval_config: str) -> None:
    try:
        EmbeddingPipeline(Path(embedding_config)).run()
    except Exception as e:
        logger.error(f"Embedding pipeline failed: {e}")
        raise RuntimeError(f"Embedding pipeline failed {e}")

    try:
        QdrantIndexBuilder(Path(retrieval_config)).run()
    except Exception as e:
        logger.error(f"Qdrant index build failed: {e}")
        raise RuntimeError(f"Qdrant index build failed: {e}")


if __name__ == "__main__":
    embedding_config = "configs/config_embedding.yaml"
    retrieval_config = "configs/config_agents.yaml"
    main(embedding_config, retrieval_config)
