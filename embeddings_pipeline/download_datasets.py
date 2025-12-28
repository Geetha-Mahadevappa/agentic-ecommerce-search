#!/usr/bin/env python3

"""
Utility script to download the Kaggle dataset and copy it into a target directory.

This script uses kagglehub to fetch the dataset into the local cache, then copies
the CSV files into whatever folder the caller specifies. It does not assume any
fixed directory structure, so it can be reused from other scripts.
"""

import logging
import shutil
from pathlib import Path
import kagglehub

from logging_config import setup_logging
setup_logging("logs/embeddings.log")
logger = logging.getLogger(__name__)


def download_and_copy(target_dir: Path, dataset: str="pruthvirajgshitole/e-commerce-purchases-and-reviews") -> None:
    """Download the Kaggle dataset and copy the CSV files into the given folder."""

    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset: {dataset}")
    cache_path = Path(kagglehub.dataset_download(dataset))
    logger.info(f"Download complete. Cached at: {cache_path}")

    for f in cache_path.iterdir():
        if f.is_file():
            dest = target_dir / f.name
            shutil.copy(f, dest)
            logger.info(f"Copied {f.name} -> {dest}")

    logger.info(f"Dataset copied into {target_dir.resolve()}")
