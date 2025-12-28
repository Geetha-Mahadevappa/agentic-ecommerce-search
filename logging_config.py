"""
Central logging setup for this project.

This module defines a single logging configuration that can be shared
across the embedding pipeline and the agent runtime. Each module should
import and call `setup_logging()` once at startup.
"""

from logging.config import dictConfig
from pathlib import Path


def setup_logging(log_file: str = "logs/system.log"):
    """
    Configure logging.

    Logs are written to both the console and a file. The file path can be
    overridden by callers if they want separate logs for embeddings or agents.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)s %(name)s: %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": str(log_path),
                "formatter": "default",
                "level": "INFO",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    }

    dictConfig(config)
