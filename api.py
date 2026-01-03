#!/usr/bin/env python3
"""
FastAPI application that exposes the product search pipeline.

The pipeline is initialized once during startup so that all incoming
requests share the same in‑memory components. This avoids repeated
model loading and keeps latency predictable.
"""

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from search_orchestration import build_search_pipeline, run_search

# Initialize logging
from logging_config import setup_logging
setup_logging("logs/agent_runtime.log")
logger = logging.getLogger(__name__)

app = FastAPI(title="Product Search API")

# These are populated during startup
pipeline = None
pipeline_ready = False


class SearchRequest(BaseModel):
    """Request body for the /search endpoint."""
    query: str


@app.on_event("startup")
def initialize_pipeline() -> None:
    """
    Build the search pipeline once at application startup.

    This ensures that heavy components such as embedding models,
    FAISS indexes, and LLM clients are loaded a single time.
    """
    global pipeline, pipeline_ready

    logger.info("Starting pipeline initialization...")

    try:
        pipeline = build_search_pipeline("configs/config_agents.yaml")
        pipeline_ready = True
        logger.info("Pipeline initialized successfully.")

    except Exception:
        pipeline_ready = False
        logger.exception("Pipeline failed to initialize.")
        raise


@app.post("/search")
def search(request: SearchRequest):
    """
    Execute the search pipeline for the provided query.
    """
    if not pipeline_ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    try:
        result = run_search(pipeline, request.query)

        # Handle empty results cleanly
        if not result.get("results"):
            logger.info("No results found for query: %s", request.query)
        return result

    except Exception:
        logger.exception("Search pipeline error")
        raise HTTPException(status_code=500, detail="Internal search error")


@app.get("/health")
def health() -> dict:
    """
    Liveness probe used by container orchestration systems.
    This endpoint only confirms that the process is running.
    """
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """
    Readiness probe indicating whether the pipeline has finished
    initializing and is ready to serve traffic.
    """
    return {"ready": pipeline_ready}
