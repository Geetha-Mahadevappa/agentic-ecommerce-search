#!/usr/bin/env python3
"""
FastAPI service that exposes the multi‑agent product search pipeline.
The pipeline is built once at startup and reused for all incoming
requests to ensure low latency and consistent behavior.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from search_orchestration import build_search_pipeline, run_search


app = FastAPI(title="Product Search API")


# Build the pipeline once during application startup
pipeline = build_search_pipeline("configs/config_agents.yaml")


class SearchRequest(BaseModel):
    """
    Request body for the /search endpoint.
    """
    query: str


@app.post("/search")
def search(req: SearchRequest):
    """
    Run the full search pipeline for the provided query string.

    Parameters
    ----------
    req : SearchRequest
        The incoming request containing the user's search query.

    Returns
    -------
    dict
        The ranked search results produced by the multi‑agent pipeline.
    """
    return run_search(pipeline, req.query)


@app.get("/health")
def health():
    """
    Check used by container orchestration systems
    to verify that the service is running.
    """
    return {"status": "ok"}

@app.get("/ready")
def ready():
    """
    Verifies that the search pipeline has been fully initialized and is ready to serve traffic.
    """
    is_ready = pipeline is not None
    return {"ready": is_ready}
