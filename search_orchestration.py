#!/usr/bin/env python3
"""
Orchestration layer for the multi‑agent search pipeline.

Loads configuration and resources, builds agents, and exposes
a run_search(...) function that executes the full workflow.
"""

import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List

import yaml
import faiss
import pandas as pd
import sqlalchemy as sa
from sentence_transformers import SentenceTransformer

from agents.agents import QueryUnderstandingAgent
from agents.agents import BM25RetrievalAgent, HybridRetrievalAgent, RerankerAgent
from agents.memory_agent import MemoryAgent
from llm.llm_client import LLMClient


# Config dataclasses
@dataclass
class PathsConfig:
    embeddings_npz: str
    mapping_parquet: str
    metadata_db: str
    faiss_index: str
    memory_dir: str
    user_prefs_json: str
    procedural_memory_yaml: str
    activity_log_json: str


@dataclass
class ModelConfig:
    name: str
    device: str  # "cpu" or "cuda"


@dataclass
class LLMConfig:
    model_name: str
    device: str
    max_new_tokens: int


@dataclass
class SearchConfig:
    faiss_top_k: int
    final_top_k: int
    short_term_history: int
    recent_activity_days: int


@dataclass
class AppConfig:
    paths: PathsConfig
    model: ModelConfig
    llm: LLMConfig
    search: SearchConfig


# Config loader
def load_config(path: str) -> AppConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        paths=PathsConfig(**raw["paths"]),
        model=ModelConfig(**raw["model"]),
        llm=LLMConfig(**raw["llm"]),
        search=SearchConfig(**raw["search"]),
    )


# Resource loaders
def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

def load_mapping(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def load_chunk_texts(db_path: str, mapping: pd.DataFrame) -> List[str]:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT product_id, text FROM chunks", conn)
    conn.close()

    df["product_id"] = df["product_id"].astype(str)
    mapping["product_id"] = mapping["product_id"].astype(str)

    # Collapse duplicates: one canonical text per product
    df = df.groupby("product_id")["text"].apply(lambda x: " ".join(x)).reset_index()

    # Align to mapping order
    df = mapping.merge(df, on="product_id", how="left")
    return df["text"].tolist()

def create_engine(db_path: str) -> sa.Engine:
    return sa.create_engine(f"sqlite:///{db_path}", future=True)


# Pipeline builder
@dataclass
class SearchPipeline:
    config: AppConfig
    memory_agent: MemoryAgent
    query_agent: QueryUnderstandingAgent
    retrieval_agent: HybridRetrievalAgent
    reranker: RerankerAgent


def build_search_pipeline(
    config_path: str = "configs/config_agents.yaml",
) -> SearchPipeline:
    config = load_config(config_path)

    # Load resources
    mapping_df = load_mapping(config.paths.mapping_parquet)
    mapping_df["product_id"] = mapping_df["product_id"].astype(str)
    faiss_index = load_faiss_index(config.paths.faiss_index)
    chunk_texts = load_chunk_texts(config.paths.metadata_db, mapping_df)
    db_engine = create_engine(config.paths.metadata_db)

    # Memory subsystem
    memory_agent = MemoryAgent(
        memory_dir=Path(config.paths.memory_dir),
        user_prefs_path=Path(config.paths.user_prefs_json),
        procedural_memory_path=Path(config.paths.procedural_memory_yaml),
        activity_log_path=Path(config.paths.activity_log_json),
        short_term_limit=config.search.short_term_history,
    )

    # Embedding model
    embedding_model = SentenceTransformer(
        config.model.name,
        device=config.model.device,
    )
    _ = embedding_model.encode(["warmup query"], convert_to_numpy=True)

    # LLM client (used for reranking)
    llm_client = LLMClient(
        model_name=config.llm.model_name,
        device=config.llm.device,
        max_new_tokens=config.llm.max_new_tokens
    )
    _ = llm_client.generate("warmup", max_tokens=16)

    # Agents
    query_agent = QueryUnderstandingAgent(
        procedural_memory=memory_agent.procedural_memory,
        embedding_model=embedding_model,
    )

    bm25_agent = BM25RetrievalAgent(
        corpus=chunk_texts,
        mapping=mapping_df,
        top_k=config.search.faiss_top_k,
    )

    hybrid_agent = HybridRetrievalAgent(
        model_name=config.model.name,
        device=config.model.device,
        faiss_index=faiss_index,
        mapping=mapping_df,
        bm25_agent=bm25_agent,
        top_k=config.search.faiss_top_k,
    )

    reranker = RerankerAgent(
        engine=db_engine,
        memory_agent=memory_agent,
        llm_client=llm_client,
        final_top_k=config.search.final_top_k,
    )

    return SearchPipeline(
        config=config,
        memory_agent=memory_agent,
        query_agent=query_agent,
        retrieval_agent=hybrid_agent,
        reranker=reranker,
    )


# Public search entrypoint
def run_search(pipeline: SearchPipeline, raw_query: str) -> Dict[str, Any]:
    # Parse query
    q = pipeline.query_agent.run(raw_query)
    clean_query = q["clean_query"]

    # Memory update from query
    pipeline.memory_agent.update_preferences_from_query(clean_query)

    # Retrieval
    candidates = pipeline.retrieval_agent.run(clean_query)

    # Final reranking (with metadata and LLM rerank)
    final_results = pipeline.reranker.run(clean_query, candidates)
    final_results.sort(key=lambda x: x["score"], reverse=True)

    # Memory updates
    pipeline.memory_agent.add_short_term(clean_query, final_results)
    pipeline.memory_agent.log_activity(
        clean_query,
        [r["product_id"] for r in final_results],
    )
    recent_activity = pipeline.memory_agent.get_recent_activity(
        days=pipeline.config.search.recent_activity_days
    )

    return {
        "query": q,
        "results": final_results,
        "recent_activity": recent_activity,
    }
