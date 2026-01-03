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
from flashtext2 import KeywordProcessor

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
    user_infer_json: str
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
    df = pd.read_sql("SELECT variant_id, text FROM chunks", conn)
    conn.close()

    df["variant_id"] = df["variant_id"].astype(str)
    mapping["variant_id"] = mapping["variant_id"].astype(str)

    # No collapsing — one canonical text per variant
    df = mapping.merge(df, on="variant_id", how="left")
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


def build_search_pipeline(config_path: str = "configs/config_agents.yaml",) -> SearchPipeline:
    config = load_config(config_path)

    # Load resources
    mapping_df = load_mapping(config.paths.mapping_parquet)
    mapping_df["variant_id"] = mapping_df["variant_id"].astype(str)
    faiss_index = load_faiss_index(config.paths.faiss_index)
    chunk_texts = load_chunk_texts(config.paths.metadata_db, mapping_df)
    db_engine = create_engine(config.paths.metadata_db)

    metadata_df = pd.read_sql(
        "SELECT variant_id, product_id, product_name, category, country, price, text FROM chunks",
        db_engine
    )
    metadata_df["variant_id"] = metadata_df["variant_id"].astype(str)

    # Memory subsystem
    memory_agent = MemoryAgent(
        memory_dir=Path(config.paths.memory_dir),
        user_prefs_path=Path(config.paths.user_prefs_json),
        user_infer_path=Path(config.paths.user_infer_json),
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
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for _, row in metadata_df.iterrows():
        keyword_processor.add_keyword(row["product_name"])
        keyword_processor.add_keyword(row["category"])

    product_types = (
        metadata_df["product_name"].str.lower().tolist() +
        metadata_df["category"].str.lower().tolist()
    )
    product_types = list(set(product_types))
    pt_embeddings = embedding_model.encode(product_types, convert_to_numpy=True).astype("float32")

    d = pt_embeddings.shape[1]
    pt_index = faiss.IndexFlatL2(d)
    pt_index.add(pt_embeddings)

    countries = (
        metadata_df["country"]
        .dropna()
        .str.lower()
        .unique()
        .tolist()
    )

    query_agent = QueryUnderstandingAgent(
        procedural_memory=memory_agent.procedural_memory,
        embedding_model=embedding_model,
        keyword_processor=keyword_processor,
        memory_agent=memory_agent,
        product_types=product_types,
        product_type_index=pt_index,
        countries=countries
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
        metadata=metadata_df,
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
    """
    End-to-end search orchestration. Handles query parsing, memory updates,
    country and price filtering, hybrid retrieval, and final reranking.
    """

    # Query Understanding
    q = pipeline.query_agent.run(raw_query)
    clean_query = q["clean_query"]

    # Update long-term preferences (categories, price sensitivity)
    pipeline.memory_agent.update_preferences_from_query(clean_query, q["product_type"])

    # Update inferred country memory
    country = q.get("country")
    if country:
        pipeline.memory_agent.update_inferred({"inferred_country": country})
        pipeline.memory_agent.add_candidate_country(country)

    # Build allowed_ids (Hard Filters)
    allowed_ids = None

    # Country filter
    if q.get("country"):
        country = q["country"]
        allowed_ids = set(
            pipeline.retrieval_agent.metadata[
                pipeline.retrieval_agent.metadata["country"] == country
            ]["variant_id"].astype(str)
        )

    # Numeric price constraint (combine with country if both exist)
    if q["has_price_constraint"] and q["max_price"] is not None:
        price_filtered = set(
            pipeline.retrieval_agent.metadata[
                pipeline.retrieval_agent.metadata["price"] <= q["max_price"]
            ]["variant_id"].astype(str)
        )
        allowed_ids = price_filtered if allowed_ids is None else allowed_ids & price_filtered

    # Hybrid Retrieval
    candidates = pipeline.retrieval_agent.run(q, allowed_ids=allowed_ids)

    # Pass full query_info so reranker can use price_level intent
    final_results = pipeline.reranker.run(q, candidates)
    final_results.sort(key=lambda x: x["score"], reverse=True)

    # Memory Updates
    pipeline.memory_agent.add_short_term(clean_query, final_results)
    pipeline.memory_agent.log_activity(clean_query, [r["variant_id"] for r in final_results])

    return {
        "query": q,
        "results": final_results
    }
