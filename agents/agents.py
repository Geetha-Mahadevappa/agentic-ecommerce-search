#!/usr/bin/env python3
"""
Agents for the multi‑agent search pipeline.
Each agent handles one focused step in the workflow.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

import faiss
import logging
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from logging_config import setup_logging
setup_logging("logs/agent_runtime.log")
logger = logging.getLogger(__name__)


# Query understanding
@dataclass
class QueryUnderstandingAgent:
    """
    Normalizes the query and applies synonym expansion, category hints,
    and product-type inference using FlashText and Vector Classification fallback.
    """

    procedural_memory: Dict[str, Any]
    embedding_model: SentenceTransformer
    keyword_processor: Any
    product_types: List[str] = None
    product_type_index: Any = None

    def __post_init__(self):
        self.synonyms = self.procedural_memory.get("synonyms", {})
        self.category_keywords = self.procedural_memory.get("category_keywords", [])

        logger.info(
            f"QueryUnderstandingAgent initialized with "
            f"{len(self.synonyms)} synonyms and "
            f"{len(self.category_keywords)} category keywords"
        )

    def _normalize(self, text: str) -> str:
        cleaned = text.lower().strip()
        logger.debug(f"Normalized query: {text}, {cleaned}")
        return cleaned

    def _apply_synonyms(self, text: str) -> str:
        words = text.split()
        expanded = []

        for w in words:
            syn_list = self.synonyms.get(w)
            if isinstance(syn_list, list):
                expanded.extend(syn_list)
            else:
                expanded.append(w)

        mapped_text = " ".join(expanded)
        if mapped_text != text:
            logger.info(f"Applied synonyms: {text}, {mapped_text}")
        return mapped_text

    def _infer_category(self, text: str) -> str:
        for kw in self.category_keywords:
            if kw in text:
                logger.info(f"Inferred category {kw} from query {text}")
                return kw
        logger.info(f"No category inferred for query {text}")
        return "unknown"

    def _infer_product_type_vector(self, text: str) -> str | None:
        if self.product_type_index is None or not self.product_types:
            return None

        try:
            q_emb = self.embedding_model.encode(text, convert_to_numpy=True).astype("float32")
            q_emb = q_emb.reshape(1, -1)

            scores, idxs = self.product_type_index.search(q_emb, 1)
            best_idx = idxs[0][0]

            if best_idx < 0:
                return None

            pt = self.product_types[best_idx]
            logger.info(f"Inferred product_type '{pt}' via vector classifier")
            return pt

        except Exception as e:
            logger.error(f"Vector product-type inference failed: {e}")
            return None

    def _infer_product_type(self, text: str) -> str | None:
        # FlashText exact match
        matches = self.keyword_processor.extract_keywords(text)
        if matches:
            pt = max(matches, key=len)
            logger.info(f"Inferred product_type '{pt}' via FlashText")
            return pt

        # Vector fallback
        return self._infer_product_type_vector(text)

    def _infer_price_intent(self, text: str) -> tuple[bool, float | None]:
        has_price_words = any(w in text for w in ["cheap", "low price", "budget", "under", "<", "<="])
        max_price = None

        tokens = text.replace("$", " ").replace("€", " ").split()
        for t in tokens:
            try:
                val = float(t)
                max_price = val
                has_price_words = True
                break
            except ValueError:
                continue

        return has_price_words, max_price

    def run(self, raw_query: str) -> Dict[str, Any]:
        logger.info(f"Running QueryUnderstandingAgent on query: {raw_query}")

        cleaned = self._normalize(raw_query)
        cleaned = self._apply_synonyms(cleaned)

        category = self._infer_category(cleaned)
        product_type = self._infer_product_type(cleaned)
        has_price_constraint, max_price = self._infer_price_intent(cleaned)

        return {
            "raw_query": raw_query,
            "clean_query": cleaned,
            "category": category,
            "product_type": product_type,
            "has_price_constraint": has_price_constraint,
            "max_price": max_price
        }


# BM25 retrieval
@dataclass
class BM25RetrievalAgent:
    """BM25 retrieval over canonical texts."""

    corpus: List[str]
    mapping: pd.DataFrame
    top_k: int = 50

    def __post_init__(self):
        logger.info(f"Initializing BM25RetrievalAgent with {len(self.corpus)} documents")
        try:
            tokenized = [doc.lower().split() for doc in self.corpus]
            self._bm25 = BM25Okapi(tokenized)
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
            raise

    def run(self, query: str, allowed_ids: set[str] | None = None):
        """Run BM25 retrieval and optionally filter by allowed variant IDs."""
        logger.info(f"Running BM25 retrieval for query: {query}")

        try:
            tokens = query.lower().split()
            scores = self._bm25.get_scores(tokens)
        except Exception as e:
            logger.error(f"BM25 scoring failed for query {query}: {e}")
            return []

        idxs = scores.argsort()[-self.top_k:][::-1]
        logger.info(f"BM25 returned {len(idxs)} candidates")

        out: List[Dict[str, Any]] = []
        for idx in idxs:
            row = self.mapping.iloc[idx]
            vid = str(row["variant_id"]).strip()

            if allowed_ids is not None and vid not in allowed_ids:
                continue

            out.append({"variant_id": vid, "score": float(scores[idx])})
        return out


# Hybrid retrieval (FAISS + BM25)
@dataclass
class HybridRetrievalAgent:
    """Combines FAISS semantic search with BM25 lexical search and applies product-type filtering."""

    model_name: str
    device: str
    faiss_index: faiss.Index
    mapping: pd.DataFrame
    metadata: pd.DataFrame
    bm25_agent: BM25RetrievalAgent
    top_k: int = 50
    semantic_weight: float = 0.6
    bm25_weight: float = 0.4

    def __post_init__(self):
        logger.info(f"Initializing HybridRetrievalAgent with model {self.model_name}")
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise

        # Normalize variant_id for consistent lookup
        self.mapping["variant_id"] = self.mapping["variant_id"].astype(str).str.strip()
        self.metadata["variant_id"] = self.metadata["variant_id"].astype(str).str.strip()

    def _semantic_search(self, query: str, allowed_ids: set[str] | None = None):
        logger.info(f"Running FAISS semantic search for query: {query}")
        try:
            q_emb = self._model.encode(query, convert_to_numpy=True).astype("float32")
            q_emb = q_emb.reshape(1, -1)
            scores, idxs = self.faiss_index.search(q_emb, self.top_k)
        except Exception as e:
            logger.error(f"FAISS search failed for query {query}: {e}")
            return []

        out: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], idxs[0]):
            row = self.mapping.iloc[idx]
            vid = str(row["variant_id"]).strip()

            if allowed_ids is not None and vid not in allowed_ids:
                continue

            out.append({"variant_id": vid, "score": float(score)})

        logger.info(f"FAISS returned {len(out)} candidates")
        return out

    def run(self, query_info: Dict[str, Any], allowed_ids: set[str] | None = None) -> List[Dict[str, Any]]:
        """Run hybrid retrieval with optional pre-filtered allowed_ids."""
        query = query_info["clean_query"]
        product_type = query_info.get("product_type")

        logger.info(f"Running HybridRetrievalAgent for query: {query}")

        # Semantic retrieval
        sem_results = self._semantic_search(query, allowed_ids=allowed_ids)

        # Lexical retrieval
        bm25_results = self.bm25_agent.run(query, allowed_ids=allowed_ids)

        # Merge scores
        combined: Dict[str, float] = {}
        for r in sem_results:
            vid = r["variant_id"]
            combined.setdefault(vid, 0.0)
            combined[vid] += r["score"] * self.semantic_weight

        for r in bm25_results:
            vid = r["variant_id"]
            combined.setdefault(vid, 0.0)
            combined[vid] += r["score"] * self.bm25_weight

        merged = [{"variant_id": vid, "score": score} for vid, score in combined.items()]
        merged.sort(key=lambda x: x["score"], reverse=True)

        # Apply product-type filtering
        if product_type:
            pt_tokens = product_type.lower().split()
            filtered = []

            for item in merged:
                vid = item["variant_id"]

                if allowed_ids is not None and vid not in allowed_ids:
                    continue

                row = self.metadata[self.metadata["variant_id"] == vid].iloc[0]
                name = str(row["product_name"]).lower()
                category = str(row["category"]).lower()

                text_tokens = f"{name} {category}".split()

                if any(tok in text_tokens for tok in pt_tokens):
                    filtered.append(item)

            if filtered:
                merged = filtered

        merged.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Hybrid retrieval produced {len(merged)} final candidates")
        return merged[: self.top_k]


@dataclass
class RerankerAgent:
    """
    Final ranking step. Enriches with metadata and can
    apply LLM-based reranking on the top candidates.
    """

    engine: Any
    memory_agent: Any
    llm_client: Any
    final_top_k: int = 10

    def _fetch_metadata(self, variant_id: str) -> Dict[str, Any]:
        logger.debug(f"Fetching metadata for variant_id {variant_id}")
        query = text(
            """
            SELECT variant_id, product_id, product_name, category, price
            FROM chunks
            WHERE variant_id = :vid
            LIMIT 1
            """
        )
        try:
            with self.engine.begin() as conn:
                row = conn.execute(query, {"vid": variant_id}).fetchone()
                return dict(row._mapping) if row else {}
        except Exception as e:
            logger.error(f"Metadata fetch failed for variant_id {variant_id}: {e}")
            return {}

    def _apply_metadata(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Applying metadata to {len(candidates)} candidates")
        enriched: List[Dict[str, Any]] = []
        for c in candidates:
            meta = self._fetch_metadata(c["variant_id"])
            if not meta:
                logger.warning("No metadata found for variant_id '%s'", c["variant_id"])
                continue
            enriched.append({**c, **meta})
        return enriched

    def _llm_rerank(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Running LLM reranking on {len(items)} items.")
        try:
            prompt_lines = [
                "Reorder these products from best to worst match for the query.",
                f"Query: {query}",
                "Products:",
            ]
            for i, item in enumerate(items, start=1):
                prompt_lines.append(
                    f"{i}. id={item['variant_id']}, "
                    f"name={item['product_name']}, "
                    f"category={item['category']}, "
                    f"price={item['price']}"
                )
            prompt_lines.append(
                "Return a comma-separated list of variant_ids in the new order."
            )
            prompt = "\n".join(prompt_lines)

            text_response = self.llm_client.generate(prompt, max_tokens=128).strip()
            variant_ids = [p.strip() for p in text_response.split(",") if p.strip()]

            order = {vid: idx for idx, vid in enumerate(variant_ids)}
            items.sort(key=lambda x: order.get(x["variant_id"], len(items)))
            return items

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}.")
            return items

    def run(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Running RerankerAgent for query {query} with {len(candidates)} candidates.")

        enriched = self._apply_metadata(candidates)
        enriched.sort(key=lambda x: x["score"], reverse=True)
        top = enriched[: self.final_top_k]

        if self.llm_client is not None and top:
            try:
                top = self._llm_rerank(query, top)
            except Exception as e:
                logger.error(f"LLM rerank crashed: {e}. Falling back to score ranking.")
                top.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"RerankerAgent returning {len(top)} final results.")
        return top
