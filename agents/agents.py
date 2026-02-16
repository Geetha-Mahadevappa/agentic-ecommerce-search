#!/usr/bin/env python3
"""
Agents for the multi‑agent search pipeline.
Each agent handles one focused step in the workflow.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda

import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import text, bindparam

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
    memory_agent: Any
    product_types: List[str] = None
    product_type_index: Any = None
    countries: List[str] = None

    def __post_init__(self):
        self.synonyms = self.procedural_memory.get("synonyms", {})
        self.category_keywords = self.procedural_memory.get("category_keywords", [])
        self.supported_countries = [
            c.lower() for c in self.procedural_memory.get("supported_countries", [])
        ]

        # Load currency intent rules from procedural memory
        cur_rules = self.procedural_memory.get("currency_rules", {})
        self.currency_symbols = cur_rules.get("symbols", [])

        # Load price intent rules from procedural memory
        rules = self.procedural_memory.get("query_rules", {})
        self.low_price_intent = rules.get("low_price_intent", [])
        self.mid_price_intent = rules.get("mid_price_intent", [])
        self.high_price_intent = rules.get("high_price_intent", [])
        self.numeric_price_triggers = rules.get("numeric_price_triggers", [])

        logger.info(
            f"QueryUnderstandingAgent initialized with "
            f"{len(self.synonyms)} synonyms and "
            f"{len(self.category_keywords)} category keywords"
        )

    def _normalize(self, text: str) -> str:
        cleaned = text.lower().strip()
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
            q_norm = np.linalg.norm(q_emb) + 1e-12
            cand = self.product_type_index
            scores = (cand @ q_emb) / ((np.linalg.norm(cand, axis=1) + 1e-12) * q_norm)
            best_idx = int(np.argmax(scores))
            if best_idx < 0:
                return None

            pt = self.product_types[best_idx]
            logger.info(f"Inferred product_type '{pt}' via vector classifier")
            return pt

        except Exception as e:
            logger.error(f"Vector product-type inference failed: {e}")
            return None

    def _infer_product_type(self, text: str) -> str | None:
        matches = self.keyword_processor.extract_keywords(text)
        if matches:
            pt = max(matches, key=len)
            logger.info(f"Inferred product_type '{pt}' via FlashText")
            return pt

        return self._infer_product_type_vector(text)

    def _infer_price_intent(self, text: str) -> tuple[bool, float | None, str | None]:
        text = text.lower().replace("-", " ")

        has_price_words = False
        price_level = None
        max_price = None

        # Detect price-level words
        if any(key in text for key in self.low_price_intent):
            price_level = "Low"
            has_price_words = True
        elif any(key in text for key in self.high_price_intent):
            price_level = "High"
            has_price_words = True
        elif any(key in text for key in self.mid_price_intent):
            price_level = "Mid"
            has_price_words = True

        # Detect numeric price triggers
        trigger_found = None
        for trig in self.numeric_price_triggers:
            if trig in text:
                trigger_found = trig
                has_price_words = True
                break

        # Only extract numeric price if a trigger is present
        if trigger_found:
            cleaned = text
            for sym in self.currency_symbols:
                cleaned = cleaned.replace(sym, " ")


            tokens = cleaned.split()
            for i, tok in enumerate(tokens):
                if tok == trigger_found and i + 1 < len(tokens):
                    try:
                        max_price = float(tokens[i + 1])
                    except ValueError:
                        pass
                    break
        return has_price_words, max_price, price_level

    def _infer_country(self, cleaned: str) -> Optional[str]:
        if not self.countries:
            return None

        for c in self.countries:
            if c in cleaned:
                return c
        return None

    def run(self, raw_query: str) -> Dict[str, Any]:
        logger.info(f"Running QueryUnderstandingAgent on query: {raw_query}")

        cleaned = self._normalize(raw_query)
        cleaned = self._apply_synonyms(cleaned)

        category = self._infer_category(cleaned)
        product_type = self._infer_product_type(cleaned)
        has_price_constraint, max_price, price_level = self._infer_price_intent(cleaned)
        country = self._infer_country(cleaned)

        # If user did not specify price intent, fall back to memory
        if not has_price_constraint:
            inferred_data = self.memory_agent.get_inferred()
            mem_price = inferred_data.get("price_sensitivity")
            if mem_price:
                price_level = mem_price

        # If user did not specify a country, fall back to inferred/preferred/candidate
        if country is None:
            inferred_data = self.memory_agent.get_inferred()
            fallback = (
                inferred_data.get("preferred_country")
                or inferred_data.get("inferred_country")
                or (inferred_data.get("candidate_countries") or [None])[0]
            )
            if fallback:
                country = fallback.title()

        country = country.title() if country else None
        is_vague = (product_type is None and category == "unknown")

        return {
            "raw_query": raw_query,
            "clean_query": cleaned,
            "category": category,
            "product_type": product_type,
            "country": country,
            "price_level": price_level,
            "has_price_constraint": has_price_constraint,
            "max_price": max_price,
            "is_vague": is_vague
        }


# Vector retrieval (Qdrant + LangChain)
@dataclass
class QdrantRetrievalAgent:
    """
    Semantic retrieval using Qdrant as the vector backend and LangChain
    for vector-store orchestration.
    """

    model_name: str
    device: str
    qdrant_client: QdrantClient
    collection_name: str
    metadata: pd.DataFrame
    top_k: int = 50

    def __post_init__(self):
        logger.info("Initializing QdrantRetrievalAgent with model %s", self.model_name)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self._embeddings,
            content_payload_key="text",
        )

        self.metadata["variant_id"] = self.metadata["variant_id"].astype(str).str.strip()
        self.metadata = self.metadata.set_index("variant_id", drop=False)

    def _make_filter(self, allowed_ids: set[str] | None) -> Filter | None:
        if not allowed_ids:
            return None
        return Filter(
            must=[
                FieldCondition(
                    key="variant_id",
                    match=MatchAny(any=[str(v) for v in sorted(allowed_ids)]),
                )
            ]
        )

    def _semantic_search(self, query: str, allowed_ids: set[str] | None = None) -> List[Dict[str, Any]]:
        q_filter = self._make_filter(allowed_ids)
        docs_and_scores = self._vector_store.similarity_search_with_score(
            query=query,
            k=self.top_k,
            filter=q_filter,
        )

        out: List[Dict[str, Any]] = []
        for doc, score in docs_and_scores:
            variant_id = str(doc.metadata.get("variant_id", "")).strip()
            if not variant_id:
                continue
            sim_score = 1.0 / (1.0 + float(score))
            out.append({"variant_id": variant_id, "score": sim_score})

        logger.info("Qdrant returned %d candidates", len(out))
        return out

    def run(self, query_info: Dict[str, Any], allowed_ids: set[str] | None = None) -> List[Dict[str, Any]]:
        query = query_info["clean_query"]
        product_type = query_info.get("product_type")

        retrieval_chain = RunnableLambda(
            lambda inp: self._semantic_search(inp["query"], inp.get("allowed_ids"))
        )

        merged = retrieval_chain.invoke({"query": query, "allowed_ids": allowed_ids})

        if product_type:
            pt_tokens = product_type.lower().split()
            filtered = []
            for item in merged:
                vid = item["variant_id"]
                try:
                    row = self.metadata.loc[vid]
                except KeyError:
                    continue

                name = str(row["product_name"]).lower()
                category = str(row["category"]).lower()
                text_tokens = f"{name} {category}".split()
                if any(tok in text_tokens for tok in pt_tokens):
                    filtered.append(item)
            merged = filtered

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[: self.top_k]


# Re-ranking
@dataclass
class RerankerAgent:
    """
    Final ranking step. Enriches candidates with metadata and optionally
    applies LLM-based reranking. Price level intent is applied as a soft
    semantic boost before LLM reranking. Metadata is fetched in a single
    batch query to avoid N+1 performance issues.
    """

    engine: Any
    memory_agent: Any
    llm_client: Any
    final_top_k: int = 10
    score_cutoff_ratio: float = 0.50   # NEW: configurable score cutoff

    def _fetch_metadata_batch(self, variant_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch metadata for all variant_ids in a single SQL query.
        Avoids the N+1 query pattern and significantly improves performance.
        """
        if not variant_ids:
            return {}

        logger.info(f"Batch fetching metadata for {len(variant_ids)} variants")

        query = text(
            """ 
                SELECT variant_id, product_id, product_name, category, price_level, price
                FROM chunks
                WHERE variant_id IN :vids 
            """
        )
        query = query.bindparams(bindparam("vids", expanding=True))

        try:
            with self.engine.begin() as conn:
                rows = conn.execute(query, {"vids": tuple(variant_ids)}).fetchall()
                return {row.variant_id: dict(row._mapping) for row in rows}
        except Exception as e:
            logger.error(f"Batch metadata fetch failed: {e}")
            return {}

    def _apply_metadata(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Applying metadata to {len(candidates)} candidates")

        variant_ids = [c["variant_id"] for c in candidates]
        meta_map = self._fetch_metadata_batch(variant_ids)

        enriched: List[Dict[str, Any]] = []
        for c in candidates:
            meta = meta_map.get(c["variant_id"])
            if not meta:
                logger.warning("No metadata found for variant_id '%s'", c["variant_id"])
                continue
            enriched.append({**c, **meta})

        return enriched

    def _apply_price_level_boost(self, items, user_price_level):
        logger.info(f"Applying price-level boost: {user_price_level}")

        if not user_price_level:
            return

        user_price_level = user_price_level.lower()
        for item in items:
            product_level = item.get("price_level")
            if not product_level:
                continue

            product_level = product_level.lower()

            # Boost when user intent matches product metadata
            if product_level == user_price_level:
                item["score"] *= 1.15

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
                    f"price_level={item['price_level']}"
                )

            prompt_lines.append("[RESULT START]")
            prompt_lines.append("Return only a comma-separated list of variant_ids.")
            prompt_lines.append("[RESULT END]")

            prompt = "\n".join(prompt_lines)

            text_response = self.llm_client.generate(prompt, max_tokens=self.llm_client.max_new_tokens).strip()

            start = text_response.find("[RESULT START]")
            end = text_response.find("[RESULT END]")

            if start != -1 and end != -1:
                content = text_response[start + len("[RESULT START]"):end].strip()
            else:
                content = text_response

            variant_ids = [p.strip() for p in content.split(",") if p.strip()]

            order = {vid: idx for idx, vid in enumerate(variant_ids)}
            items.sort(key=lambda x: order.get(x["variant_id"], len(items)))
            return items

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}.")
            return items

    def _filter_by_price_level(self, items, user_price_level):
        """
        Hard filtering logic:
        - If user asks for High → return High + Mid
        - If user asks for Mid → return Mid only
        - If user asks for Low → return Low only
        """
        if not user_price_level:
            return items

        user_price_level = user_price_level.lower()

        if user_price_level == "high":
            return [i for i in items if i.get("price_level", "").lower() in ("high", "mid")]

        if user_price_level == "mid":
            return [i for i in items if i.get("price_level", "").lower() == "mid"]

        if user_price_level == "low":
            return [i for i in items if i.get("price_level", "").lower() == "low"]

        return items

    def run(self, query_info: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(
            f"Running RerankerAgent for query '{query_info['raw_query']}' "
            f"with {len(candidates)} candidates."
        )

        enriched = self._apply_metadata(candidates)
        enriched.sort(key=lambda x: x["score"], reverse=True)

        # Score cutoff filtering
        if enriched:
            top_score = enriched[0]["score"]
            cutoff = top_score * self.score_cutoff_ratio
            enriched = [item for item in enriched if item["score"] >= cutoff]

        # Query-specific price level takes precedence
        price_level = query_info.get("price_level")

        # Fallback to memory only if query is silent
        if not price_level:
            inferred = self.memory_agent.get_inferred()
            price_level = inferred.get("price_level") or inferred.get("price_sensitivity")

        # Hard filtering
        enriched = self._filter_by_price_level(enriched, price_level)

        # After filtering, take top_k
        top = enriched[: self.final_top_k]

        if price_level:
            self._apply_price_level_boost(top, price_level)

        # LLM reranking
        if self.llm_client is not None and top:
            try:
                top = self._llm_rerank(query_info["raw_query"], top)
            except Exception as e:
                logger.error(f"LLM rerank crashed: {e}. Falling back to score ranking.")
                top.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"RerankerAgent returning {len(top)} final results.")
        return top
