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
    """Normalizes the query and applies simple synonym and category hints."""

    procedural_memory: Dict[str, Any]
    embedding_model: SentenceTransformer

    def __post_init__(self):
        self.synonyms = self.procedural_memory.get("synonyms", {})
        self.category_keywords = self.procedural_memory.get("category_keywords", [])
        logger.info(f"QueryUnderstandingAgent initialized with {len(self.synonyms)} synonyms and {len(self.category_keywords)} category keywords")

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

    def run(self, raw_query: str) -> Dict[str, Any]:
        logger.info(f"Running QueryUnderstandingAgent on query: {raw_query}")
        cleaned = self._normalize(raw_query)
        cleaned = self._apply_synonyms(cleaned)
        category = self._infer_category(cleaned)

        return {
            "raw_query": raw_query,
            "clean_query": cleaned,
            "category": category,
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

    def run(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Running BM25 retrieval for query: {query}", )
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
            try:
                row = self.mapping.iloc[idx]
                out.append({"product_id": row["product_id"], "score": float(scores[idx])})
            except Exception as e:
                logger.error(f"Failed to map BM25 index {idx} to product: {e}")
        return out


# Hybrid retrieval (FAISS + BM25)
@dataclass
class HybridRetrievalAgent:
    """Combines FAISS semantic search with BM25 lexical search."""

    model_name: str
    device: str
    faiss_index: faiss.Index
    mapping: pd.DataFrame
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

    def _semantic_search(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Running FAISS semantic search for query: {query}")
        try:#!/usr/bin/env python3
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
    """Normalizes the query and applies simple synonym and category hints."""

    procedural_memory: Dict[str, Any]
    embedding_model: SentenceTransformer

    def __post_init__(self):
        self.synonyms = self.procedural_memory.get("synonyms", {})
        self.category_keywords = self.procedural_memory.get("category_keywords", [])
        logger.info(f"QueryUnderstandingAgent initialized with {len(self.synonyms)} synonyms and {len(self.category_keywords)} category keywords")

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

    def run(self, raw_query: str) -> Dict[str, Any]:
        logger.info(f"Running QueryUnderstandingAgent on query: {raw_query}")
        cleaned = self._normalize(raw_query)
        cleaned = self._apply_synonyms(cleaned)
        category = self._infer_category(cleaned)

        return {
            "raw_query": raw_query,
            "clean_query": cleaned,
            "category": category,
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

    def run(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Running BM25 retrieval for query: {query}", )
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
            try:
                row = self.mapping.iloc[idx]
                out.append({"product_id": row["product_id"], "score": float(scores[idx])})
            except Exception as e:
                logger.error(f"Failed to map BM25 index {idx} to product: {e}")
        return out


# Hybrid retrieval (FAISS + BM25)
@dataclass
class HybridRetrievalAgent:
    """Combines FAISS semantic search with BM25 lexical search."""

    model_name: str
    device: str
    faiss_index: faiss.Index
    mapping: pd.DataFrame
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

    def _semantic_search(self, query: str) -> List[Dict[str, Any]]:
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
            if idx < 0:
                continue
            try:
                row = self.mapping.iloc[idx]
                out.append({"product_id": row["product_id"], "score": float(score)})
            except Exception as e:
                logger.error(f"Failed to map FAISS index {idx} to product: {e}")
        logger.info(f"FAISS returned {len(out)} candidates")
        return out

    def run(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Running HybridRetrievalAgent for query: {query}")

        sem_results = self._semantic_search(query)
        bm25_results = self.bm25_agent.run(query)

        combined: Dict[str, float] = {}

        for r in sem_results:
            combined.setdefault(r["product_id"], 0.0)
            combined[r["product_id"]] += r["score"] * self.semantic_weight

        for r in bm25_results:
            combined.setdefault(r["product_id"], 0.0)
            combined[r["product_id"]] += r["score"] * self.bm25_weight

        merged = [{"product_id": pid, "score": score} for pid, score in combined.items()]
        merged.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Hybrid retrieval produced {len(merged)} merged candidates")
        return merged


# Reranker with metadata and LLM reranking
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

    def _fetch_metadata(self, product_id: str) -> Dict[str, Any]:
        logger.debug(f"Fetching metadata for product_id {product_id}")
        query = text(
            """
            SELECT product_id, product_name, category, price
            FROM chunks
            WHERE product_id = :pid
            LIMIT 1
            """
        )
        try:
            with self.engine.begin() as conn:
                row = conn.execute(query, {"pid": product_id}).fetchone()
                return dict(row._mapping) if row else {}
        except Exception as e:
            logger.error(f"Metadata fetch failed for product_id {product_id}: {e}")
            return {}

    def _apply_metadata(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Applying metadata to {len(candidates)} candidates")
        enriched: List[Dict[str, Any]] = []
        for c in candidates:
            meta = self._fetch_metadata(c["product_id"])
            if not meta:
                logger.warning("No metadata found for product_id '%s'", c["product_id"])
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
                    f"{i}. id={item['product_id']}, "
                    f"name={item['product_name']}, "
                    f"category={item['category']}, "
                    f"price={item['price']}"
                )
            prompt_lines.append(
                "Return a comma-separated list of product_ids in the new order."
            )
            prompt = "\n".join(prompt_lines)

            text_response = self.llm_client.generate(prompt, max_tokens=128).strip()
            product_ids = [p.strip() for p in text_response.split(",") if p.strip()]

            order = {pid: idx for idx, pid in enumerate(product_ids)}
            items.sort(key=lambda x: order.get(x["product_id"], len(items)))
            return items

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}.")
            return items

    def run(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Running RerankerAgent for query {query} with {len(candidates)} candidates.")

        enriched = self._apply_metadata(candidates)
        enriched.sort(key=lambda x: x["score"], reverse=True)
        top = enriched[: self.final_top_k]#!/usr/bin/env python3
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
    """Normalizes the query and applies simple synonym and category hints."""

    procedural_memory: Dict[str, Any]
    embedding_model: SentenceTransformer

    def __post_init__(self):
        self.synonyms = self.procedural_memory.get("synonyms", {})
        self.category_keywords = self.procedural_memory.get("category_keywords", [])
        logger.info(f"QueryUnderstandingAgent initialized with {len(self.synonyms)} synonyms and {len(self.category_keywords)} category keywords")

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

    def run(self, raw_query: str) -> Dict[str, Any]:
        logger.info(f"Running QueryUnderstandingAgent on query: {raw_query}")
        cleaned = self._normalize(raw_query)
        cleaned = self._apply_synonyms(cleaned)
        category = self._infer_category(cleaned)

        return {
            "raw_query": raw_query,
            "clean_query": cleaned,
            "category": category,
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

    def run(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Running BM25 retrieval for query: {query}", )
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
            try:
                row = self.mapping.iloc[idx]
                out.append({"product_id": row["product_id"], "score": float(scores[idx])})
            except Exception as e:
                logger.error(f"Failed to map BM25 index {idx} to product: {e}")
        return out


# Hybrid retrieval (FAISS + BM25)
@dataclass
class HybridRetrievalAgent:
    """Combines FAISS semantic search with BM25 lexical search."""

    model_name: str
    device: str
    faiss_index: faiss.Index
    mapping: pd.DataFrame
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

    def _semantic_search(self, query: str) -> List[Dict[str, Any]]:
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
            if idx < 0:
                continue
            try:
                row = self.mapping.iloc[idx]
                out.append({"product_id": row["product_id"], "score": float(score)})
            except Exception as e:
                logger.error(f"Failed to map FAISS index {idx} to product: {e}")
        logger.info(f"FAISS returned {len(out)} candidates")
        return out

    def run(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Running HybridRetrievalAgent for query: {query}")

        sem_results = self._semantic_search(query)
        bm25_results = self.bm25_agent.run(query)

        combined: Dict[str, float] = {}

        for r in sem_results:
            combined.setdefault(r["product_id"], 0.0)
            combined[r["product_id"]] += r["score"] * self.semantic_weight

        for r in bm25_results:
            combined.setdefault(r["product_id"], 0.0)
            combined[r["product_id"]] += r["score"] * self.bm25_weight

        merged = [{"product_id": pid, "score": score} for pid, score in combined.items()]
        merged.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Hybrid retrieval produced {len(merged)} merged candidates")
        return merged


# Reranker with metadata and LLM reranking
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

    def _fetch_metadata(self, product_id: str) -> Dict[str, Any]:
        logger.debug(f"Fetching metadata for product_id {product_id}")
        query = text(
            """
            SELECT product_id, product_name, category, price
            FROM chunks
            WHERE product_id = :pid
            LIMIT 1
            """
        )
        try:
            with self.engine.begin() as conn:
                row = conn.execute(query, {"pid": product_id}).fetchone()
                return dict(row._mapping) if row else {}
        except Exception as e:
            logger.error(f"Metadata fetch failed for product_id {product_id}: {e}")
            return {}

    def _apply_metadata(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Applying metadata to {len(candidates)} candidates")
        enriched: List[Dict[str, Any]] = []
        for c in candidates:
            meta = self._fetch_metadata(c["product_id"])
            if not meta:
                logger.warning("No metadata found for product_id '%s'", c["product_id"])
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
                    f"{i}. id={item['product_id']}, "
                    f"name={item['product_name']}, "
                    f"category={item['category']}, "
                    f"price={item['price']}"
                )
            prompt_lines.append(
                "Return a comma-separated list of product_ids in the new order."
            )
            prompt = "\n".join(prompt_lines)

            text_response = self.llm_client.generate(prompt, max_tokens=128).strip()
            product_ids = [p.strip() for p in text_response.split(",") if p.strip()]

            order = {pid: idx for idx, pid in enumerate(product_ids)}
            items.sort(key=lambda x: order.get(x["product_id"], len(items)))
            return items

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}.")
            return items

    def run(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Running RerankerAgent for query {query} with {len(candidates)} candidates.")

        enriched = self._apply_metadata(candidates)
        enriched.sort(key=lambda x: x["score"], reverse=True)
        top = enriched[: self.final_top_k]
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
    """Normalizes the query and applies simple synonym and category hints."""

    procedural_memory: Dict[str, Any]
    embedding_model: SentenceTransformer

    def __post_init__(self):
        self.synonyms = self.procedural_memory.get("synonyms", {})
        self.category_keywords = self.procedural_memory.get("category_keywords", [])
        logger.info(f"QueryUnderstandingAgent initialized with {len(self.synonyms)} synonyms and {len(self.category_keywords)} category keywords")

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

    def run(self, raw_query: str) -> Dict[str, Any]:
        logger.info(f"Running QueryUnderstandingAgent on query: {raw_query}")
        cleaned = self._normalize(raw_query)
        cleaned = self._apply_synonyms(cleaned)
        category = self._infer_category(cleaned)

        return {
            "raw_query": raw_query,
            "clean_query": cleaned,
            "category": category,
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

    def run(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Running BM25 retrieval for query: {query}", )
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
            try:
                row = self.mapping.iloc[idx]
                out.append({"product_id": row["product_id"], "score": float(scores[idx])})
            except Exception as e:
                logger.error(f"Failed to map BM25 index {idx} to product: {e}")
        return out


# Hybrid retrieval (FAISS + BM25)
@dataclass
class HybridRetrievalAgent:
    """Combines FAISS semantic search with BM25 lexical search."""

    model_name: str
    device: str
    faiss_index: faiss.Index
    mapping: pd.DataFrame
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

    def _semantic_search(self, query: str) -> List[Dict[str, Any]]:
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
            if idx < 0:
                continue
            try:
                row = self.mapping.iloc[idx]
                out.append({"product_id": row["product_id"], "score": float(score)})
            except Exception as e:
                logger.error(f"Failed to map FAISS index {idx} to product: {e}")
        logger.info(f"FAISS returned {len(out)} candidates")
        return out

    def run(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Running HybridRetrievalAgent for query: {query}")

        sem_results = self._semantic_search(query)
        bm25_results = self.bm25_agent.run(query)

        combined: Dict[str, float] = {}

        for r in sem_results:
            combined.setdefault(r["product_id"], 0.0)
            combined[r["product_id"]] += r["score"] * self.semantic_weight

        for r in bm25_results:
            combined.setdefault(r["product_id"], 0.0)
            combined[r["product_id"]] += r["score"] * self.bm25_weight

        merged = [{"product_id": pid, "score": score} for pid, score in combined.items()]
        merged.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Hybrid retrieval produced {len(merged)} merged candidates")
        return merged


# Reranker with metadata and LLM reranking
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

    def _fetch_metadata(self, product_id: str) -> Dict[str, Any]:
        logger.debug(f"Fetching metadata for product_id {product_id}")
        query = text(
            """
            SELECT product_id, product_name, category, price
            FROM chunks
            WHERE product_id = :pid
            LIMIT 1
            """
        )
        try:
            with self.engine.begin() as conn:
                row = conn.execute(query, {"pid": product_id}).fetchone()
                return dict(row._mapping) if row else {}
        except Exception as e:
            logger.error(f"Metadata fetch failed for product_id {product_id}: {e}")
            return {}

    def _apply_metadata(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Applying metadata to {len(candidates)} candidates")
        enriched: List[Dict[str, Any]] = []
        for c in candidates:
            meta = self._fetch_metadata(c["product_id"])
            if not meta:
                logger.warning("No metadata found for product_id '%s'", c["product_id"])
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
                    f"{i}. id={item['product_id']}, "
                    f"name={item['product_name']}, "
                    f"category={item['category']}, "
                    f"price={item['price']}"
                )
            prompt_lines.append(
                "Return a comma-separated list of product_ids in the new order."
            )
            prompt = "\n".join(prompt_lines)

            text_response = self.llm_client.generate(prompt, max_tokens=128).strip()
            product_ids = [p.strip() for p in text_response.split(",") if p.strip()]

            order = {pid: idx for idx, pid in enumerate(product_ids)}
            items.sort(key=lambda x: order.get(x["product_id"], len(items)))
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

        if self.llm_client is not None and top:
            try:
                top = self._llm_rerank(query, top)
            except Exception as e:
                logger.error(f"LLM rerank crashed: {e}. Falling back to score ranking.")
                top.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"RerankerAgent returning {len(top)} final results.")
        return top


        if self.llm_client is not None and top:
            try:
                top = self._llm_rerank(query, top)
            except Exception as e:
                logger.error(f"LLM rerank crashed: {e}. Falling back to score ranking.")
                top.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"RerankerAgent returning {len(top)} final results.")
        return top

            q_emb = self._model.encode(query, convert_to_numpy=True).astype("float32")
            q_emb = q_emb.reshape(1, -1)
            scores, idxs = self.faiss_index.search(q_emb, self.top_k)
        except Exception as e:
            logger.error(f"FAISS search failed for query {query}: {e}")
            return []

        out: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            try:
                row = self.mapping.iloc[idx]
                out.append({"product_id": row["product_id"], "score": float(score)})
            except Exception as e:
                logger.error(f"Failed to map FAISS index {idx} to product: {e}")
        logger.info(f"FAISS returned {len(out)} candidates")
        return out

    def run(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Running HybridRetrievalAgent for query: {query}")

        sem_results = self._semantic_search(query)
        bm25_results = self.bm25_agent.run(query)

        combined: Dict[str, float] = {}

        for r in sem_results:
            combined.setdefault(r["product_id"], 0.0)
            combined[r["product_id"]] += r["score"] * self.semantic_weight

        for r in bm25_results:
            combined.setdefault(r["product_id"], 0.0)
            combined[r["product_id"]] += r["score"] * self.bm25_weight

        merged = [{"product_id": pid, "score": score} for pid, score in combined.items()]
        merged.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Hybrid retrieval produced {len(merged)} merged candidates")
        return merged


# Reranker with metadata and LLM reranking
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

    def _fetch_metadata(self, product_id: str) -> Dict[str, Any]:
        logger.debug(f"Fetching metadata for product_id {product_id}")
        query = text(
            """
            SELECT product_id, product_name, category, price
            FROM chunks
            WHERE product_id = :pid
            LIMIT 1
            """
        )
        try:
            with self.engine.begin() as conn:
                row = conn.execute(query, {"pid": product_id}).fetchone()
                return dict(row._mapping) if row else {}
        except Exception as e:
            logger.error(f"Metadata fetch failed for product_id {product_id}: {e}")
            return {}

    def _apply_metadata(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Applying metadata to {len(candidates)} candidates")
        enriched: List[Dict[str, Any]] = []
        for c in candidates:
            meta = self._fetch_metadata(c["product_id"])
            if not meta:
                logger.warning("No metadata found for product_id '%s'", c["product_id"])
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
                    f"{i}. id={item['product_id']}, "
                    f"name={item['product_name']}, "
                    f"category={item['category']}, "
                    f"price={item['price']}"
                )
            prompt_lines.append(
                "Return a comma-separated list of product_ids in the new order."
            )
            prompt = "\n".join(prompt_lines)

            text_response = self.llm_client.generate(prompt, max_tokens=128).strip()
            product_ids = [p.strip() for p in text_response.split(",") if p.strip()]

            order = {pid: idx for idx, pid in enumerate(product_ids)}
            items.sort(key=lambda x: order.get(x["product_id"], len(items)))
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
