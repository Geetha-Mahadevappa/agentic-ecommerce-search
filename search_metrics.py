#!/usr/bin/env python3
"""
Evaluation module for the e-commerce search system.

Measures, over a sample of real purchase-log queries:
    - P99 latency
    - RAGAS faithfulness: is the surfaced answer grounded in the retrieved product text?
    - RAGAS retrieval quality (LLMContextPrecisionWithoutReference): are the retrieved
      contexts relevant to the query?
    - RAGAS answer relevance (SemanticSimilarity, response vs. query): does the surfaced
      answer stay on-topic for the query? Substituted for RAGAS's own ResponseRelevancy,
      which reliably crashes CUDA here -- its multi-completion "strictness" sampling is
      incompatible with greedy decoding on a small local completion-only Qwen2-1.5B pipeline.

Writes results/metrics.json.
"""

import json
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from transformers import pipeline as hf_pipeline

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, SemanticSimilarity
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFacePipeline

from search_orchestration import build_search_pipeline, run_search

# Retrieved contexts and query sample are capped to keep the LLM-judged metrics
# (one generation call per context, per query) tractable on a single local GPU.
MAX_CONTEXTS_PER_SAMPLE = 3
DEFAULT_SAMPLE_SIZE = 10


class SharedSentenceTransformerEmbeddings(BaseRagasEmbeddings):
    """Wraps the pipeline's already-loaded SentenceTransformer for RAGAS, avoiding a second GPU-resident copy."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.run_config = RunConfig()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


def build_ragas_llm(llm_client):
    """Wrap the pipeline's already-loaded Qwen model/tokenizer for RAGAS, avoiding a second GPU-resident copy."""
    llm_client.tokenizer.padding_side = "left"  # required for correct batched generation with this model
    pipe = hf_pipeline(
        "text-generation",
        model=llm_client.model,
        tokenizer=llm_client.tokenizer,
        device=llm_client.model.device,
        max_new_tokens=128,
        do_sample=False,
        batch_size=1,
        pad_token_id=llm_client.tokenizer.eos_token_id,
        return_full_text=False,
    )
    return LangchainLLMWrapper(HuggingFacePipeline(pipeline=pipe))


def build_eval_queries(purchase_df: pd.DataFrame, sample_size: int) -> List[str]:
    """Build a mix of baseline and price-intent queries from real purchase logs (80/20 split)."""
    if len(purchase_df) > sample_size:
        purchase_df = purchase_df.sample(n=sample_size, random_state=42)

    low_price = ["cheap", "budget", "affordable", "low price"]
    mid_price = ["mid range", "mid priced", "value for money"]
    high_price = ["premium", "luxury", "high-end", "expensive", "top tier"]
    numeric_triggers = ["under", "below", "less than"]
    numeric_values = ["50", "100", "150", "200", "300", "500"]

    queries = []
    for _, row in purchase_df.iterrows():
        product_name = str(row["ProductName"])
        category = str(row["ProductCategory"])
        country = str(row["Country"])

        if np.random.rand() < 0.2:
            intent_type = np.random.choice(["low", "mid", "high", "numeric"])
            if intent_type == "low":
                qualifier = np.random.choice(low_price)
            elif intent_type == "mid":
                qualifier = np.random.choice(mid_price)
            elif intent_type == "high":
                qualifier = np.random.choice(high_price)
            else:
                trig = np.random.choice(numeric_triggers)
                val = np.random.choice(numeric_values)
                qualifier = f"{trig} {val}"
            queries.append(f"{qualifier} {product_name} {category} in {country}")
        else:
            queries.append(f"{product_name} {category} in {country}")

    return queries


def build_ragas_samples(pipeline, queries: List[str]):
    """Run each query through the search pipeline, measuring latency and building RAGAS samples."""
    metadata = pipeline.retrieval_agent.metadata
    latencies = []
    samples = []

    for query in queries:
        start = time.perf_counter()
        results = run_search(pipeline, query)["results"]
        latencies.append((time.perf_counter() - start) * 1000)

        if not results:
            continue

        contexts = [
            str(metadata.loc[r["variant_id"], "text"])
            for r in results[:MAX_CONTEXTS_PER_SAMPLE]
            if r["variant_id"] in metadata.index
        ]
        if not contexts:
            continue

        top = results[0]
        response = (
            f"{top.get('product_name')} - {top.get('category')} product, "
            f"priced at ${top.get('price')} ({top.get('price_level')} tier)."
        )

        samples.append(
            SingleTurnSample(
                user_input=query,
                response=response,
                retrieved_contexts=contexts,
                reference=query,
            )
        )

    return samples, latencies


if __name__ == "__main__":
    print("Building search pipeline...")
    pipeline = build_search_pipeline("configs/config_agents.yaml")

    purchase_df = pd.read_csv("data/raw/customer_purchase_data.csv")
    queries = build_eval_queries(purchase_df, sample_size=DEFAULT_SAMPLE_SIZE)

    print(f"Running {len(queries)} queries through the search pipeline...")
    samples, latencies = build_ragas_samples(pipeline, queries)

    print(f"Scoring {len(samples)} samples with RAGAS...")
    ragas_llm = build_ragas_llm(pipeline.reranker.llm_client)
    ragas_embeddings = SharedSentenceTransformerEmbeddings(pipeline.query_agent.embedding_model)

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), LLMContextPrecisionWithoutReference(), SemanticSimilarity()],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        # This model/pipeline is not safe under concurrent GPU calls -- RAGAS's default
        # concurrency corrupts the CUDA context (device-side asserts on later jobs).
        run_config=RunConfig(max_workers=1),
    )

    summary: Dict[str, float] = {
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        **{k: float(v) for k, v in result._repr_dict.items()},
    }

    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation complete. Summary saved to results/metrics.json")
    print(json.dumps(summary, indent=2))
