#!/usr/bin/env python3
"""
Evaluation module for the e-commerce search system.

This script measures:
    - P99 latency
    - nDCG@10
    - Zero-result rate
    - Conversion per search

It evaluates a sampled subset of purchase logs and writes:
    - eval_queries.txt
    - success_cases.txt
    - failure_cases.txt
    - metrics.json
"""

import json
import os
import time
from typing import List, Dict

import numpy as np
import pandas as pd

from search_orchestration import build_search_pipeline, run_search


class SearchEvaluator:
    """
    Runs evaluation over a set of labeled queries and computes core metrics.
    """

    def __init__(self, orchestrator, eval_data: List[Dict], variant_to_product: Dict[str, str]):
        self.orchestrator = orchestrator
        self.eval_data = eval_data
        self.variant_map = variant_to_product

        os.makedirs("results", exist_ok=True)
        self.success_log = open("results/success_cases.txt", "w", encoding="utf-8")
        self.failure_log = open("results/failure_cases.txt", "w", encoding="utf-8")
        self.query_log = open("results/eval_queries.txt", "w", encoding="utf-8")

    def ndcg_at_10(self, results: List[str], relevant: List[str]) -> float:
        """Compute nDCG@10 using binary relevance."""
        k = 10
        dcg = 0.0

        for i, pid in enumerate(results[:k]):
            if pid in relevant:
                dcg += 1.0 / np.log2(i + 2)

        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0.0

    def conversion_score(self, results: List[str], purchased: str) -> float:
        """
        Rank-aware conversion metric:
            - 1.0 for rank 1
            - 0.7 for rank 2–3
            - 0.4 for rank 4–10
            - 0.0 otherwise
        """
        if purchased not in results:
            return 0.0

        rank = results.index(purchased)

        if rank == 0:
            return 1.0
        if rank <= 2:
            return 0.7
        if rank <= 9:
            return 0.4
        return 0.0

    def log_case(self, query: str, purchased: str, found: List[str], ndcg: float, conv: float) -> None:
        """Write success or failure cases to text files."""
        entry = (
            f"QUERY: {query}\n"
            f"Purchased ProductID: {purchased}\n"
            f"Returned ProductIDs: {found}\n"
            f"nDCG@10: {ndcg:.4f}\n"
            f"Conversion Score: {conv:.2f}\n"
            f"{'-' * 60}\n"
        )

        if conv > 0:
            self.success_log.write(entry)
        else:
            self.failure_log.write(entry)

    def run(self) -> Dict[str, float]:
        """Run evaluation and write summary metrics."""
        latencies = []
        ndcg_scores = []
        conversion_scores = []
        zero_results = 0

        for item in self.eval_data:
            query = item["query"]
            relevant = item["relevant"]
            purchased = item["purchased"]

            self.query_log.write(query + "\n")

            # Use time.perf_counter() for accurate p99 measurement.
            start = time.perf_counter()
            raw_results = self.orchestrator(query)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            found_pids = [
                self.variant_map.get(str(r.get("variant_id")))
                for r in raw_results
            ]
            found_pids = [p for p in found_pids if p is not None]

            if not found_pids:
                zero_results += 1

            ndcg = self.ndcg_at_10(found_pids, relevant)
            conv = self.conversion_score(found_pids, purchased)

            ndcg_scores.append(ndcg)
            conversion_scores.append(conv)

            self.log_case(query, purchased, found_pids, ndcg, conv)

        self.success_log.close()
        self.failure_log.close()
        self.query_log.close()

        summary = {
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "ndcg@10": float(np.mean(ndcg_scores)),
            "zero_result_rate": zero_results / len(self.eval_data),
            "conversion_per_search": float(np.mean(conversion_scores)),
        }

        with open("results/metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("Evaluation complete. Summary saved to results/metrics.json")
        return summary


def build_eval_data(purchase_df: pd.DataFrame, review_df: pd.DataFrame, sample_size: int = 150) -> List[Dict]:
    """
    Build evaluation data using a mix of:
        - baseline queries (product name + category + country)
        - price-intent queries (cheap, premium, under 100, etc.)

    80% baseline, 20% price-intent.
    """
    if len(purchase_df) > sample_size:
        purchase_df = purchase_df.sample(n=sample_size, random_state=42)

    eval_data = []

    positive_words = ["great", "amazing", "perfect", "love", "recommend", "satisfied"]
    review_df["is_positive"] = review_df["ReviewText"].str.contains(
        "|".join(positive_words), case=False, na=False
    )

    positive_pids = set(review_df[review_df["is_positive"]]["ProductID"].astype(str))

    # Price intent vocabulary
    low_price = ["cheap", "budget", "affordable", "low price"]
    mid_price = ["mid range", "mid priced", "value for money"]
    high_price = ["premium", "luxury", "high-end", "expensive", "top tier"]
    numeric_triggers = ["under", "below", "less than"]
    numeric_values = ["50", "100", "150", "200", "300", "500"]

    for _, row in purchase_df.iterrows():
        product_name = str(row["ProductName"])
        category = str(row["ProductCategory"])
        country = str(row["Country"])
        purchased_pid = str(row["ProductID"])

        # Decide if this query will include price intent
        use_price_intent = (np.random.rand() < 0.2)  # 20%

        if use_price_intent:
            # Choose a price intent type
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

            # Build price-intent query
            query = f"{qualifier} {product_name} {category} in {country}"

        else:
            # Baseline query
            query = f"{product_name} {category} in {country}"

        # Build relevance set
        category_peers = (
            purchase_df[purchase_df["ProductCategory"] == category]["ProductID"]
            .astype(str)
            .unique()
            .tolist()
        )

        relevant = [purchased_pid] + [p for p in category_peers if p in positive_pids]
        relevant = list(set(relevant))

        eval_data.append(
            {
                "query": query,
                "relevant": relevant,
                "purchased": purchased_pid,
            }
        )
    return eval_data

if __name__ == "__main__":
    print("Building search pipeline...")
    pipeline = build_search_pipeline("configs/config_agents.yaml")

    def orchestrator_fn(query: str):
        output = run_search(pipeline, query)
        return output["results"]

    purchase_df = pd.read_csv("data/raw/customer_purchase_data.csv")
    review_df = pd.read_csv("data/raw/customer_reviews_data.csv")

    metadata_df = pipeline.retrieval_agent.metadata
    variant_to_product = dict(
        zip(
            metadata_df["variant_id"].astype(str),
            metadata_df["product_id"].astype(str),
        )
    )

    eval_data = build_eval_data(purchase_df, review_df, sample_size=150)
    evaluator = SearchEvaluator(orchestrator_fn, eval_data, variant_to_product)
    evaluator.run()
