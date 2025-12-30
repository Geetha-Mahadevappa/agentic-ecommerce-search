#!/usr/bin/env python3
"""
Memory manager for the search pipeline.

This module organizes information into four memory layers:
1. Procedural memory (YAML): fixed system knowledge
   - category keywords, domain rules

2. Long-term memory (JSON): persistent user preferences
   - preferred categories, typical price range

3. Short-term memory history: session-level context
   - recent queries, result counts

4. Behavioral/Activity log (JSON): recent user activity for personalization
   - timestamped queries, clicked product IDs

All disk writes use file locks and atomic temp writes to avoid corruption.
"""

import re
import json
import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

import yaml
from filelock import FileLock

from logging_config import setup_logging
setup_logging("logs/agent_runtime.log")
logger = logging.getLogger(__name__)


@dataclass
class MemoryAgent:
    memory_dir: Path
    user_prefs_path: Path
    procedural_memory_path: Path
    activity_log_path: Path
    short_term_limit: int = 5

    short_term_history: List[Dict[str, Any]] = field(default_factory=list)
    procedural_memory: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize memory directories and load or create memory files."""
        logger.info(f"Initializing MemoryAgent at {self.memory_dir}")
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Ensure procedural memory exists
        if not self.procedural_memory_path.exists():
            raise FileNotFoundError(f"Missing procedural memory file: {self.procedural_memory_path}. "
                "Please create it before running the system.")

        # Ensure user preferences exist
        if not self.user_prefs_path.exists():
            logger.info(f"Creating user preferences file: {self.user_prefs_path}")
            self._safe_write_json(self.user_prefs_path, {
                "preferred_categories": [],
                "price_sensitivity": None
            })

        # Ensure activity log exists
        if not self.activity_log_path.exists():
            logger.info(f"Creating activity log file: {self.activity_log_path}")
            self._safe_write_json(self.activity_log_path, [])

        # Load procedural memory
        self.procedural_memory = self._safe_read_yaml(
            self.procedural_memory_path,
            default={"synonyms": {}, "category_keywords": []}
        )
        logger.info(
            "Loaded procedural memory with %d category keywords",
            len(self.procedural_memory.get("category_keywords", []))
        )

    # Safe file helpers
    def _safe_read_json(self, path: Path, default):
        lock = FileLock(str(path) + ".lock")
        with lock:
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read JSON file {path}: {e}. Rewriting default.")
                self._safe_write_json(path, default)
                return default

    def _safe_write_json(self, path: Path, data):
        lock = FileLock(str(path) + ".lock")
        with lock:
            tmp = Path(str(path) + ".tmp")
            try:
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                tmp.replace(path)
            except Exception as e:
                logger.error(f"Failed to write JSON file {path}: {e}")

    def _safe_read_yaml(self, path: Path, default):
        lock = FileLock(str(path) + ".lock")
        with lock:
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                    return data if data is not None else default
            except Exception as e:
                logger.error(f"Failed to read YAML file {path}: {e}. Rewriting default.")
                self._safe_write_yaml(path, default)
                return default

    def _safe_write_yaml(self, path: Path, data):
        lock = FileLock(str(path) + ".lock")
        with lock:
            tmp = Path(str(path) + ".tmp")
            try:
                with open(tmp, "w") as f:
                    yaml.safe_dump(data, f)
                    f.flush()
                    os.fsync(f.fileno())
                tmp.replace(path)
            except Exception as e:
                logger.error(f"Failed to write YAML file {path}: {e}")

    def _infer_price_sensitivity(self, query: str, product_type: str | None = None) -> str | None:
        """Infer the user's price sensitivity using price bands from procedural memory."""
        q = query.lower()

        # Load price bands from procedural memory
        price_bands = self.procedural_memory.get("price_bands", {})

        # Determine which category's price bands to use
        category_key = None
        if product_type:
            pt = product_type.lower()
            for cat in price_bands.keys():
                if cat in pt:
                    category_key = cat
                    break

        if not category_key and price_bands:
            category_key = next(iter(price_bands.keys()))

        # Extract numeric price intent from query
        import re
        match = re.search(r"(under|below|less than)\s+(\d+)", q)
        if match and category_key:
            amount = int(match.group(2))
            bands = price_bands.get(category_key, {})

            for band_name, (low, high) in bands.items():
                if low <= amount <= high:
                    return band_name

        # Keyword-based inference
        detect_terms = self.procedural_memory.get("query_rules", {}).get("detect_price_intent", [])

        low_terms = ["cheap", "budget", "affordable", "low cost"]
        mid_terms = ["mid range", "mid-range", "value for money", "reasonable"]
        high_terms = ["premium", "high end", "high-end", "expensive", "flagship"]

        if any(term in q for term in low_terms + detect_terms):
            return "low"
        if any(term in q for term in mid_terms):
            return "medium"
        if any(term in q for term in high_terms):
            return "high"
        return None

    # Procedural memory
    def get_category_keywords(self) -> List[str]:
        return self.procedural_memory.get("category_keywords", [])


    # User preferences
    def _read_user_prefs(self):
        return self._safe_read_json(
            self.user_prefs_path,
            default={"preferred_categories": [], "price_sensitivity": None}
        )

    def _write_user_prefs(self, data):
        self._safe_write_json(self.user_prefs_path, data)

    def update_preferences_from_query(self, query: str, product_type: str | None = None):
        """Update user preferences based on category keywords found in the query."""
        logger.info(f"Updating user preferences from query: {query}")

        prefs = self._read_user_prefs()
        categories = prefs.get("preferred_categories", [])
        q = query.lower()

        if product_type:
            pt = product_type.lower().strip()
            if pt and pt not in categories:
                logger.info(f"Adding preferred category from product_type: {pt}")
                categories.append(pt)

        for kw in self.get_category_keywords():
            if kw in q and kw not in categories:
                logger.info(f"Adding preferred category: {kw}")
                categories.append(kw)

        prefs["preferred_categories"] = categories
        price_pref = self._infer_price_sensitivity(query)
        if price_pref:
            prefs["price_sensitivity"] = price_pref
        self._write_user_prefs(prefs)

    def get_preferences(self):
        return self._read_user_prefs()


    # Short-term memory
    def add_short_term(self, query: str, results: List[Any]):
        """Add a short-term memory entry for the current query."""
        logger.info(f"Adding short-term memory entry: '{query}' ({len(results)} results)")

        entry = {"query": query, "num_results": len(results)}
        self.short_term_history.append(entry)

        if len(self.short_term_history) > self.short_term_limit:
            removed = self.short_term_history.pop(0)
            logger.info(f"Short-term memory limit reached. Removing oldest entry: {removed}")

    def get_short_term(self):
        return self.short_term_history


    # Activity log
    def _read_activity_log(self):
        return self._safe_read_json(self.activity_log_path, default=[])

    def _write_activity_log(self, data):
        self._safe_write_json(self.activity_log_path, data)

    def log_activity(self, query: str, result_ids: List[Any]):
        """Append a timestamped activity entry for personalization."""
        logger.info(f"Logging activity: query='{query}', {len(result_ids)} result IDs")

        log = self._read_activity_log()
        log.append({
            "query": query,
            "result_ids": result_ids,
            "timestamp": time.time()
        })
        self._write_activity_log(log)

    def get_recent_activity(self, days: int = 7):
        cutoff = time.time() - days * 86400
        log = self._read_activity_log()
        return [x for x in log if x.get("timestamp", 0) >= cutoff]
