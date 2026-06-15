"""
Configuration. Plain dataclass read from env (no pydantic-settings in base).

One knob — MEMORY_TIER — selects an adapter set in wiring.py. Per-port overrides are
allowed for advanced/mixed setups. Defaults reproduce era-core's tuning constants so
ranking is identical across tiers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


@dataclass
class Settings:
    tier: int = 0

    # Embedding / dimension lock-in (single-tier-for-life; recorded + guarded).
    embedding_model: str = "in-memory-hash"
    embedding_dimensions: int = 64

    # RRF / ranking — era-core parity constants.
    rrf_k: int = 60
    rrf_semantic_weight: float = 0.6
    rrf_lexical_weight: float = 0.4
    recency_half_life_days: float = 30.0
    recency_weight: float = 0.3
    deep_search_threshold: float = 0.5

    # Dedup / extraction.
    dedup_similarity_threshold: float = 0.85
    entropy_threshold: float = 0.35
    max_memories_per_session: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            tier=_env_int("MEMORY_TIER", 0),
            embedding_model=_env("MEMORY_EMBEDDING_MODEL", "in-memory-hash"),
            embedding_dimensions=_env_int("MEMORY_EMBEDDING_DIMENSIONS", 64),
            rrf_k=_env_int("MEMORY_RRF_K", 60),
            rrf_semantic_weight=_env_float("MEMORY_RRF_SEMANTIC_WEIGHT", 0.6),
            rrf_lexical_weight=_env_float("MEMORY_RRF_LEXICAL_WEIGHT", 0.4),
            recency_half_life_days=_env_float("MEMORY_RECENCY_HALF_LIFE_DAYS", 30.0),
            recency_weight=_env_float("MEMORY_RECENCY_WEIGHT", 0.3),
            deep_search_threshold=_env_float("MEMORY_DEEP_SEARCH_THRESHOLD", 0.5),
            dedup_similarity_threshold=_env_float("MEMORY_DEDUP_THRESHOLD", 0.85),
            entropy_threshold=_env_float("MEMORY_ENTROPY_THRESHOLD", 0.35),
            max_memories_per_session=_env_int("MEMORY_MAX_PER_SESSION", 8),
        )
