"""
Reciprocal Rank Fusion + recency decay + final scoring.

Pure functions, no I/O — lifted from era-core era-memory-service/src/services/search.py.
Constants (k=60, semantic 0.6 / lexical 0.4, 30-day half-life) live in Settings so every
tier ranks identically.
"""

from __future__ import annotations

import math


def rrf_fuse(
    semantic: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    *,
    k: int = 60,
    semantic_weight: float = 0.6,
    lexical_weight: float = 0.4,
) -> dict[str, float]:
    """
    Fuse two ranked id lists into ``{id: rrf_score}``.

    Each list is assumed pre-sorted best-first. RRF uses *rank*, not the raw score, so it
    is robust to incomparable score scales (cosine vs BM25) — the reason a weaker lexical
    backend (ts_rank vs true BM25) still fuses cleanly.
    """
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(semantic, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + semantic_weight / (k + rank)
    for rank, (doc_id, _) in enumerate(lexical, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + lexical_weight / (k + rank)
    return scores


def recency_decay(age_days: float, *, half_life_days: float = 30.0) -> float:
    """Exponential decay in [0, 1]: 1.0 at age 0, ~0.5 at one half-life."""
    if half_life_days <= 0:
        return 1.0
    age = max(0.0, age_days)
    return math.exp(-0.693 * age / half_life_days)


def final_score(
    base_rrf: float,
    importance: float,
    recency: float,
    *,
    recency_weight: float = 0.3,
) -> float:
    """``base_rrf * importance * (1 - w + w*recency)`` — era-core's combiner."""
    recency_factor = (1.0 - recency_weight) + recency_weight * recency
    return base_rrf * max(importance, 1e-6) * recency_factor
