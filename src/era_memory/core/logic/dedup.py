"""
Cosine similarity + dedup decision. Pure Python (no numpy in core).

Mirrors era-core era-memory-encoder/src/services/encoder.py: cosine over L2 norms,
duplicate when similarity >= threshold (default 0.85).
"""

from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def is_duplicate(
    candidate: list[float],
    existing: list[list[float]],
    *,
    threshold: float = 0.85,
) -> bool:
    """True if ``candidate`` is within ``threshold`` cosine of any existing embedding."""
    return any(cosine_similarity(candidate, e) >= threshold for e in existing)


def most_similar(
    candidate: list[float],
    existing: list[tuple[str, list[float]]],
) -> tuple[str | None, float]:
    """Return ``(id, similarity)`` of the closest existing embedding (or ``(None, 0.0)``)."""
    best_id: str | None = None
    best = 0.0
    for doc_id, emb in existing:
        sim = cosine_similarity(candidate, emb)
        if sim > best:
            best, best_id = sim, doc_id
    return best_id, best
