"""
Matryoshka truncate + L2-normalize. Pure Python.

Lifted from era-core era-memory-service/src/services/embedding.py: slice an embedding to a
target dimension (valid only *within* a Matryoshka model family) and L2-normalize so cosine
== dot product downstream.
"""

from __future__ import annotations

import math


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def truncate_and_normalize(vec: list[float], target_dim: int) -> list[float]:
    if target_dim <= 0 or target_dim >= len(vec):
        return l2_normalize(vec)
    return l2_normalize(vec[:target_dim])
