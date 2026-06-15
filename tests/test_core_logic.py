from __future__ import annotations

import math

import pytest

from era_memory.core.logic.dedup import cosine_similarity, is_duplicate
from era_memory.core.logic.entropy import entropy_score
from era_memory.core.logic.matryoshka import l2_normalize, truncate_and_normalize
from era_memory.core.logic.rrf import final_score, recency_decay, rrf_fuse


def test_rrf_fuse_weights_and_ranks():
    semantic = [("a", 0.9), ("b", 0.8)]
    lexical = [("b", 5.0), ("c", 1.0)]
    scores = rrf_fuse(semantic, lexical, k=60, semantic_weight=0.6, lexical_weight=0.4)
    # a: only semantic rank1; b: semantic rank2 + lexical rank1; c: lexical rank2
    assert scores["a"] == pytest.approx(0.6 / 61)
    assert scores["b"] == pytest.approx(0.6 / 62 + 0.4 / 61)
    assert scores["c"] == pytest.approx(0.4 / 62)
    # b appears in both lists -> should outrank a and c
    assert scores["b"] > scores["a"] > scores["c"]


def test_recency_decay_half_life():
    assert recency_decay(0, half_life_days=30) == pytest.approx(1.0)
    assert recency_decay(30, half_life_days=30) == pytest.approx(0.5, abs=1e-3)
    assert recency_decay(60, half_life_days=30) == pytest.approx(0.25, abs=1e-3)


def test_final_score_combiner():
    # recency_factor = (1-0.3) + 0.3*recency
    assert final_score(0.1, 1.0, 1.0, recency_weight=0.3) == pytest.approx(0.1 * 1.0)
    assert final_score(0.1, 1.0, 0.0, recency_weight=0.3) == pytest.approx(0.1 * 0.7)
    # importance scales linearly
    assert final_score(0.1, 0.5, 1.0, recency_weight=0.3) == pytest.approx(0.05)


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], []) == 0.0


def test_is_duplicate_threshold():
    assert is_duplicate([1.0, 0.0], [[0.99, 0.01]], threshold=0.85)
    assert not is_duplicate([1.0, 0.0], [[0.0, 1.0]], threshold=0.85)


def test_l2_normalize_and_truncate():
    v = l2_normalize([3.0, 4.0])
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0)
    t = truncate_and_normalize([3.0, 4.0, 100.0], 2)
    assert len(t) == 2
    assert math.isclose(math.sqrt(sum(x * x for x in t)), 1.0)


def test_entropy_score_bounds():
    assert entropy_score("") == 0.0
    assert entropy_score("hello") == 0.0  # single token
    assert entropy_score("aaa aaa aaa") == pytest.approx(0.0)  # zero vocab entropy
    assert entropy_score("the quick brown fox jumps") > 0.5
