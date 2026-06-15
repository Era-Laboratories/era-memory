"""Embedder conformance — every adapter must pass this unchanged."""

from __future__ import annotations

import math

import pytest

from era_memory.core.logic.dedup import cosine_similarity


async def test_dimensions_and_shape(embedder):
    out = await embedder.embed(["hello world", "another text"])
    assert len(out) == 2
    assert all(len(v) == embedder.dimensions for v in out)


async def test_deterministic(embedder):
    a = (await embedder.embed(["repeatable input"]))[0]
    b = (await embedder.embed(["repeatable input"]))[0]
    assert a == b


async def test_l2_normalized(embedder):
    v = (await embedder.embed(["some content here"]))[0]
    norm = math.sqrt(sum(x * x for x in v))
    assert norm == pytest.approx(1.0, abs=1e-6)


async def test_semantic_similarity_signal(embedder):
    coffee1 = (await embedder.embed(["dark roast coffee espresso"]))[0]
    coffee2 = (await embedder.embed(["espresso coffee beans"]))[0]
    unrelated = (await embedder.embed(["quantum physics lecture notes"]))[0]
    assert cosine_similarity(coffee1, coffee2) > cosine_similarity(coffee1, unrelated)


async def test_model_id_present(embedder):
    assert isinstance(embedder.model_id, str) and embedder.model_id
