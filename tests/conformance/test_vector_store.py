"""VectorStore conformance — every adapter must pass this unchanged (in-memory + sqlite)."""

from __future__ import annotations

import pytest

from era_memory.errors import VectorStoreWriteError
from era_memory.models import MemoryType, SearchFilters, VectorRecord


def _emb(vals, dim):
    """Pad a short logical vector to the fixture dimension (empty stays empty)."""
    if not vals:
        return []
    return list(vals) + [0.0] * (dim - len(vals))


def _vr(id, user, vals, dim, **kw):
    return VectorRecord(id=id, user_id=user, embedding=_emb(vals, dim), **kw)


async def test_insert_and_cosine_ordering(vector_store, vec_dim):
    await vector_store.insert(
        [
            _vr("near", "u1", [1.0, 0.0, 0.0], vec_dim),
            _vr("mid", "u1", [0.7, 0.7, 0.0], vec_dim),
            _vr("far", "u1", [0.0, 0.0, 1.0], vec_dim),
        ]
    )
    hits = await vector_store.search("u1", _emb([1.0, 0.0, 0.0], vec_dim), 3, SearchFilters())
    assert [i for i, _ in hits] == ["near", "mid", "far"]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)


async def test_user_scoping(vector_store, vec_dim):
    await vector_store.insert([_vr("a", "alice", [1.0, 0.0], vec_dim)])
    assert await vector_store.search("bob", _emb([1.0, 0.0], vec_dim), 5, SearchFilters()) == []


async def test_filter_by_memory_type(vector_store, vec_dim):
    await vector_store.insert(
        [
            _vr("e", "u1", [1.0, 0.0], vec_dim, memory_type=MemoryType.EPISODE),
            _vr("s", "u1", [1.0, 0.0], vec_dim, memory_type=MemoryType.SEMANTIC),
        ]
    )
    hits = await vector_store.search(
        "u1", _emb([1.0, 0.0], vec_dim), 5, SearchFilters(memory_type=MemoryType.SEMANTIC)
    )
    assert [i for i, _ in hits] == ["s"]


async def test_delete(vector_store, vec_dim):
    await vector_store.insert([_vr("a", "u1", [1.0, 0.0], vec_dim)])
    assert await vector_store.delete(["a"]) == 1
    assert await vector_store.search("u1", _emb([1.0, 0.0], vec_dim), 5, SearchFilters()) == []


async def test_empty_embedding_raises(vector_store, vec_dim):
    with pytest.raises(VectorStoreWriteError):
        await vector_store.insert([_vr("a", "u1", [], vec_dim)])


async def test_is_connected(vector_store):
    assert vector_store.is_connected() is True
