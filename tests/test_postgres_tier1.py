"""
Tier-1 Postgres+pgvector specifics: single-DB dual-write collapse, atomicity, recall.
Gated on MEMORY_TEST_PG_DSN (set when a pgvector container is available).
"""

from __future__ import annotations

import os

import pytest

PG_DSN = os.environ.get("MEMORY_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not PG_DSN, reason="no MEMORY_TEST_PG_DSN")

pytest.importorskip("asyncpg")

from era_memory.adapters.postgres import PgVectorStore  # noqa: E402
from era_memory.models import MemoryRecord, SearchRequest  # noqa: E402
from era_memory.wiring import build_memory_async  # noqa: E402


def test_vector_store_is_co_transactional():
    assert PgVectorStore.co_transactional is True


async def test_store_persists_and_searches(clock):
    mem = await build_memory_async(tier=1, dsn=PG_DSN, reset=True, clock=clock)
    try:
        await mem.store(MemoryRecord(user_id="u1", content="dark roast coffee beans"))
        await mem.store(MemoryRecord(user_id="u1", content="quarterly revenue report"))
        res = await mem.search(SearchRequest(user_id="u1", query="coffee"))
        assert res.results and "coffee" in res.results[0].content
    finally:
        await mem._pg_backend.close()  # type: ignore[attr-defined]


async def test_collapse_is_atomic_on_vector_failure(clock):
    mem = await build_memory_async(tier=1, dsn=PG_DSN, reset=True, clock=clock)
    try:
        async def boom(*a, **k):
            raise RuntimeError("vector write failed inside uow")

        mem.vector_store._upsert = boom  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError):
            await mem.store(MemoryRecord(user_id="u1", content="should roll back entirely"))

        pool = mem._pg_backend.pool  # type: ignore[attr-defined]
        async with pool.acquire() as c:
            assert await c.fetchval("SELECT count(*) FROM memories") == 0
            assert await c.fetchval("SELECT count(*) FROM memory_vectors") == 0
    finally:
        await mem._pg_backend.close()  # type: ignore[attr-defined]


async def test_user_isolation(clock):
    mem = await build_memory_async(tier=1, dsn=PG_DSN, reset=True, clock=clock)
    try:
        await mem.store(MemoryRecord(user_id="alice", content="alice likes espresso"))
        await mem.store(MemoryRecord(user_id="bob", content="bob likes espresso"))
        res = await mem.search(SearchRequest(user_id="alice", query="espresso"))
        assert res.results and all(r.content.startswith("alice") for r in res.results)
    finally:
        await mem._pg_backend.close()  # type: ignore[attr-defined]
