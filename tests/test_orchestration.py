from __future__ import annotations

import pytest

from era_memory.adapters.memory import InMemoryRecordStore, InMemoryVectorStore
from era_memory.core.orchestration import (
    DETAIL_MEMORY,
    dual_write,
    dual_write_batch,
    soft_delete,
)
from era_memory.errors import DualWriteVectorError, VectorStoreWriteError
from era_memory.models import MemoryRecord


class FailingVectorStore(InMemoryVectorStore):
    async def insert(self, records):
        raise VectorStoreWriteError("simulated milvus down")


class CountingVectorStore(InMemoryVectorStore):
    def __init__(self):
        super().__init__()
        self.insert_calls = 0

    async def insert(self, records):
        self.insert_calls += 1
        return await super().insert(records)


_DEFAULT_EMB = [1.0, 0.0]


def _rec(content="hello world", emb=_DEFAULT_EMB, **kw):
    return MemoryRecord(user_id="u1", content=content, embedding=list(emb), **kw)


# Orchestrator semantics are backend-agnostic; these use in-memory split stores (the
# Tier-2-shaped path with co_transactional=False). The SQLite collapse is covered in
# test_sqlite_tier0.py and by the parametrized e2e suite.


def _stores():
    return InMemoryRecordStore(), InMemoryVectorStore()


async def test_dual_write_happy_path():
    rs, vs = _stores()
    stored = await dual_write(rs, vs, _rec())
    fetched = await rs.fetch_by_ids("u1", [stored.id])
    assert fetched and fetched[0].id == stored.id
    assert await vs.search("u1", [1.0, 0.0], 5, _filters())


async def test_vector_failure_keeps_record_and_raises_503_detail():
    rs = InMemoryRecordStore()
    vs = FailingVectorStore()
    rec = _rec()
    with pytest.raises(DualWriteVectorError) as ei:
        await dual_write(rs, vs, rec)
    # The exact era-core detail string is preserved.
    assert ei.value.detail == DETAIL_MEMORY
    # Fail-fast-keep-record: the record is still durably present.
    fetched = await rs.fetch_by_ids("u1", [rec.id])
    assert fetched and fetched[0].id == rec.id


async def test_was_inserted_false_skips_vector_write():
    rs = InMemoryRecordStore()
    vs = CountingVectorStore()
    r1 = _rec(content_hash="h1")
    r2 = MemoryRecord(user_id="u1", content="hello world", embedding=[1.0, 0.0], content_hash="h1")
    await dual_write(rs, vs, r1)
    await dual_write(rs, vs, r2)  # ON-CONFLICT dedup -> was_inserted False
    assert vs.insert_calls == 1  # second write skipped the vector insert


async def test_record_with_no_embedding_skips_vector():
    rs = InMemoryRecordStore()
    vs = CountingVectorStore()
    await dual_write(rs, vs, _rec(emb=[]))
    assert vs.insert_calls == 0


async def test_dual_write_batch():
    rs, vs = _stores()
    recs = [_rec(content=f"item {i}", content_hash=f"h{i}") for i in range(3)]
    stored = await dual_write_batch(rs, vs, recs)
    assert len(stored) == 3


async def test_soft_delete_swallows_vector_failure():
    rs = InMemoryRecordStore()
    vs = FailingVectorStore()
    rec = _rec()
    async with rs.unit_of_work() as uow:
        await rs.insert_memory(uow, rec)
    # delete must not raise even though vector delete is best-effort
    deleted = await soft_delete(rs, vs, "u1", rec.id)
    assert deleted is True
    assert not await rs.fetch_by_ids("u1", [rec.id])


async def test_unit_of_work_rolls_back_on_exception():
    rs = InMemoryRecordStore()
    rec = _rec()
    with pytest.raises(RuntimeError):
        async with rs.unit_of_work() as uow:
            await rs.insert_memory(uow, rec)
            raise RuntimeError("boom")
    assert not await rs.fetch_by_ids("u1", [rec.id])  # rolled back


def _filters():
    from era_memory.models import SearchFilters

    return SearchFilters()
