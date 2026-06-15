"""RecordStore conformance — every adapter must pass this unchanged."""

from __future__ import annotations

from era_memory.models import MemoryRecord, MemoryType, SearchFilters


def _rec(user="u1", content="dark roast coffee is great", **kw):
    return MemoryRecord(user_id=user, content=content, **kw)


async def test_insert_and_fetch_round_trip(record_store):
    rec = _rec()
    async with record_store.unit_of_work() as uow:
        stored, inserted = await record_store.insert_memory(uow, rec)
    assert inserted is True
    got = await record_store.fetch_by_ids("u1", [stored.id])
    assert got and got[0].content == rec.content


async def test_user_scoping_isolation(record_store):
    a = _rec(user="alice", content="alice secret")
    async with record_store.unit_of_work() as uow:
        await record_store.insert_memory(uow, a)
    # Bob cannot fetch Alice's record by id.
    assert await record_store.fetch_by_ids("bob", [a.id]) == []


async def test_content_hash_dedup_signals_not_inserted(record_store):
    async with record_store.unit_of_work() as uow:
        _, first = await record_store.insert_memory(uow, _rec(content_hash="h"))
        _, second = await record_store.insert_memory(uow, _rec(content_hash="h"))
    assert first is True
    assert second is False


async def test_lexical_search_ranks_and_filters(record_store):
    async with record_store.unit_of_work() as uow:
        await record_store.insert_memory(uow, _rec(content="coffee and espresso", content_hash="1"))
        await record_store.insert_memory(uow, _rec(content="tea and herbs", content_hash="2"))
        await record_store.insert_memory(
            uow,
            _rec(content="coffee beans roasting", memory_type=MemoryType.SEMANTIC, content_hash="3"),
        )
    hits = await record_store.lexical_search("u1", "coffee", SearchFilters(), 10)
    ids = [i for i, _ in hits]
    assert len(ids) == 2  # only the two coffee docs
    # filter by memory_type narrows results
    filtered = await record_store.lexical_search(
        "u1", "coffee", SearchFilters(memory_type=MemoryType.SEMANTIC), 10
    )
    assert len(filtered) == 1


async def test_soft_delete_removes_from_reads(record_store):
    rec = _rec(content_hash="x")
    async with record_store.unit_of_work() as uow:
        await record_store.insert_memory(uow, rec)
    assert await record_store.soft_delete("u1", rec.id) is True
    assert await record_store.fetch_by_ids("u1", [rec.id]) == []
    assert await record_store.lexical_search("u1", "coffee", SearchFilters(), 10) == []
    # deleting again is a no-op
    assert await record_store.soft_delete("u1", rec.id) is False


async def test_unit_of_work_rollback(record_store):
    rec = _rec(content_hash="rollback")
    try:
        async with record_store.unit_of_work() as uow:
            await record_store.insert_memory(uow, rec)
            raise RuntimeError("abort")
    except RuntimeError:
        pass
    assert await record_store.fetch_by_ids("u1", [rec.id]) == []
