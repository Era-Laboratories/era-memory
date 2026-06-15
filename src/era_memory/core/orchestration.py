"""
The dual-write orchestrator — the crux of the design.

Preserves era-core's exact write semantics across both store topologies:
  * Tier 0/1 (single store): VectorStore.insert is a no-op; the vector is written inside
    the RecordStore unit-of-work, so the whole write is one atomic transaction and the
    503 fail-fast path is structurally impossible.
  * Tier 2 (split store): the record commits FIRST (durable), THEN the vector is written;
    a vector failure raises DualWriteVectorError (-> 503) WITHOUT rolling back the record.

This is the only place that branches on topology; the ports stay identical.
"""

from __future__ import annotations

from ..errors import DualWriteVectorError, VectorStoreWriteError
from ..models import MemoryRecord
from ..ports import RecordStore, VectorStore

# era-core's three literal 503 detail strings — part of the API contract.
DETAIL_MEMORY = "Memory saved to database but vector index insert failed"
DETAIL_BATCH = "Memories saved to database but vector index batch insert failed"
DETAIL_SESSION = "session_content saved to database but vector index insert failed"


async def dual_write(
    record_store: RecordStore,
    vector_store: VectorStore,
    record: MemoryRecord,
    *,
    detail: str = DETAIL_MEMORY,
) -> MemoryRecord:
    """Insert a record then (if newly inserted and embedded) its vector. See module docstring."""
    async with record_store.unit_of_work() as uow:
        stored, was_inserted = await record_store.insert_memory(uow, record)
        # Single-store tiers: write the vector INSIDE the uow so the whole thing is atomic
        # and the 503 fail-fast path is structurally impossible.
        if vector_store.co_transactional and was_inserted and stored.embedding:
            await vector_store.insert_in_uow(uow, [stored.to_vector_record()])
    # Commit happened on context exit — matches era-core's commit-before-vector ordering.

    if not vector_store.co_transactional and was_inserted and stored.embedding:
        try:
            await vector_store.insert([stored.to_vector_record()])
        except VectorStoreWriteError as e:
            raise DualWriteVectorError(stored, detail) from e
    return stored


async def dual_write_batch(
    record_store: RecordStore,
    vector_store: VectorStore,
    records: list[MemoryRecord],
    *,
    detail: str = DETAIL_BATCH,
) -> list[MemoryRecord]:
    """Batch variant: all records commit in one uow, then one vector batch insert."""
    stored_inserted: list[MemoryRecord] = []
    async with record_store.unit_of_work() as uow:
        for record in records:
            stored, was_inserted = await record_store.insert_memory(uow, record)
            if was_inserted:
                stored_inserted.append(stored)
        if vector_store.co_transactional:
            to_index = [r.to_vector_record() for r in stored_inserted if r.embedding]
            if to_index:
                await vector_store.insert_in_uow(uow, to_index)

    if not vector_store.co_transactional:
        to_index = [r.to_vector_record() for r in stored_inserted if r.embedding]
        if to_index:
            try:
                await vector_store.insert(to_index)
            except VectorStoreWriteError as e:
                raise DualWriteVectorError(stored_inserted, detail) from e
    return stored_inserted


async def soft_delete(
    record_store: RecordStore,
    vector_store: VectorStore,
    user_id: str,
    memory_id: str,
) -> bool:
    """RecordStore is authoritative; VectorStore delete is best-effort (failure swallowed)."""
    deleted = await record_store.soft_delete(user_id, memory_id)
    try:
        await vector_store.delete([memory_id])
    except Exception:  # noqa: BLE001 - best-effort, mirrors era-core's swallow-and-warn
        pass
    return deleted
