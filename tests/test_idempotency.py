"""
Write-time idempotency + content_hash population (0.1.2).

Exercised against every backend via the parametrized ``memory`` fixture (in-memory,
the SQLite collapse, and pgvector when a DSN is provided) — "works on a laptop" ==
"works in prod".
"""

from __future__ import annotations

import hashlib

from era_memory.models import MemoryRecord


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def test_store_populates_content_hash(memory):
    content = "Ada prefers dark roast coffee"
    stored = await memory.store(MemoryRecord(user_id="u1", content=content))
    assert stored.content_hash == _sha256(content)


async def test_content_hash_persists_and_round_trips(memory):
    content = "quarterly revenue report"
    stored = await memory.store(MemoryRecord(user_id="u1", content=content))
    # Re-read straight from the record store to prove the adapter persisted the hash.
    fetched = await memory.record_store.fetch_by_ids("u1", [stored.id])
    assert fetched and fetched[0].content_hash == _sha256(content)


async def test_duplicate_create_is_idempotent(memory):
    first, dedup_first = await memory.store_with_dedup(
        MemoryRecord(user_id="u1", content="exact same body")
    )
    second, dedup_second = await memory.store_with_dedup(
        MemoryRecord(user_id="u1", content="exact same body")
    )
    assert dedup_first is False  # fresh insert
    assert dedup_second is True  # retry collapsed onto the first record
    assert second.id == first.id  # same row returned, untouched


async def test_same_content_different_experience_id_collapses_onto_first(memory):
    first, _ = await memory.store_with_dedup(
        MemoryRecord(user_id="u1", content="byte identical body", experience_id="exp-A")
    )
    second, deduped = await memory.store_with_dedup(
        MemoryRecord(user_id="u1", content="byte identical body", experience_id="exp-B")
    )
    # Dedup keys on (user_id, content_hash) only — the first record wins untouched.
    assert deduped is True
    assert second.id == first.id
    assert second.experience_id == "exp-A"


async def test_same_content_different_user_not_deduped(memory):
    alice, dedup_alice = await memory.store_with_dedup(
        MemoryRecord(user_id="alice", content="shared sentence")
    )
    bob, dedup_bob = await memory.store_with_dedup(
        MemoryRecord(user_id="bob", content="shared sentence")
    )
    assert dedup_alice is False
    assert dedup_bob is False  # different user → different (user_id, content_hash) key
    assert alice.id != bob.id
