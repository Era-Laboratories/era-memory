"""
Tier-0 SQLite specifics: the single-file dual-write collapse, atomicity, the (model,dim)
lock-in guard, and persistence across process restarts. Needs the [sqlite] extra.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("sqlite_vec")

from era_memory.adapters.sqlite import SqliteVectorStore, open_sqlite_stores  # noqa: E402
from era_memory.errors import ConfigurationError  # noqa: E402
from era_memory.models import MemoryRecord, SearchRequest  # noqa: E402
from era_memory.wiring import build_memory  # noqa: E402


def test_vector_store_is_co_transactional():
    assert SqliteVectorStore.co_transactional is True


async def test_store_persists_record_and_vector_in_one_file(tmp_path):
    path = str(tmp_path / "m.db")
    mem = build_memory(tier=0, db_path=path)
    stored = await mem.store(MemoryRecord(user_id="u1", content="dark roast coffee"))

    con = sqlite3.connect(path)
    assert con.execute("SELECT count(*) FROM memories WHERE id=?", (stored.id,)).fetchone()[0] == 1
    res = await mem.search(SearchRequest(user_id="u1", query="coffee"))
    assert any(r.id == stored.id for r in res.results)


async def test_collapse_is_atomic_on_vector_failure(tmp_path):
    """If the in-uow vector write fails, the record rolls back too — no orphan, no 503."""
    path = str(tmp_path / "m.db")
    mem = build_memory(tier=0, db_path=path)

    def boom(*a, **k):
        raise RuntimeError("vector write failed inside uow")

    mem.vector_store._upsert = boom  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        await mem.store(MemoryRecord(user_id="u1", content="should roll back entirely"))

    con = sqlite3.connect(path)
    assert con.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM memories_fts").fetchone()[0] == 0


def test_model_dim_lock_in_guard(tmp_path):
    path = str(tmp_path / "m.db")
    open_sqlite_stores(path, 64, "model-a")
    with pytest.raises(ConfigurationError):
        open_sqlite_stores(path, 32, "model-a")  # dim changed
    with pytest.raises(ConfigurationError):
        open_sqlite_stores(path, 64, "model-b")  # model changed


async def test_persistence_across_reopen(tmp_path):
    path = str(tmp_path / "m.db")
    mem1 = build_memory(tier=0, db_path=path)
    stored = await mem1.store(
        MemoryRecord(user_id="u1", content="persistent currywurst memory")
    )
    # A brand-new Memory on the same file (simulates a process restart).
    mem2 = build_memory(tier=0, db_path=path)
    res = await mem2.search(SearchRequest(user_id="u1", query="currywurst"))
    assert any(r.id == stored.id for r in res.results)


async def test_persisted_kms_key_round_trips(tmp_path):
    """The persisted local KMS key survives a restart (M1 fix vs ephemeral)."""
    key_path = str(tmp_path / "kms.key")
    mem1 = build_memory(tier=0, db_path=str(tmp_path / "a.db"), kms_key_path=key_path)
    dek = await mem1.kms.generate_dek()
    wrapped = await mem1.kms.wrap_dek(dek, "u1:content:1")

    mem2 = build_memory(tier=0, db_path=str(tmp_path / "b.db"), kms_key_path=key_path)
    assert await mem2.kms.unwrap_dek(wrapped, "u1:content:1") == dek
