from __future__ import annotations

import itertools
import os

import pytest

from era_memory.adapters.memory import (
    HeuristicExtractor,
    InMemoryEmbedder,
    InMemoryRecordStore,
    InMemoryVectorStore,
    LocalKMS,
)
from era_memory.config import Settings
from era_memory.memory import Memory
from era_memory.wiring import build_memory, build_memory_async

# Fixed dimension for store conformance (sqlite/pg vector columns are fixed-dim).
VEC_DIM = 8

# Postgres is included in the parametrized backends only when a DSN is provided.
PG_DSN = os.environ.get("MEMORY_TEST_PG_DSN")
_STORE_BACKENDS = ["memory", "sqlite"] + (["postgres"] if PG_DSN else [])


@pytest.fixture
def vec_dim() -> int:
    return VEC_DIM


@pytest.fixture
def settings() -> Settings:
    return Settings(embedding_dimensions=64)


# --- per-port fixtures, parametrized over backends: the SAME conformance suite runs
#     against in-memory, SQLite, AND Postgres. "Works on a laptop" == "works in prod". ---


@pytest.fixture(params=_STORE_BACKENDS)
async def record_store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryRecordStore()
    elif request.param == "sqlite":
        from era_memory.adapters.sqlite import open_sqlite_stores

        rs, _ = open_sqlite_stores(str(tmp_path / "rs.db"), VEC_DIM, "test")
        yield rs
    else:
        from era_memory.adapters.postgres import open_pg_stores

        rs, _, backend = await open_pg_stores(PG_DSN, VEC_DIM, "test", reset=True)
        yield rs
        await backend.close()


@pytest.fixture(params=_STORE_BACKENDS)
async def vector_store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryVectorStore()
    elif request.param == "sqlite":
        from era_memory.adapters.sqlite import open_sqlite_stores

        _, vs = open_sqlite_stores(str(tmp_path / "vs.db"), VEC_DIM, "test")
        yield vs
    else:
        from era_memory.adapters.postgres import open_pg_stores

        _, vs, backend = await open_pg_stores(PG_DSN, VEC_DIM, "test", reset=True)
        yield vs
        await backend.close()


@pytest.fixture
def embedder():
    return InMemoryEmbedder(dim=64)


@pytest.fixture
def kms():
    return LocalKMS()


@pytest.fixture
def extractor():
    return HeuristicExtractor()


@pytest.fixture
def clock():
    """Deterministic monotonic clock so tests never depend on wall time."""
    counter = itertools.count(1_700_000_000)
    return lambda: float(next(counter))


@pytest.fixture(params=_STORE_BACKENDS)
async def memory(request, tmp_path, clock) -> Memory:
    """The wired facade, exercised against in-memory, the SQLite collapse, and pgvector."""
    if request.param == "memory":
        yield build_memory(tier=0, clock=clock)
    elif request.param == "sqlite":
        yield build_memory(tier=0, db_path=str(tmp_path / "mem.db"), clock=clock)
    else:
        mem = await build_memory_async(tier=1, dsn=PG_DSN, reset=True, clock=clock)
        yield mem
        await mem._pg_backend.close()  # type: ignore[attr-defined]
