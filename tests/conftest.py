from __future__ import annotations

import itertools

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
from era_memory.wiring import build_memory

# Fixed dimension for vector-store conformance (sqlite vec0 tables are fixed-dim).
VEC_DIM = 8


@pytest.fixture
def vec_dim() -> int:
    return VEC_DIM


@pytest.fixture
def settings() -> Settings:
    return Settings(embedding_dimensions=64)


# --- per-port fixtures, parametrized over backends: the SAME conformance suite runs
#     against in-memory AND SQLite. "Works on a laptop" == "works in prod". ---


@pytest.fixture(params=["memory", "sqlite"])
def record_store(request, tmp_path):
    if request.param == "memory":
        return InMemoryRecordStore()
    from era_memory.adapters.sqlite import open_sqlite_stores

    rs, _ = open_sqlite_stores(str(tmp_path / "rs.db"), VEC_DIM, "test")
    return rs


@pytest.fixture(params=["memory", "sqlite"])
def vector_store(request, tmp_path):
    if request.param == "memory":
        return InMemoryVectorStore()
    from era_memory.adapters.sqlite import open_sqlite_stores

    _, vs = open_sqlite_stores(str(tmp_path / "vs.db"), VEC_DIM, "test")
    return vs


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


@pytest.fixture(params=["memory", "sqlite"])
def memory(request, tmp_path, clock) -> Memory:
    """The wired facade, exercised against BOTH the in-memory and the SQLite collapse."""
    if request.param == "memory":
        return build_memory(tier=0, clock=clock)
    return build_memory(tier=0, db_path=str(tmp_path / "mem.db"), clock=clock)
