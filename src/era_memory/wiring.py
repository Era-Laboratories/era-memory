"""
Composition root. ``MEMORY_TIER`` -> adapter set -> a wired Memory.

Tier 0 uses SQLite + sqlite-vec when a ``db_path`` is given (single-file, single-tx
collapse), else the zero-dependency in-memory adapters. Tiers 1/2 raise until their
adapters land (M2/M3). Wiring is the single place that knows the tier->adapter mapping.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Callable, Optional, Union

from .adapters.memory import (
    HeuristicExtractor,
    InMemoryBlobStore,
    InMemoryEmbedder,
    InMemoryRecordStore,
    InMemoryVectorStore,
    InProcessQueue,
    LocalKMS,
    NoopAuth,
    NoopTelemetry,
)
from .config import Settings
from .embedders import resolve_embedder
from .errors import ConfigurationError
from .memory import Memory
from .ports import Embedder

# What may be passed as `embedder=`: a concrete Embedder, the string "auto" (resolve + download
# a local model if needed), or None (resolve-if-cached/endpoint, else the dev stand-in).
EmbedderArg = Union[Embedder, str, None]


def _persistent_local_kms(key_path: Optional[str]) -> LocalKMS:
    """Persist the master key so encrypted data survives restarts (M1 fix)."""
    if not key_path:
        return LocalKMS()
    p = Path(key_path).expanduser()
    if p.exists():
        return LocalKMS(master_key=p.read_bytes())
    key = os.urandom(32)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(key)
    os.chmod(p, 0o600)
    return LocalKMS(master_key=key)


def _reconcile_embedding_dim(settings: Settings, embedder: Optional[Embedder]) -> None:
    """A store is locked to one (model, dim) for its whole life (see docs/adr/0001).

    When an explicit embedder is supplied it is the source of truth for (model, dim).
    If the operator *also* pinned MEMORY_EMBEDDING_DIMENSIONS to a conflicting value,
    fail fast here with a clear message instead of later at the first write/reopen.
    """
    if embedder is None:
        return
    env_dim = os.environ.get("MEMORY_EMBEDDING_DIMENSIONS")
    if env_dim is not None and int(env_dim) != embedder.dimensions:
        raise ConfigurationError(
            f"Embedder dimension {embedder.dimensions} != MEMORY_EMBEDDING_DIMENSIONS "
            f"{env_dim}. A store is pinned to one (model, dim) for its lifetime — pick a "
            "single source of truth: unset MEMORY_EMBEDDING_DIMENSIONS, or set it to "
            f"{embedder.dimensions}."
        )
    # Keep settings consistent with the live embedder for downstream/telemetry.
    settings.embedding_dimensions = embedder.dimensions
    settings.embedding_model = embedder.model_id


_DEV_EMBEDDER_WARNING = (
    "era-memory is using the non-semantic dev embedder (hashed bag-of-words) — search will "
    "only match on shared literal words. For real semantic retrieval: run `era-memory setup` "
    "to download a local model, pass build_memory(embedder='auto'), or set MEMORY_EMBEDDING_URL "
    "to an embeddings endpoint. See INSTALL.md."
)


def _select_embedder(settings: Settings, embedder: EmbedderArg) -> Embedder:
    """Resolve `embedder=` into a concrete Embedder (see EmbedderArg / resolve_embedder)."""
    if isinstance(embedder, Embedder):
        _reconcile_embedding_dim(settings, embedder)
        return embedder
    if embedder == "auto":
        resolved = resolve_embedder(settings, allow_download=True)
        if resolved is None:
            raise ConfigurationError(
                "embedder='auto' needs the [localembed] extra: "
                "pip install 'era-memory[localembed]' (or set MEMORY_EMBEDDING_URL)."
            )
        _reconcile_embedding_dim(settings, resolved)
        return resolved
    if embedder is not None:
        raise ConfigurationError(
            f"embedder must be an Embedder, 'auto', or None; got {embedder!r}"
        )
    # None: use a configured endpoint or an already-cached local model — never download here.
    resolved = resolve_embedder(settings, allow_download=False)
    if resolved is not None:
        _reconcile_embedding_dim(settings, resolved)
        return resolved
    warnings.warn(_DEV_EMBEDDER_WARNING, stacklevel=3)
    return InMemoryEmbedder(
        dim=settings.embedding_dimensions, model_id=settings.embedding_model
    )


def build_memory(
    tier: Optional[int] = None,
    settings: Optional[Settings] = None,
    *,
    db_path: Optional[str] = None,
    embedder: EmbedderArg = None,
    kms_key_path: Optional[str] = None,
    clock: Callable[[], float] = time.time,
) -> Memory:
    settings = settings or Settings.from_env()
    if tier is not None:
        settings.tier = tier

    if settings.tier == 0:
        return _build_tier0(settings, db_path, embedder, kms_key_path, clock)
    if settings.tier == 1:
        raise ConfigurationError("Tier 1 requires async wiring — use build_memory_async().")
    if settings.tier == 2:
        raise NotImplementedError("Tier 2 (Milvus/vLLM/Redis) adapters land in M3.")
    raise ConfigurationError(f"Unknown MEMORY_TIER: {settings.tier}")


async def build_memory_async(
    tier: Optional[int] = None,
    settings: Optional[Settings] = None,
    *,
    dsn: Optional[str] = None,
    db_path: Optional[str] = None,
    embedder: EmbedderArg = None,
    kms_key_path: Optional[str] = None,
    reset: bool = False,
    clock: Callable[[], float] = time.time,
) -> Memory:
    """Async composition root (needed for Tier 1's connection pool). Tier 0 delegates sync."""
    settings = settings or Settings.from_env()
    if tier is not None:
        settings.tier = tier

    if settings.tier == 0:
        return _build_tier0(settings, db_path, embedder, kms_key_path, clock)
    if settings.tier == 1:
        return await _build_tier1(settings, dsn, embedder, kms_key_path, reset, clock)
    if settings.tier == 2:
        raise NotImplementedError("Tier 2 (Milvus/vLLM/Redis) adapters land in M3.")
    raise ConfigurationError(f"Unknown MEMORY_TIER: {settings.tier}")


async def _build_tier1(
    settings: Settings,
    dsn: Optional[str],
    embedder: EmbedderArg,
    kms_key_path: Optional[str],
    reset: bool,
    clock: Callable[[], float],
) -> Memory:
    if not dsn:
        raise ConfigurationError("Tier 1 requires a Postgres DSN.")
    from .adapters.postgres import open_pg_stores

    # Production passes an OpenAICompatibleEmbedder; "auto"/cached/endpoint also resolve here.
    embedder = _select_embedder(settings, embedder)
    record_store, vector_store, backend = await open_pg_stores(
        dsn, embedder.dimensions, embedder.model_id, reset=reset
    )
    mem = Memory(
        record_store=record_store,
        vector_store=vector_store,
        embedder=embedder,
        extractor=HeuristicExtractor(),
        settings=settings,
        kms=_persistent_local_kms(kms_key_path),
        auth=NoopAuth(),
        blob_store=InMemoryBlobStore(),
        telemetry=NoopTelemetry(),
        clock=clock,
    )
    mem.queue = InProcessQueue(consumer=mem.encode_consumer)
    mem._pg_backend = backend  # type: ignore[attr-defined]  # for explicit close in tests
    return mem


def _build_tier0(
    settings: Settings,
    db_path: Optional[str],
    embedder: EmbedderArg,
    kms_key_path: Optional[str],
    clock: Callable[[], float],
) -> Memory:
    embedder = _select_embedder(settings, embedder)

    if db_path:
        from .adapters.sqlite import open_sqlite_stores

        record_store, vector_store = open_sqlite_stores(
            db_path, embedder.dimensions, embedder.model_id
        )
    else:
        record_store = InMemoryRecordStore()
        vector_store = InMemoryVectorStore()

    mem = Memory(
        record_store=record_store,
        vector_store=vector_store,
        embedder=embedder,
        extractor=HeuristicExtractor(),
        settings=settings,
        kms=_persistent_local_kms(kms_key_path),
        auth=NoopAuth(),
        blob_store=InMemoryBlobStore(),
        telemetry=NoopTelemetry(),
        clock=clock,
    )
    mem.queue = InProcessQueue(consumer=mem.encode_consumer)
    return mem
