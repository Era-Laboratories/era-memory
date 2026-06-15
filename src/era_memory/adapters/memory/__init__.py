"""
In-memory reference adapters for all nine ports.

Purpose: lock the port contracts (via the conformance suite) before any infrastructure,
and provide a zero-dependency tier for tests and quick embedding-in-an-app. The RecordStore
and VectorStore here are SEPARATE stores, so M0 exercises the split-store (Tier-2-shaped)
dual-write path; the collapsed single-store path arrives with the SQLite adapter (M1).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from contextlib import asynccontextmanager
from typing import Optional

from ...core.logic.dedup import cosine_similarity
from ...core.logic.entropy import entropy_score
from ...core.logic.matryoshka import l2_normalize
from ...errors import VectorStoreWriteError
from ...models import (
    ExtractedMemory,
    ExtractionResult,
    MemoryRecord,
    MemoryType,
    SearchFilters,
    SessionPayload,
    VectorRecord,
)
from ...ports import (
    Auth,
    BlobStore,
    Embedder,
    Extractor,
    KMS,
    Queue,
    RecordStore,
    Telemetry,
    UnitOfWork,
    VectorStore,
)
from ...ports.queue import Consumer

_TOKEN = re.compile(r"\w+")


def _matches(rec: MemoryRecord, f: SearchFilters) -> bool:
    if f.memory_type is not None and rec.memory_type != f.memory_type:
        return False
    if f.experience_id is not None and (rec.experience_id or "") != f.experience_id:
        return False
    if f.created_after is not None and rec.created_at < f.created_after:
        return False
    if f.created_before is not None and rec.created_at > f.created_before:
        return False
    return True


# --------------------------------------------------------------------------- RecordStore


class _MemUoW(UnitOfWork):
    pass


class InMemoryRecordStore(RecordStore):
    def __init__(self) -> None:
        self._by_user: dict[str, dict[str, MemoryRecord]] = {}
        self._hash_index: dict[tuple[str, str], str] = {}

    @asynccontextmanager
    async def unit_of_work(self):
        snap_users = {u: dict(d) for u, d in self._by_user.items()}
        snap_hash = dict(self._hash_index)
        try:
            yield _MemUoW()
        except BaseException:
            self._by_user = snap_users
            self._hash_index = snap_hash
            raise

    async def insert_memory(self, uow, record):
        users = self._by_user.setdefault(record.user_id, {})
        if record.content_hash:
            key = (record.user_id, record.content_hash)
            existing = self._hash_index.get(key)
            if existing is not None:
                return users[existing], False
            users[record.id] = record
            self._hash_index[key] = record.id
            return record, True
        users[record.id] = record
        return record, True

    async def fetch_by_ids(self, user_id, ids):
        users = self._by_user.get(user_id, {})
        out = [users[i] for i in ids if i in users and users[i].status == "active"]
        return out

    async def lexical_search(self, user_id, query, filters, limit):
        users = self._by_user.get(user_id, {})
        terms = set(_TOKEN.findall(query.lower()))
        scored: list[tuple[str, float]] = []
        for rec in users.values():
            if rec.status != "active" or not _matches(rec, filters):
                continue
            content_terms = _TOKEN.findall(rec.content.lower())
            if not content_terms:
                continue
            overlap = sum(1 for t in content_terms if t in terms)
            if overlap > 0:
                scored.append((rec.id, float(overlap)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def soft_delete(self, user_id, memory_id):
        rec = self._by_user.get(user_id, {}).get(memory_id)
        if rec is None or rec.status != "active":
            return False
        rec.status = "archived"
        return True


# --------------------------------------------------------------------------- VectorStore


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._vecs: dict[str, VectorRecord] = {}

    async def insert(self, records):
        for r in records:
            if not r.embedding:
                raise VectorStoreWriteError(f"empty embedding for {r.id}")
            self._vecs[r.id] = r
        return len(records)

    async def search(self, user_id, embedding, limit, filters):
        scored: list[tuple[str, float]] = []
        for r in self._vecs.values():
            if r.user_id != user_id:
                continue
            if filters.memory_type is not None and r.memory_type != filters.memory_type:
                continue
            if filters.experience_id is not None and r.experience_id != filters.experience_id:
                continue
            sim = cosine_similarity(embedding, r.embedding)
            scored.append((r.id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def delete(self, ids):
        n = 0
        for i in ids:
            if i in self._vecs:
                del self._vecs[i]
                n += 1
        return n


# ----------------------------------------------------------------------------- Embedder


class InMemoryEmbedder(Embedder):
    """Deterministic hashed bag-of-words embedding. Shared words -> higher cosine."""

    def __init__(self, dim: int = 64, model_id: str = "in-memory-hash") -> None:
        self._dim = dim
        self._model = model_id

    @property
    def dimensions(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model

    async def embed(self, texts):
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        return l2_normalize(vec)


# -------------------------------------------------------------------------------- Queue


class InProcessQueue(Queue):
    """Runs the consumer synchronously on submit — no Redis, no durability."""

    def __init__(self, consumer: Consumer) -> None:
        self._consumer = consumer

    async def submit(self, payload: SessionPayload) -> None:
        await self._consumer(payload)


# ---------------------------------------------------------------------------- Extractor


class HeuristicExtractor(Extractor):
    """Offline, no-LLM: each substantive line of the conversation becomes a candidate."""

    def __init__(self, min_words: int = 3) -> None:
        self._min_words = min_words

    async def extract(self, payload):
        memories: list[ExtractedMemory] = []
        for line in payload.conversation.splitlines():
            line = line.strip()
            if len(line.split()) < self._min_words:
                continue
            memories.append(
                ExtractedMemory(
                    content=line,
                    memory_type=MemoryType.EPISODE,
                    importance_score=min(1.0, max(0.1, entropy_score(line))),
                )
            )
        return ExtractionResult(memories=memories)


# ---------------------------------------------------------------------------- BlobStore


class InMemoryBlobStore(BlobStore):
    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    async def put(self, key, data):
        self._blobs[key] = data

    async def get(self, key):
        return self._blobs.get(key)


# ---------------------------------------------------------------------------------- KMS


class LocalKMS(KMS):
    """
    Dependency-free local/dev KMS: HMAC-SHA256 keystream + HMAC tag (authenticated wrap).
    Tier 0's real local KMS (M1) can swap to AESGCM via the [encryption] extra.
    """

    def __init__(self, master_key: Optional[bytes] = None) -> None:
        self._mk = master_key or os.urandom(32)

    @property
    def provider_name(self) -> str:
        return "local"

    async def generate_dek(self) -> bytes:
        return os.urandom(32)

    def _keystream(self, nonce: bytes, length: int) -> bytes:
        out = b""
        counter = 0
        while len(out) < length:
            out += hmac.new(
                self._mk, nonce + counter.to_bytes(4, "big"), hashlib.sha256
            ).digest()
            counter += 1
        return out[:length]

    async def wrap_dek(self, dek, aad):
        nonce = os.urandom(16)
        ks = self._keystream(nonce, len(dek))
        ct = bytes(a ^ b for a, b in zip(dek, ks))
        tag = hmac.new(self._mk, nonce + ct + aad.encode("utf-8"), hashlib.sha256).digest()
        return nonce + tag + ct

    async def unwrap_dek(self, wrapped, aad):
        nonce, tag, ct = wrapped[:16], wrapped[16:48], wrapped[48:]
        expected = hmac.new(self._mk, nonce + ct + aad.encode("utf-8"), hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("KMS unwrap authentication failure")
        ks = self._keystream(nonce, len(ct))
        return bytes(a ^ b for a, b in zip(ct, ks))


# --------------------------------------------------------------------------------- Auth


class NoopAuth(Auth):
    """Single-user / trusted-caller: returns X-User-Id header or a default."""

    def __init__(self, default_user: str = "local") -> None:
        self._default = default_user

    async def authenticate(self, headers):
        return headers.get("X-User-Id", self._default)


# ---------------------------------------------------------------------------- Telemetry


class NoopTelemetry(Telemetry):
    def event(self, name, **fields):
        return None
