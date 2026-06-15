"""
The Memory facade — the public, in-process API. Ties the ports together and stamps time.

Usage is backend-agnostic: the same calls work whether wired to in-memory, SQLite, or
Milvus adapters.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from .config import Settings
from .core.orchestration import dual_write, soft_delete
from .core.pipeline import encode_session
from .core.search import hybrid_search
from .models import MemoryRecord, SearchRequest, SearchResponse, SessionPayload
from .ports import (
    Auth,
    BlobStore,
    Embedder,
    Extractor,
    KMS,
    Queue,
    RecordStore,
    Telemetry,
    VectorStore,
)


class Memory:
    def __init__(
        self,
        *,
        record_store: RecordStore,
        vector_store: VectorStore,
        embedder: Embedder,
        extractor: Extractor,
        settings: Settings,
        kms: Optional[KMS] = None,
        auth: Optional[Auth] = None,
        blob_store: Optional[BlobStore] = None,
        telemetry: Optional[Telemetry] = None,
        queue: Optional[Queue] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.record_store = record_store
        self.vector_store = vector_store
        self.embedder = embedder
        self.extractor = extractor
        self.settings = settings
        self.kms = kms
        self.auth = auth
        self.blob_store = blob_store
        self.telemetry = telemetry
        self.queue = queue
        self._clock = clock

    async def store(self, record: MemoryRecord) -> MemoryRecord:
        """Embed (if needed), stamp time, and dual-write a single memory."""
        now = self._clock()
        if not record.created_at:
            record.created_at = now
            record.updated_at = now
        if record.embedding is None:
            record.embedding = (await self.embedder.embed([record.content]))[0]
        return await dual_write(self.record_store, self.vector_store, record)

    async def search(self, request: SearchRequest) -> SearchResponse:
        t0 = self._clock()
        resp = await hybrid_search(
            self.record_store,
            self.vector_store,
            self.embedder,
            request,
            self.settings,
            now=self._clock(),
        )
        resp.latency_ms = round((self._clock() - t0) * 1000.0, 1)
        return resp

    async def encode(self, payload: SessionPayload) -> list[MemoryRecord]:
        """Run a raw conversation through the extraction pipeline."""
        return await encode_session(
            payload,
            record_store=self.record_store,
            vector_store=self.vector_store,
            embedder=self.embedder,
            extractor=self.extractor,
            settings=self.settings,
            now=self._clock(),
        )

    async def encode_consumer(self, payload: SessionPayload) -> None:
        """Queue-consumer adapter: runs the pipeline and discards the return value."""
        await self.encode(payload)

    async def submit_session(self, payload: SessionPayload) -> None:
        """Hand a session to the Queue (in-process or durable)."""
        if self.queue is None:
            await self.encode(payload)
        else:
            await self.queue.submit(payload)

    async def delete(self, user_id: str, memory_id: str) -> bool:
        return await soft_delete(self.record_store, self.vector_store, user_id, memory_id)
