"""
Encoder pipeline — conversation -> stored memories.

Stages (mirrors era-core era-memory-encoder): entropy gate -> extract -> embed ->
cosine-dedup against existing vectors -> dual-write. Backend-agnostic; the Queue adapter
decides whether this runs in-process (Tier 0) or off a stream (Tier 1/2).
"""

from __future__ import annotations

from ..config import Settings
from ..models import MemoryRecord, SessionPayload
from ..ports import Embedder, Extractor, RecordStore, VectorStore
from .logic.dedup import is_duplicate
from .logic.entropy import passes_entropy
from .orchestration import dual_write


async def encode_session(
    payload: SessionPayload,
    *,
    record_store: RecordStore,
    vector_store: VectorStore,
    embedder: Embedder,
    extractor: Extractor,
    settings: Settings,
    now: float,
) -> list[MemoryRecord]:
    """Run one session through the pipeline. Returns the memories actually written."""
    if not passes_entropy(payload.conversation, threshold=settings.entropy_threshold):
        return []

    extraction = await extractor.extract(payload)
    candidates = extraction.memories[: settings.max_memories_per_session]
    if not candidates:
        return []

    texts = [c.content for c in candidates]
    embeddings = await embedder.embed(texts)

    written: list[MemoryRecord] = []
    seen_embeddings: list[list[float]] = []
    for cand, emb in zip(candidates, embeddings):
        # Dedup against this batch AND existing nearby memories.
        existing = await vector_store.search(
            payload.user_id, emb, 5, _no_filters()
        )
        nearby = [e for e in seen_embeddings]
        if existing:
            top_sim = max(s for _, s in existing)
            if top_sim >= settings.dedup_similarity_threshold:
                continue
        if is_duplicate(emb, nearby, threshold=settings.dedup_similarity_threshold):
            continue

        record = MemoryRecord(
            user_id=payload.user_id,
            content=cand.content,
            memory_type=cand.memory_type,
            importance_score=cand.importance_score,
            confidence=cand.confidence,
            entities=cand.entities,
            topics=cand.topics,
            embedding=emb,
            session_id=payload.session_id,
            experience_id=payload.experience_id,
            created_at=now,
            updated_at=now,
        )
        stored = await dual_write(record_store, vector_store, record)
        written.append(stored)
        seen_embeddings.append(emb)

    return written


def _no_filters():
    from ..models import SearchFilters

    return SearchFilters()
