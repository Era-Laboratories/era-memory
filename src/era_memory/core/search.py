"""
Hybrid search — vector ANN + lexical, fused by RRF, scored by importance * recency.

Backend-agnostic: drives the VectorStore + RecordStore ports and the pure rrf logic.
Reproduces era-core's ranking (k=60, 0.6/0.4, 30-day half-life) and the deep-search
weak-result trigger (top cosine < threshold OR fewer than `limit` results).
"""

from __future__ import annotations

from ..config import Settings
from ..models import (
    MemoryRecord,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchStrategy,
)
from ..ports import Embedder, RecordStore, VectorStore
from .logic.rrf import final_score, recency_decay, rrf_fuse

_DAY_SECONDS = 86400.0


async def hybrid_search(
    record_store: RecordStore,
    vector_store: VectorStore,
    embedder: Embedder,
    request: SearchRequest,
    settings: Settings,
    *,
    now: float,
) -> SearchResponse:
    candidate_limit = request.limit * 3

    # Resolve the query embedding (unless lexical-only).
    query_embedding = request.query_embedding
    if request.strategy != SearchStrategy.BM25_ONLY and query_embedding is None:
        query_embedding = (await embedder.embed([request.query]))[0]

    semantic: list[tuple[str, float]] = []
    lexical: list[tuple[str, float]] = []
    if request.strategy in (SearchStrategy.HYBRID, SearchStrategy.VECTOR_ONLY, SearchStrategy.DEEP):
        semantic = await vector_store.search(
            request.user_id, query_embedding or [], candidate_limit, request.filters
        )
    if request.strategy in (SearchStrategy.HYBRID, SearchStrategy.BM25_ONLY, SearchStrategy.DEEP):
        lexical = await record_store.lexical_search(
            request.user_id, request.query, request.filters, candidate_limit
        )

    fused = rrf_fuse(
        semantic,
        lexical,
        k=settings.rrf_k,
        semantic_weight=settings.rrf_semantic_weight,
        lexical_weight=settings.rrf_lexical_weight,
    )
    total_candidates = len(fused)

    records = {
        r.id: r
        for r in await record_store.fetch_by_ids(request.user_id, list(fused.keys()))
    }
    top_cosine = max((s for _, s in semantic), default=0.0)

    scored: list[SearchResult] = []
    for doc_id, base in fused.items():
        rec = records.get(doc_id)
        if rec is None:
            continue
        age_days = max(0.0, (now - rec.created_at) / _DAY_SECONDS) if rec.created_at else 0.0
        recency = recency_decay(age_days, half_life_days=settings.recency_half_life_days)
        score = final_score(
            base, rec.importance_score, recency, recency_weight=settings.recency_weight
        )
        scored.append(_to_result(rec, score))

    scored.sort(key=lambda r: r.score, reverse=True)
    results = scored[: request.limit]

    # Deep-search weak-result signal (the fallback itself lands with the session adapters).
    weak = len(results) < request.limit or top_cosine < settings.deep_search_threshold
    strategy = SearchStrategy.DEEP if (request.strategy == SearchStrategy.DEEP and weak) else request.strategy

    return SearchResponse(
        results=results,
        strategy=strategy,
        total_candidates=total_candidates,
        latency_ms=0.0,  # stamped by the caller/HTTP layer
    )


def _to_result(rec: MemoryRecord, score: float) -> SearchResult:
    return SearchResult(
        id=rec.id,
        score=score,
        content=rec.content,
        memory_type=rec.memory_type,
        importance_score=rec.importance_score,
        confidence=rec.confidence,
        source_type="memory",
        created_at=rec.created_at,
        experience_id=rec.experience_id,
        metadata=rec.metadata,
    )
