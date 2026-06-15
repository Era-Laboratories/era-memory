"""
Optional HTTP surface (FastAPI). The library is fully usable in-process without this;
the app exists for standalone deployment (e.g. era-labs-tools Cloud Run / GKE).

Needs the ``[server]`` extra. Routes mirror era-core's shapes so Tier 2 can stay
API-compatible: POST /api/memories, POST /api/memories/search, GET /health, GET /ready.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .memory import Memory
from .models import MemoryRecord, MemoryType, SearchFilters, SearchRequest, SearchStrategy


class CreateMemoryBody(BaseModel):
    content: str = Field(min_length=1)
    memory_type: MemoryType = MemoryType.EPISODE
    importance_score: float = 0.5
    confidence: float = 1.0
    experience_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchBody(BaseModel):
    query: str
    strategy: SearchStrategy = SearchStrategy.HYBRID
    limit: int = 10
    memory_type: Optional[MemoryType] = None
    experience_id: Optional[str] = None


async def _current_user(request: Request) -> str:
    memory: Memory = request.app.state.memory
    user_id = await memory.auth.authenticate(dict(request.headers)) if memory.auth else None
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return user_id


def create_app(memory: Optional[Memory] = None) -> FastAPI:
    """Build the app. Pass a prebuilt ``memory`` (tests) or omit to build from env on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if memory is not None:
            app.state.memory = memory
        else:
            app.state.memory = await _build_from_env()
        try:
            yield
        finally:
            backend = getattr(app.state.memory, "_pg_backend", None)
            if backend is not None:
                await backend.close()

    app = FastAPI(title="era-memory", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        mem: Memory = request.app.state.memory
        return {"status": "ready", "vector_store_connected": mem.vector_store.is_connected()}

    @app.post("/api/memories")
    async def create_memory(body: CreateMemoryBody, user_id: str = Depends(_current_user)):
        mem: Memory = app.state.memory
        stored = await mem.store(
            MemoryRecord(
                user_id=user_id,
                content=body.content,
                memory_type=body.memory_type,
                importance_score=body.importance_score,
                confidence=body.confidence,
                experience_id=body.experience_id,
                metadata=body.metadata,
            )
        )
        return {"id": stored.id, "memory_type": stored.memory_type.value, "source_type": "memory"}

    @app.post("/api/memories/search")
    async def search_memories(body: SearchBody, user_id: str = Depends(_current_user)):
        mem: Memory = app.state.memory
        resp = await mem.search(
            SearchRequest(
                user_id=user_id,
                query=body.query,
                strategy=body.strategy,
                limit=body.limit,
                filters=SearchFilters(
                    memory_type=body.memory_type, experience_id=body.experience_id
                ),
            )
        )
        return {
            "results": [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "memory_type": r.memory_type.value,
                    "source_type": r.source_type,
                }
                for r in resp.results
            ],
            "strategy": resp.strategy.value,
            "total_candidates": resp.total_candidates,
            "latency_ms": resp.latency_ms,
        }

    return app


async def _build_from_env() -> Memory:
    from .adapters.auth import BearerAuth
    from .wiring import build_memory, build_memory_async

    tier = int(os.environ.get("MEMORY_TIER", "0"))
    token = os.environ.get("MEMORY_BEARER_TOKEN")
    embedder = _embedder_from_env()

    if tier == 0:
        mem = build_memory(tier=0, db_path=os.environ.get("MEMORY_DB_PATH"), embedder=embedder)
    elif tier == 1:
        mem = await build_memory_async(
            tier=1, dsn=os.environ["MEMORY_PG_DSN"], embedder=embedder, reset=False
        )
    else:
        raise RuntimeError(f"unsupported MEMORY_TIER for HTTP app: {tier}")

    if token:
        mem.auth = BearerAuth(token)
    return mem


def _embedder_from_env():
    url = os.environ.get("MEMORY_EMBEDDING_URL")
    if not url:
        return None  # falls back to the dev hash embedder in wiring
    from .adapters.openai import OpenAICompatibleEmbedder

    return OpenAICompatibleEmbedder(
        base_url=url,
        model=os.environ.get("MEMORY_EMBEDDING_MODEL", "text-embedding-3-small"),
        dimensions=int(os.environ.get("MEMORY_EMBEDDING_DIMENSIONS", "1536")),
        api_key=os.environ.get("MEMORY_EMBEDDING_API_KEY"),
    )
