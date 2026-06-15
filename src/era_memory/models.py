"""
Core data models — stdlib dataclasses only (no pydantic in the base package).

These are the contract that flows across every port. Field names that are part of the
era-core API surface (source_type, importance_score, confidence, memory_type) are kept
identical so Tier 2 can stay API-compatible.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


class MemoryType(str, enum.Enum):
    EPISODE = "episode"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    ENTITY = "entity"


class SearchStrategy(str, enum.Enum):
    HYBRID = "hybrid"
    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    DEEP = "deep"


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class MemoryRecord:
    """A stored memory. ``embedding`` is optional — a record may exist without a vector."""

    user_id: str
    content: str
    id: str = field(default_factory=_new_id)
    memory_type: MemoryType = MemoryType.EPISODE
    status: str = "active"
    embedding: Optional[list[float]] = None
    importance_score: float = 0.5
    confidence: float = 1.0
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    category: Optional[str] = None
    temporal_anchor: Optional[str] = None
    content_hash: Optional[str] = None
    session_id: Optional[str] = None
    source_memory_id: Optional[str] = None
    experience_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0  # epoch seconds; stamped by the store/caller (no Date.now in core)
    updated_at: float = 0.0
    last_accessed_at: float = 0.0
    access_count: int = 0

    def to_vector_record(self) -> "VectorRecord":
        return VectorRecord(
            id=self.id,
            user_id=self.user_id,
            memory_type=self.memory_type,
            embedding=self.embedding or [],
            importance_score=self.importance_score,
            created_at=int(self.created_at),
            experience_id=self.experience_id or "",
        )


@dataclass
class VectorRecord:
    """The projection written to the vector index (mirrors era-core's Milvus payload)."""

    id: str
    user_id: str
    embedding: list[float]
    memory_type: MemoryType = MemoryType.EPISODE
    importance_score: float = 0.5
    created_at: int = 0
    experience_id: str = ""


@dataclass
class SearchFilters:
    memory_type: Optional[MemoryType] = None
    experience_id: Optional[str] = None
    created_after: Optional[int] = None
    created_before: Optional[int] = None


@dataclass
class SearchRequest:
    user_id: str
    query: str
    query_embedding: Optional[list[float]] = None
    strategy: SearchStrategy = SearchStrategy.HYBRID
    limit: int = 10
    filters: SearchFilters = field(default_factory=SearchFilters)
    recency_weight: float = 0.3


@dataclass
class SearchResult:
    id: str
    score: float
    content: str
    memory_type: MemoryType = MemoryType.EPISODE
    importance_score: float = 0.5
    confidence: float = 1.0
    source_type: str = "memory"  # "memory" | "raw_conversation" (deep search)
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    results: list[SearchResult]
    strategy: SearchStrategy
    total_candidates: int = 0
    latency_ms: float = 0.0


@dataclass
class ExtractedMemory:
    content: str
    memory_type: MemoryType = MemoryType.EPISODE
    importance_score: float = 0.5
    confidence: float = 1.0
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    memories: list[ExtractedMemory] = field(default_factory=list)
    episode_summary: Optional[str] = None
    entity_map: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionPayload:
    """A raw conversation handed to the encoder pipeline."""

    user_id: str
    session_id: str
    conversation: str
    experience_id: Optional[str] = None
