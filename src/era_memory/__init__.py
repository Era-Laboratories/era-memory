"""era-memory: portable, tier-selected memory system (ports & adapters).

Top-level convenience exports. The composition root (``build_memory`` /
``build_memory_async``) and the public data models are importable directly:

    from era_memory import build_memory, MemoryRecord, SearchRequest

Backend-specific adapters (SQLite, Postgres, OpenAI-compatible embedder, …) live under
``era_memory.adapters.*`` and are imported lazily so the base package keeps zero
third-party dependencies — import them explicitly only when you use that backend.
"""

from .config import Settings
from .memory import Memory
from .models import (
    ExtractedMemory,
    ExtractionResult,
    MemoryRecord,
    MemoryType,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchStrategy,
    SessionPayload,
    VectorRecord,
)
from .wiring import build_memory, build_memory_async

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "Memory",
    "build_memory",
    "build_memory_async",
    "MemoryRecord",
    "MemoryType",
    "SearchFilters",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearchStrategy",
    "SessionPayload",
    "ExtractedMemory",
    "ExtractionResult",
    "VectorRecord",
    "__version__",
]
