"""Domain errors. Kept free of any backend import."""

from __future__ import annotations


class EraMemoryError(Exception):
    """Base for all era-memory errors."""


class VectorStoreWriteError(EraMemoryError):
    """A VectorStore adapter failed to persist vectors. Raised by adapters."""


class DualWriteVectorError(EraMemoryError):
    """
    The record committed durably but the vector write then failed.

    Mirrors era-core's fail-fast-keep-record semantics: the record is NOT rolled back.
    The orchestrator raises this; the HTTP layer maps it to 503 with ``detail``.
    ``stored`` is the already-committed record so callers can surface/repair it.
    """

    def __init__(self, stored, detail: str) -> None:
        super().__init__(detail)
        self.stored = stored
        self.detail = detail


class EmbeddingDimensionMismatch(EraMemoryError):
    """An embedding's length does not match the store's configured dimension."""


class ConfigurationError(EraMemoryError):
    """Invalid or incompatible configuration / wiring."""
