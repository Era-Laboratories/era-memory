from __future__ import annotations

import abc
from contextlib import AbstractAsyncContextManager

from ..models import MemoryRecord, SearchFilters


class UnitOfWork(abc.ABC):
    """Opaque transaction handle passed to write methods. Adapters subclass as needed."""


class RecordStore(abc.ABC):
    """Durable records + lexical search. Owns the unit-of-work boundary."""

    @abc.abstractmethod
    def unit_of_work(self) -> AbstractAsyncContextManager[UnitOfWork]:
        """Async context manager; commits on clean exit, rolls back on exception."""

    @abc.abstractmethod
    async def insert_memory(
        self, uow: UnitOfWork, record: MemoryRecord
    ) -> tuple[MemoryRecord, bool]:
        """
        Insert (idempotently). Returns ``(stored, was_inserted)``.

        ``was_inserted`` is False when an ON-CONFLICT/dedup path returned an existing row —
        the orchestrator uses it to skip the vector write. This signal is load-bearing.
        """

    @abc.abstractmethod
    async def fetch_by_ids(self, user_id: str, ids: list[str]) -> list[MemoryRecord]:
        """Fetch records scoped to ``user_id`` (cross-user ids are silently dropped)."""

    @abc.abstractmethod
    async def lexical_search(
        self, user_id: str, query: str, filters: SearchFilters, limit: int
    ) -> list[tuple[str, float]]:
        """Return ``[(id, rank_score)]`` best-first for the lexical (BM25/FTS/ts_rank) leg."""

    @abc.abstractmethod
    async def soft_delete(self, user_id: str, memory_id: str) -> bool:
        """Authoritative delete (archive). Returns True if a row was affected."""

    async def update_access(self, user_id: str, ids: list[str]) -> None:
        """Optional: bump access_count/last_accessed_at. Default no-op."""
        return None
