from __future__ import annotations

import abc

from ..models import SearchFilters, VectorRecord


class VectorStore(abc.ABC):
    """
    Vector index. At single-store tiers (0/1) it shares the RecordStore's backend, so the
    dual-write collapses into one transaction (set ``co_transactional = True`` and implement
    ``insert_in_uow``); at Tier 2 it is a separate system (Milvus) and ``insert`` can fail
    independently — preserving era-core's fail-fast-keep-record 503 semantics.
    """

    #: True when vectors are written inside the RecordStore unit-of-work (single-store
    #: tiers). The orchestrator reads this to decide where the vector write happens — it is
    #: the only topology branch in the whole system.
    co_transactional: bool = False

    @abc.abstractmethod
    async def insert(self, records: list[VectorRecord]) -> int:
        """Insert vectors (own transaction). Raise VectorStoreWriteError on failure."""

    async def insert_in_uow(self, uow, records: list[VectorRecord]) -> int:
        """
        Insert vectors inside an existing RecordStore unit-of-work (single-store collapse).
        Only called when ``co_transactional`` is True; default raises.
        """
        raise NotImplementedError("this VectorStore is not co-transactional")

    @abc.abstractmethod
    async def search(
        self,
        user_id: str,
        embedding: list[float],
        limit: int,
        filters: SearchFilters,
    ) -> list[tuple[str, float]]:
        """Return ``[(id, cosine_similarity)]`` best-first, scoped to ``user_id``."""

    @abc.abstractmethod
    async def delete(self, ids: list[str]) -> int:
        """Best-effort delete (callers may swallow failures). Return count deleted."""

    def is_connected(self) -> bool:
        """Readiness signal. Tier 2 reports (does not gate) on this. Default True."""
        return True
