from __future__ import annotations

import abc
from typing import Awaitable, Callable

from ..models import SessionPayload

# A consumer turns a session payload into persisted memories (the encoder pipeline).
Consumer = Callable[[SessionPayload], Awaitable[None]]


class Queue(abc.ABC):
    """
    Hands sessions to the encoder pipeline. Tier 0 = in-process (call now); Tier 1 =
    single Redis; Tier 2 = Redis Sentinel with consumer groups + DLQ.
    """

    @abc.abstractmethod
    async def submit(self, payload: SessionPayload) -> None:
        """Enqueue (or, in-process, run) one session through the consumer."""

    async def drain(self) -> None:
        """Optional: block until the backlog is processed (used by tests). Default no-op."""
        return None
