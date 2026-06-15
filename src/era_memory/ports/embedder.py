from __future__ import annotations

import abc


class Embedder(abc.ABC):
    """Text -> embedding. ``(model_id, dimensions)`` are recorded + guarded per store."""

    @property
    @abc.abstractmethod
    def dimensions(self) -> int: ...

    @property
    @abc.abstractmethod
    def model_id(self) -> str: ...

    @abc.abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text (already truncated + L2-normalized)."""
