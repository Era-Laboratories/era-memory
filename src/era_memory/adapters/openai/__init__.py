"""
OpenAI-compatible embedder — the Tier-1/2 production embedding path.

Points at ANY OpenAI-compatible ``/embeddings`` endpoint (OpenAI, a self-hosted vLLM/GPU
service on era-labs-tools, LiteLLM, Ollama's shim). Matryoshka truncate + L2-normalize is
applied client-side so the stored dimension matches the index dimension exactly.

Needs the ``[openai]`` extra (httpx).
"""

from __future__ import annotations

from typing import Optional

from ...core.logic.matryoshka import truncate_and_normalize
from ...ports import Embedder


def _require_httpx():
    try:
        import httpx
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "The OpenAI-compatible embedder needs the [openai] extra: "
            "pip install era-memory[openai]"
        ) from e
    return httpx


class OpenAICompatibleEmbedder(Embedder):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimensions: int,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = dimensions
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

    @property
    def dimensions(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            httpx = _require_httpx()
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(timeout=self._timeout, headers=headers)
        return self._client

    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        """POST to the endpoint; return raw (pre-truncation) embeddings. Overridable in tests."""
        client = self._get_client()
        resp = await client.post(
            f"{self._base_url}/embeddings",
            json={"input": texts, "model": self._model},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in data]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raw = await self._embed_raw(texts)
        return [truncate_and_normalize(vec, self._dim) for vec in raw]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
