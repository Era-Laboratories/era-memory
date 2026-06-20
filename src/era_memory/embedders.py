"""
Embedder resolution: pick the best real embedder available, with no surprise downloads.

Precedence when no embedder is passed explicitly:

1. **Configured endpoint** — ``MEMORY_EMBEDDING_URL`` set → the OpenAI-compatible embedder.
2. **Cached local model** — a model already fetched by ``era-memory setup`` → fastembed.
3. **Opt-in download** — only when ``allow_download=True`` (``build_memory(embedder="auto")``
   or the ``setup`` CLI): fetch the default model from Hugging Face, then serve it.
4. **Nothing** — return ``None``; the caller falls back to the non-semantic dev embedder and
   warns. A bare ``build_memory()`` never touches the network or disk beyond what's cached.
"""

from __future__ import annotations

import os
from typing import Optional

from .config import Settings
from .ports import Embedder


def _openai_from_env() -> Optional[Embedder]:
    url = os.environ.get("MEMORY_EMBEDDING_URL")
    if not url:
        return None
    from .adapters.openai import OpenAICompatibleEmbedder

    return OpenAICompatibleEmbedder(
        base_url=url,
        model=os.environ.get("MEMORY_EMBEDDING_MODEL", "text-embedding-3-small"),
        dimensions=int(os.environ.get("MEMORY_EMBEDDING_DIMENSIONS", "1536")),
        api_key=os.environ.get("MEMORY_EMBEDDING_API_KEY"),
    )


def resolve_embedder(
    settings: Optional[Settings] = None,
    *,
    allow_download: bool = False,
    model_key: Optional[str] = None,
) -> Optional[Embedder]:
    """Return the best available real embedder, or ``None`` if only the dev stand-in is possible.

    Set ``allow_download=True`` to permit fetching the default (or ``model_key``) model from
    Hugging Face when nothing is configured or cached.
    """
    endpoint = _openai_from_env()
    if endpoint is not None:
        return endpoint

    from .adapters.fastembed import (
        DEFAULT_MODEL_KEY,
        SUPPORTED_MODELS,
        FastEmbedEmbedder,
        download_model,
        fastembed_available,
        first_cached_model,
    )

    # Cache detection is a file check only — it does NOT import fastembed — and
    # FastEmbedEmbedder loads the model lazily on first embed(). So resolving an
    # already-cached model keeps `build_memory()` backend-free until search actually runs.
    cached = first_cached_model()
    if cached is not None:
        return FastEmbedEmbedder(cached)

    # Downloading is the only path that requires fastembed up front.
    if allow_download and fastembed_available():
        spec = SUPPORTED_MODELS[model_key or DEFAULT_MODEL_KEY]
        download_model(spec)
        return FastEmbedEmbedder(spec)

    return None
