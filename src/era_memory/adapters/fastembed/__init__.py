"""
Offline local embedder backed by ``fastembed`` (ONNX, CPU-only).

Downloads a small embedding model straight from Hugging Face on demand, caches it under a
known directory, and serves embeddings in-process — no embedding endpoint, GPU, or API key.
This is the "real" embedder for the offline/laptop tier; the OpenAI-compatible embedder
remains the path for hosted endpoints.

Needs the ``[localembed]`` extra (``fastembed``). The download is **never** triggered by
import or by a bare ``build_memory()`` — only by ``era-memory setup`` / ``build_memory(
embedder="auto")`` (explicit opt-in). See ``era_memory.embedders`` for the resolver and
``docs/adr/0001`` for the dimension contract.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...core.logic.matryoshka import truncate_and_normalize
from ...ports import Embedder


@dataclass(frozen=True)
class ModelSpec:
    """A downloadable local model: short key, its HF/fastembed name, dim, and footprint."""

    key: str
    fastembed_name: str
    dimensions: int
    approx_size_mb: int
    license: str


# The curated set offered by `era-memory setup`. Both are fastembed-supported and
# redistributable; keep this list small and verified rather than exposing all of fastembed.
SUPPORTED_MODELS: dict[str, ModelSpec] = {
    "bge-small": ModelSpec(
        key="bge-small",
        fastembed_name="BAAI/bge-small-en-v1.5",
        dimensions=384,
        approx_size_mb=90,
        license="MIT",
    ),
    "mxbai-large": ModelSpec(
        key="mxbai-large",
        fastembed_name="mixedbread-ai/mxbai-embed-large-v1",
        dimensions=1024,
        approx_size_mb=640,
        license="Apache-2.0",
    ),
}

DEFAULT_MODEL_KEY = "bge-small"


def default_cache_dir() -> Path:
    """Where models + readiness sentinels live. Overridable via ERA_MEMORY_MODEL_DIR."""
    env = os.environ.get("ERA_MEMORY_MODEL_DIR")
    base = Path(env).expanduser() if env else Path.home() / ".cache" / "era-memory" / "models"
    return base


def _sentinel(cache_dir: Path, spec: ModelSpec) -> Path:
    return cache_dir / f"{spec.key}.ready"


def is_cached(spec: ModelSpec, cache_dir: Optional[Path] = None) -> bool:
    """True iff this model was already downloaded by us (readiness sentinel present).

    Cheap and side-effect-free — the resolver uses it to decide *without* hitting the network.
    """
    cache_dir = cache_dir or default_cache_dir()
    return _sentinel(cache_dir, spec).exists()


def first_cached_model(cache_dir: Optional[Path] = None) -> Optional[ModelSpec]:
    cache_dir = cache_dir or default_cache_dir()
    for spec in SUPPORTED_MODELS.values():
        if is_cached(spec, cache_dir):
            return spec
    return None


def fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


def _require_fastembed():
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise RuntimeError(
            "The local embedder needs the [localembed] extra: "
            "pip install 'era-memory[localembed]'"
        ) from e
    return TextEmbedding


def download_model(spec: ModelSpec, cache_dir: Optional[Path] = None) -> Path:
    """Fetch ``spec`` from Hugging Face into the cache and write a readiness sentinel.

    Materializes the model by running one embed so the ONNX weights are fully fetched, then
    records ``<key>.ready`` so :func:`is_cached` can detect it offline afterwards.
    """
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    TextEmbedding = _require_fastembed()
    model = TextEmbedding(model_name=spec.fastembed_name, cache_dir=str(cache_dir))
    # Force a real forward pass so the download is complete before we mark it ready.
    list(model.embed(["warmup"]))
    _sentinel(cache_dir, spec).write_text(
        json.dumps({"model": spec.fastembed_name, "dim": spec.dimensions})
    )
    return cache_dir


class FastEmbedEmbedder(Embedder):
    """Serves a cached fastembed model. Construction is cheap; the model loads on first embed."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        cache_dir: Optional[Path] = None,
        dimensions: Optional[int] = None,
    ) -> None:
        self._spec = spec
        self._cache_dir = cache_dir or default_cache_dir()
        if dimensions is not None and dimensions > spec.dimensions:
            raise ValueError(
                f"requested dim {dimensions} exceeds {spec.fastembed_name}'s native "
                f"{spec.dimensions} (Matryoshka truncates down, never up)"
            )
        self._dim = dimensions or spec.dimensions
        self._model = None

    @property
    def dimensions(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._spec.fastembed_name

    def _get_model(self):
        if self._model is None:
            TextEmbedding = _require_fastembed()
            self._model = TextEmbedding(
                model_name=self._spec.fastembed_name, cache_dir=str(self._cache_dir)
            )
        return self._model

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        return [truncate_and_normalize(vec.tolist(), self._dim) for vec in model.embed(texts)]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # fastembed is synchronous + CPU-bound; keep the event loop responsive.
        return await asyncio.to_thread(self._embed_sync, texts)
