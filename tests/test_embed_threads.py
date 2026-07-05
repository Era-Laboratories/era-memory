"""
The local (fastembed/ONNX) embedder must construct its model with a bounded thread pool,
so a single inference cannot starve FastAPI probe responses under a CPU quota (0.1.2).

No network / no real fastembed: a fake TextEmbedding captures the constructor kwargs.
"""

from __future__ import annotations

import era_memory.adapters.fastembed as fe
from era_memory.adapters.fastembed import (
    DEFAULT_EMBED_THREADS,
    SUPPORTED_MODELS,
    FastEmbedEmbedder,
)


class _FakeTextEmbedding:
    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeTextEmbedding.last_kwargs = kwargs

    def embed(self, texts):
        # 384-dim zero vectors matching bge-small; content is irrelevant to this test.
        for _ in texts:
            yield [0.0] * SUPPORTED_MODELS["bge-small"].dimensions


def _patch_fastembed(monkeypatch):
    _FakeTextEmbedding.last_kwargs = {}
    monkeypatch.setattr(fe, "_require_fastembed", lambda: _FakeTextEmbedding)


def test_default_threads_passed_to_constructor(monkeypatch):
    monkeypatch.delenv("MEMORY_EMBED_THREADS", raising=False)
    _patch_fastembed(monkeypatch)

    emb = FastEmbedEmbedder(SUPPORTED_MODELS["bge-small"])
    emb._get_model()  # triggers lazy construction

    assert _FakeTextEmbedding.last_kwargs.get("threads") == DEFAULT_EMBED_THREADS
    assert DEFAULT_EMBED_THREADS == 2


def test_env_overrides_thread_count(monkeypatch):
    monkeypatch.setenv("MEMORY_EMBED_THREADS", "1")
    _patch_fastembed(monkeypatch)

    emb = FastEmbedEmbedder(SUPPORTED_MODELS["bge-small"])
    emb._get_model()

    assert _FakeTextEmbedding.last_kwargs.get("threads") == 1


def test_explicit_threads_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv("MEMORY_EMBED_THREADS", "8")
    _patch_fastembed(monkeypatch)

    emb = FastEmbedEmbedder(SUPPORTED_MODELS["bge-small"], threads=3)
    emb._get_model()

    assert _FakeTextEmbedding.last_kwargs.get("threads") == 3


def test_download_model_warmup_bounds_threads(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_EMBED_THREADS", "1")
    _patch_fastembed(monkeypatch)

    fe.download_model(SUPPORTED_MODELS["bge-small"], cache_dir=tmp_path)

    assert _FakeTextEmbedding.last_kwargs.get("threads") == 1
