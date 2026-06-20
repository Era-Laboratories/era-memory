"""Embedder resolution precedence + the auto/setup plumbing (no network in the base tests)."""

from __future__ import annotations

import os

import pytest

from era_memory import build_memory
from era_memory.adapters.openai import OpenAICompatibleEmbedder
from era_memory.embedders import resolve_embedder
from era_memory.errors import ConfigurationError


def test_endpoint_env_wins(monkeypatch):
    monkeypatch.setenv("MEMORY_EMBEDDING_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("MEMORY_EMBEDDING_MODEL", "some-model")
    monkeypatch.setenv("MEMORY_EMBEDDING_DIMENSIONS", "256")
    emb = resolve_embedder(allow_download=False)
    assert isinstance(emb, OpenAICompatibleEmbedder)
    assert emb.dimensions == 256
    assert emb.model_id == "some-model"


def test_resolves_none_when_nothing_configured(monkeypatch):
    monkeypatch.delenv("MEMORY_EMBEDDING_URL", raising=False)
    # Point the cache somewhere empty so a real cached model can't leak in.
    monkeypatch.setenv("ERA_MEMORY_MODEL_DIR", "/tmp/era-memory-empty-cache-xyz")
    from era_memory.adapters.fastembed import fastembed_available

    resolved = resolve_embedder(allow_download=False)
    if fastembed_available():
        # Nothing cached and downloads disallowed -> still None.
        assert resolved is None
    else:
        assert resolved is None


def test_build_memory_falls_back_to_dev_embedder_with_warning(monkeypatch):
    monkeypatch.delenv("MEMORY_EMBEDDING_URL", raising=False)
    monkeypatch.setenv("ERA_MEMORY_MODEL_DIR", "/tmp/era-memory-empty-cache-xyz")
    from era_memory.adapters.fastembed import fastembed_available

    if fastembed_available():
        pytest.skip("fastembed present + possibly cached; covered by the gated test")
    with pytest.warns(UserWarning, match="non-semantic dev embedder"):
        mem = build_memory(tier=0)
    assert mem.embedder.model_id == "in-memory-hash"


def test_invalid_embedder_arg_rejected():
    with pytest.raises(ConfigurationError, match="must be an Embedder"):
        build_memory(tier=0, embedder="not-a-real-option")


@pytest.mark.skipif(
    os.environ.get("ERA_MEMORY_TEST_FASTEMBED") != "1",
    reason="set ERA_MEMORY_TEST_FASTEMBED=1 to run the network-backed fastembed test",
)
async def test_fastembed_embedder_semantic_signal():
    from era_memory.adapters.fastembed import SUPPORTED_MODELS, FastEmbedEmbedder, download_model
    from era_memory.core.logic.dedup import cosine_similarity

    spec = SUPPORTED_MODELS["bge-small"]
    download_model(spec)
    emb = FastEmbedEmbedder(spec)
    out = await emb.embed(["dark roast coffee", "espresso beans", "quantum physics lecture"])
    assert all(len(v) == emb.dimensions == spec.dimensions for v in out)
    assert cosine_similarity(out[0], out[1]) > cosine_similarity(out[0], out[2])
