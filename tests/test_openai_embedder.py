"""OpenAI-compatible embedder: truncate+normalize logic, no network."""

from __future__ import annotations

import math


from era_memory.adapters.openai import OpenAICompatibleEmbedder


class _FakeEmbedder(OpenAICompatibleEmbedder):
    """Override the HTTP call so we can test the Matryoshka post-processing offline."""

    async def _embed_raw(self, texts):
        # Return raw 6-d vectors regardless of input; the adapter must truncate to dim.
        return [[float(i + 1) for i in range(6)] for _ in texts]


async def test_truncates_to_configured_dim_and_normalizes():
    emb = _FakeEmbedder(base_url="http://x/v1", model="m", dimensions=4)
    out = await emb.embed(["a", "b"])
    assert len(out) == 2
    assert all(len(v) == 4 for v in out)
    assert all(math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9) for v in out)


async def test_dimensions_and_model_id():
    emb = _FakeEmbedder(base_url="http://x/v1", model="bge-m3", dimensions=4)
    assert emb.dimensions == 4
    assert emb.model_id == "bge-m3"
