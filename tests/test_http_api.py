"""HTTP surface smoke test (in-process TestClient, Tier-0 in-memory). Needs [server]."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from era_memory.adapters.auth import BearerAuth  # noqa: E402
from era_memory.app import create_app  # noqa: E402
from era_memory.wiring import build_memory  # noqa: E402


@pytest.fixture
def client():
    mem = build_memory(tier=0)
    mem.auth = BearerAuth("test-token")
    app = create_app(mem)
    with TestClient(app) as c:
        yield c


_AUTH = {"Authorization": "Bearer test-token", "X-User-Id": "u1"}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_unauthenticated_rejected(client):
    assert client.post("/api/memories", json={"content": "hi"}).status_code == 401


def test_create_and_search(client):
    r = client.post("/api/memories", json={"content": "dark roast coffee"}, headers=_AUTH)
    assert r.status_code == 200
    assert r.json()["source_type"] == "memory"

    s = client.post("/api/memories/search", json={"query": "coffee"}, headers=_AUTH)
    assert s.status_code == 200
    body = s.json()
    assert body["results"] and "coffee" in body["results"][0]["content"]
    assert body["strategy"] == "hybrid"


def test_empty_content_422(client):
    assert client.post("/api/memories", json={"content": ""}, headers=_AUTH).status_code == 422
