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


def test_search_returns_experience_id_and_metadata(client):
    r = client.post(
        "/api/memories",
        json={
            "content": "dark roast coffee",
            "experience_id": "exp-123",
            "metadata": {"source_event": "evt-42", "initiative": "argus"},
        },
        headers=_AUTH,
    )
    assert r.status_code == 200

    s = client.post("/api/memories/search", json={"query": "coffee"}, headers=_AUTH)
    assert s.status_code == 200
    result = s.json()["results"][0]
    assert result["experience_id"] == "exp-123"
    assert result["metadata"] == {"source_event": "evt-42", "initiative": "argus"}


def test_empty_content_422(client):
    assert client.post("/api/memories", json={"content": ""}, headers=_AUTH).status_code == 422


def test_duplicate_create_returns_deduplicated_flag(client):
    body = {"content": "retry after timeout"}
    first = client.post("/api/memories", json=body, headers=_AUTH)
    assert first.status_code == 200
    assert "deduplicated" not in first.json()  # fresh insert: field absent

    second = client.post("/api/memories", json=body, headers=_AUTH)
    assert second.status_code == 200
    j = second.json()
    assert j["deduplicated"] is True
    assert j["id"] == first.json()["id"]  # same record returned, non-breaking shape
    assert j["source_type"] == "memory"


def _create(client, content="dark roast coffee"):
    r = client.post("/api/memories", json={"content": content}, headers=_AUTH)
    assert r.status_code == 200
    return r.json()["id"]


def test_delete_archives_by_default(client):
    memory_id = _create(client)
    r = client.delete(f"/api/memories/{memory_id}", headers=_AUTH)
    assert r.status_code == 200
    assert r.json() == {"id": memory_id, "deleted": True, "purged": False}
    # Archived: search no longer surfaces it.
    s = client.post("/api/memories/search", json={"query": "coffee"}, headers=_AUTH)
    assert s.json()["results"] == []


def test_delete_with_purge_erases_the_row(client):
    memory_id = _create(client)
    r = client.delete(f"/api/memories/{memory_id}?purge=true", headers=_AUTH)
    assert r.status_code == 200
    assert r.json()["purged"] is True
    # Gone, not archived — a second purge finds nothing.
    assert client.delete(f"/api/memories/{memory_id}?purge=true", headers=_AUTH).status_code == 404


def test_purge_reaches_an_already_archived_memory(client):
    # The Argus redaction path hits this: a memory soft-deleted earlier still
    # holds its content, and erasure has to be able to finish the job.
    memory_id = _create(client)
    assert client.delete(f"/api/memories/{memory_id}", headers=_AUTH).status_code == 200
    assert (
        client.delete(f"/api/memories/{memory_id}?purge=true", headers=_AUTH).status_code
        == 200
    )


def test_delete_unknown_id_is_404(client):
    assert client.delete("/api/memories/nope", headers=_AUTH).status_code == 404


def test_delete_requires_authentication(client):
    assert client.delete("/api/memories/anything").status_code == 401


def test_delete_cannot_reach_another_owner_s_memory(client):
    memory_id = _create(client)
    other = {"Authorization": "Bearer test-token", "X-User-Id": "u2"}
    # Indistinguishable from a nonexistent id, so the route cannot be used to
    # probe which ids exist.
    assert client.delete(f"/api/memories/{memory_id}", headers=other).status_code == 404
    assert client.get("/health").status_code == 200
