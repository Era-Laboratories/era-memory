"""HTTP surface smoke test (in-process TestClient, Tier-0 in-memory). Needs [server]."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from era_memory.adapters.auth import BearerAuth  # noqa: E402
from era_memory.app import create_app  # noqa: E402
from era_memory.models import SearchRequest, SearchStrategy  # noqa: E402
from era_memory.wiring import build_memory  # noqa: E402


@pytest.fixture
def client():
    mem = build_memory(tier=0)
    mem.auth = BearerAuth("test-token", "u1")
    app = create_app(mem)
    with TestClient(app) as c:
        yield c


_AUTH = {"Authorization": "Bearer test-token"}


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


def test_delete_cannot_reach_another_owner_s_memory():
    # Two servers over one store, each token bound to a different principal.
    mem = build_memory(tier=0)
    mem.auth = BearerAuth("token-u1", "u1")
    with TestClient(create_app(mem)) as c1:
        r = c1.post(
            "/api/memories",
            json={"content": "dark roast coffee"},
            headers={"Authorization": "Bearer token-u1"},
        )
        memory_id = r.json()["id"]
    mem.auth = BearerAuth("token-u2", "u2")
    with TestClient(create_app(mem)) as c2:
        # Indistinguishable from a nonexistent id, so the route cannot be used to
        # probe which ids exist.
        r = c2.delete(f"/api/memories/{memory_id}", headers={"Authorization": "Bearer token-u2"})
        assert r.status_code == 404


def test_wrong_token_401(client):
    bad = {"Authorization": "Bearer not-the-token"}
    assert client.post("/api/memories", json={"content": "hi"}, headers=bad).status_code == 401
    assert client.post("/api/memories/search", json={"query": "hi"}, headers=bad).status_code == 401


def test_user_header_cannot_select_another_owner(client):
    # A valid token plus a forged X-User-Id still reads and writes as the configured
    # principal ("u1"): the header is ignored, not trusted.
    forged = {"Authorization": "Bearer test-token", "X-User-Id": "someone-else"}
    written = client.post("/api/memories", json={"content": "forged write"}, headers=forged)
    assert written.status_code == 200

    # The write landed under u1, so a plain u1 search finds it ...
    s = client.post("/api/memories/search", json={"query": "forged"}, headers=_AUTH)
    assert [r["id"] for r in s.json()["results"]] == [written.json()["id"]]

    # ... and a u1 memory is visible to the forged request too.
    own = _create(client, content="u1 private note")
    s = client.post("/api/memories/search", json={"query": "private"}, headers=forged)
    assert own in [r["id"] for r in s.json()["results"]]

    # Nothing was stored under the forged id.
    mem = client.app.state.memory
    other = client.portal.call(
        mem.search,
        SearchRequest(user_id="someone-else", query="forged", strategy=SearchStrategy.HYBRID),
    )
    assert other.results == []


def test_prebuilt_memory_with_noop_auth_refuses_to_start():
    app = create_app(build_memory(tier=0))  # wiring default = in-process NoopAuth
    with pytest.raises(RuntimeError, match="refuses to start"):
        with TestClient(app):
            pass


@pytest.mark.parametrize(
    "env",
    [
        {"MEMORY_PRINCIPAL": "u1"},
        {"MEMORY_BEARER_TOKEN": "", "MEMORY_PRINCIPAL": "u1"},
        {"MEMORY_BEARER_TOKEN": "test-token"},
        {"MEMORY_BEARER_TOKEN": "test-token", "MEMORY_PRINCIPAL": "  "},
    ],
    ids=["token-unset", "token-empty", "principal-unset", "principal-blank"],
)
def test_env_boot_refuses_without_token_and_principal(monkeypatch, env):
    for name in ("MEMORY_BEARER_TOKEN", "MEMORY_PRINCIPAL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMORY_TIER", "0")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="refuses to start"):
        with TestClient(create_app()):
            pass


def test_env_boot_binds_token_to_principal(monkeypatch):
    monkeypatch.setenv("MEMORY_TIER", "0")
    monkeypatch.delenv("MEMORY_DB_PATH", raising=False)
    monkeypatch.delenv("MEMORY_EMBEDDING_URL", raising=False)
    monkeypatch.setenv("MEMORY_BEARER_TOKEN", "env-token")
    monkeypatch.setenv("MEMORY_PRINCIPAL", "env-owner")
    with TestClient(create_app()) as c:
        headers = {"Authorization": "Bearer env-token", "X-User-Id": "someone-else"}
        assert c.post("/api/memories", json={"content": "x"}, headers=headers).status_code == 200
        assert c.app.state.memory.auth.__class__ is BearerAuth
        assert c.post("/api/memories", json={"content": "x"}).status_code == 401
