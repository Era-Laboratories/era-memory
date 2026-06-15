"""
End-to-end acceptance for the wired Memory facade (Tier-0 logic on in-memory adapters).
The SQLite/sqlite-vec adapters (M1) will run this same scenario by swapping the fixture.
"""

from __future__ import annotations

from era_memory.models import MemoryRecord, SearchRequest, SearchStrategy, SessionPayload


async def test_store_and_search_relevance(memory):
    await memory.store(MemoryRecord(user_id="u1", content="Ada prefers dark roast coffee"))
    await memory.store(MemoryRecord(user_id="u1", content="Ada is allergic to peanuts"))
    await memory.store(MemoryRecord(user_id="u1", content="The weather today is sunny"))

    res = await memory.search(SearchRequest(user_id="u1", query="what coffee does Ada drink"))
    assert res.results
    assert "coffee" in res.results[0].content
    assert res.latency_ms >= 0.0


async def test_user_isolation(memory):
    await memory.store(MemoryRecord(user_id="alice", content="alice loves espresso"))
    await memory.store(MemoryRecord(user_id="bob", content="bob loves espresso"))
    res = await memory.search(SearchRequest(user_id="alice", query="espresso"))
    assert all(r.content.startswith("alice") for r in res.results)


async def test_vector_only_and_bm25_only(memory):
    await memory.store(MemoryRecord(user_id="u1", content="coffee espresso latte"))
    v = await memory.search(
        SearchRequest(user_id="u1", query="espresso", strategy=SearchStrategy.VECTOR_ONLY)
    )
    b = await memory.search(
        SearchRequest(user_id="u1", query="espresso", strategy=SearchStrategy.BM25_ONLY)
    )
    assert v.results and b.results


async def test_delete_removes_from_search(memory):
    stored = await memory.store(MemoryRecord(user_id="u1", content="temporary note about cats"))
    assert await memory.delete("u1", stored.id) is True
    res = await memory.search(SearchRequest(user_id="u1", query="cats"))
    assert all(r.id != stored.id for r in res.results)


async def test_encode_pipeline_creates_searchable_memories(memory):
    convo = (
        "User: I just moved to Berlin and started learning German.\n"
        "Assistant: That is exciting, how is the language going?\n"
        "User: My favorite food is currywurst from the corner stand.\n"
    )
    written = await memory.encode(
        SessionPayload(user_id="u1", session_id="s1", conversation=convo)
    )
    assert written  # heuristic extractor produced memories
    res = await memory.search(SearchRequest(user_id="u1", query="what food does the user like"))
    assert any("currywurst" in r.content for r in res.results)


async def test_encode_dedups_repeated_content(memory):
    line = "User: My dog's name is Rex and he is a golden retriever.\n"
    convo = line + line + line  # same memory three times
    written = await memory.encode(
        SessionPayload(user_id="u1", session_id="s2", conversation=convo)
    )
    # dedup @0.85 collapses the repeats to a single stored memory
    assert len(written) == 1


async def test_submit_session_via_in_process_queue(memory):
    convo = "User: I work as a marine biologist studying coral reefs.\n"
    await memory.submit_session(
        SessionPayload(user_id="u1", session_id="s3", conversation=convo)
    )
    res = await memory.search(SearchRequest(user_id="u1", query="what is the user's job"))
    assert any("biologist" in r.content for r in res.results)
