"""
Tier-0 SQLite adapters: records + vectors in ONE file.

RecordStore (sidecar tables + FTS5) and VectorStore (sqlite-vec ``vec0``) share a single
connection, so the dual-write collapses into one transaction (``co_transactional = True``)
and the 503 orphan-row failure mode is structurally impossible. Vectors are L2-normalized
upstream, so cosine-distance ordering is exact.

Needs the ``[sqlite]`` extra (``sqlite-vec``). FTS5 ships with stdlib sqlite3.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from contextlib import asynccontextmanager

from ...errors import ConfigurationError, VectorStoreWriteError
from ...models import MemoryRecord, MemoryType, SearchFilters, VectorRecord
from ...ports import RecordStore, UnitOfWork, VectorStore

_TOKEN = re.compile(r"\w+")


def _require_sqlite_vec():
    try:
        import sqlite_vec
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Tier 0 SQLite needs the [sqlite] extra: pip install era-memory[sqlite]"
        ) from e
    return sqlite_vec


def _fts_match_expr(query: str) -> str:
    """Build a safe FTS5 MATCH expression: quoted tokens OR-ed (any-term semantics)."""
    tokens = _TOKEN.findall(query.lower())
    return " OR ".join(f'"{t}"' for t in tokens)


# ------------------------------------------------------------------------ shared backend


class SqliteBackend:
    """Owns the single connection + lock + schema. Shared by the record and vector stores."""

    def __init__(self, path: str, dim: int, model_id: str) -> None:
        self._sqlite_vec = _require_sqlite_vec()
        self.dim = dim
        self.model_id = model_id
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.isolation_level = None  # manual BEGIN/COMMIT
        self.conn.enable_load_extension(True)
        self._sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.lock = asyncio.Lock()
        self._init_schema()

    def serialize(self, vec: list[float]) -> bytes:
        return self._sqlite_vec.serialize_float32(vec)

    def _init_schema(self) -> None:
        c = self.conn
        c.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        # (model, dim) lock-in guard — refuse to open a mismatched store.
        row = c.execute("SELECT value FROM meta WHERE key='embedding'").fetchone()
        sig = json.dumps({"model": self.model_id, "dim": self.dim})
        if row is None:
            c.execute("INSERT INTO meta(key,value) VALUES('embedding',?)", (sig,))
        elif row[0] != sig:
            raise ConfigurationError(
                f"store was created with {row[0]} but embedder is {sig}; "
                "single-tier-for-life — re-embed into a new file to change model/dim"
            )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL,
                memory_type TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
                importance_score REAL, confidence REAL, entities TEXT, topics TEXT,
                category TEXT, temporal_anchor TEXT, content_hash TEXT,
                session_id TEXT, source_memory_id TEXT, experience_id TEXT,
                metadata TEXT, created_at REAL, updated_at REAL,
                last_accessed_at REAL, access_count INTEGER DEFAULT 0
            )
            """
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS ix_mem_user_hash ON memories(user_id, content_hash)"
        )
        c.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(id UNINDEXED, content)"
        )
        c.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{self.dim}] distance_metric=cosine,
                user_id TEXT partition key,
                memory_type TEXT,
                experience_id TEXT,
                created_at INTEGER
            )
            """
        )


def _row_to_record(r: sqlite3.Row | tuple) -> MemoryRecord:
    (
        id_, user_id, content, memory_type, status, importance, confidence, entities,
        topics, category, temporal_anchor, content_hash, session_id, source_memory_id,
        experience_id, metadata, created_at, updated_at, last_accessed_at, access_count,
    ) = r
    return MemoryRecord(
        id=id_, user_id=user_id, content=content, memory_type=MemoryType(memory_type),
        status=status, importance_score=importance, confidence=confidence,
        entities=json.loads(entities) if entities else [],
        topics=json.loads(topics) if topics else [],
        category=category, temporal_anchor=temporal_anchor, content_hash=content_hash,
        session_id=session_id, source_memory_id=source_memory_id,
        experience_id=experience_id, metadata=json.loads(metadata) if metadata else {},
        created_at=created_at or 0.0, updated_at=updated_at or 0.0,
        last_accessed_at=last_accessed_at or 0.0, access_count=access_count or 0,
    )


_COLS = (
    "id,user_id,content,memory_type,status,importance_score,confidence,entities,topics,"
    "category,temporal_anchor,content_hash,session_id,source_memory_id,experience_id,"
    "metadata,created_at,updated_at,last_accessed_at,access_count"
)


# ------------------------------------------------------------------------ RecordStore


class _SqliteUoW(UnitOfWork):
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn


class SqliteRecordStore(RecordStore):
    def __init__(self, backend: SqliteBackend) -> None:
        self._b = backend

    @asynccontextmanager
    async def unit_of_work(self):
        async with self._b.lock:
            conn = self._b.conn
            conn.execute("BEGIN")
            try:
                yield _SqliteUoW(conn)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    async def insert_memory(self, uow, record: MemoryRecord) -> tuple[MemoryRecord, bool]:
        conn = self._b.conn
        if record.content_hash:
            existing = conn.execute(
                f"SELECT {_COLS} FROM memories WHERE user_id=? AND content_hash=? AND status='active'",
                (record.user_id, record.content_hash),
            ).fetchone()
            if existing is not None:
                return _row_to_record(existing), False
        conn.execute(
            f"INSERT INTO memories ({_COLS}) VALUES ({','.join('?' * 20)})",
            (
                record.id, record.user_id, record.content, record.memory_type.value,
                record.status, record.importance_score, record.confidence,
                json.dumps(record.entities), json.dumps(record.topics), record.category,
                record.temporal_anchor, record.content_hash, record.session_id,
                record.source_memory_id, record.experience_id, json.dumps(record.metadata),
                record.created_at, record.updated_at, record.last_accessed_at,
                record.access_count,
            ),
        )
        conn.execute(
            "INSERT INTO memories_fts (id, content) VALUES (?, ?)", (record.id, record.content)
        )
        return record, True

    async def fetch_by_ids(self, user_id: str, ids: list[str]) -> list[MemoryRecord]:
        if not ids:
            return []
        async with self._b.lock:
            placeholders = ",".join("?" * len(ids))
            rows = self._b.conn.execute(
                f"SELECT {_COLS} FROM memories "
                f"WHERE user_id=? AND status='active' AND id IN ({placeholders})",
                (user_id, *ids),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    async def lexical_search(self, user_id, query, filters: SearchFilters, limit):
        match = _fts_match_expr(query)
        if not match:
            return []
        clauses = ["m.user_id=?", "m.status='active'", "memories_fts MATCH ?"]
        params: list = [user_id, match]
        if filters.memory_type is not None:
            clauses.append("m.memory_type=?")
            params.append(filters.memory_type.value)
        if filters.experience_id is not None:
            clauses.append("m.experience_id=?")
            params.append(filters.experience_id)
        params.append(limit)
        async with self._b.lock:
            rows = self._b.conn.execute(
                f"""
                SELECT memories_fts.id, bm25(memories_fts) AS score
                FROM memories_fts JOIN memories m ON m.id = memories_fts.id
                WHERE {' AND '.join(clauses)}
                ORDER BY score LIMIT ?
                """,
                params,
            ).fetchall()
        # bm25 is lower-is-better; negate so higher-is-better (order already best-first).
        return [(i, -s) for i, s in rows]

    async def soft_delete(self, user_id: str, memory_id: str) -> bool:
        async with self._b.lock:
            conn = self._b.conn
            conn.execute("BEGIN")
            try:
                cur = conn.execute(
                    "UPDATE memories SET status='archived' WHERE user_id=? AND id=? AND status='active'",
                    (user_id, memory_id),
                )
                affected = cur.rowcount > 0
                if affected:
                    conn.execute("DELETE FROM memories_fts WHERE id=?", (memory_id,))
                    conn.execute("DELETE FROM vec_memories WHERE id=?", (memory_id,))
                conn.execute("COMMIT")
                return affected
            except BaseException:
                conn.execute("ROLLBACK")
                raise


# ------------------------------------------------------------------------ VectorStore


class SqliteVectorStore(VectorStore):
    co_transactional = True

    def __init__(self, backend: SqliteBackend) -> None:
        self._b = backend

    def _upsert(self, conn: sqlite3.Connection, records: list[VectorRecord]) -> int:
        for r in records:
            if not r.embedding:
                raise VectorStoreWriteError(f"empty embedding for {r.id}")
            if len(r.embedding) != self._b.dim:
                raise VectorStoreWriteError(
                    f"embedding dim {len(r.embedding)} != store dim {self._b.dim}"
                )
            conn.execute("DELETE FROM vec_memories WHERE id=?", (r.id,))
            conn.execute(
                "INSERT INTO vec_memories(id,embedding,user_id,memory_type,experience_id,created_at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    r.id, self._b.serialize(r.embedding), r.user_id,
                    r.memory_type.value, r.experience_id, r.created_at,
                ),
            )
        return len(records)

    async def insert(self, records: list[VectorRecord]) -> int:
        async with self._b.lock:
            conn = self._b.conn
            conn.execute("BEGIN")
            try:
                n = self._upsert(conn, records)
                conn.execute("COMMIT")
                return n
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    async def insert_in_uow(self, uow, records: list[VectorRecord]) -> int:
        # Lock already held by the RecordStore uow; write on the shared connection.
        return self._upsert(self._b.conn, records)

    async def search(self, user_id, embedding, limit, filters: SearchFilters):
        clauses = ["embedding MATCH ?", "k=?", "user_id=?"]
        params: list = [self._b.serialize(embedding), limit, user_id]
        if filters.memory_type is not None:
            clauses.append("memory_type=?")
            params.append(filters.memory_type.value)
        if filters.experience_id is not None:
            clauses.append("experience_id=?")
            params.append(filters.experience_id)
        async with self._b.lock:
            rows = self._b.conn.execute(
                f"SELECT id, distance FROM vec_memories WHERE {' AND '.join(clauses)} ORDER BY distance",
                params,
            ).fetchall()
        # cosine distance -> similarity (vectors are L2-normalized).
        return [(i, 1.0 - d) for i, d in rows]

    async def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        async with self._b.lock:
            conn = self._b.conn
            conn.execute("BEGIN")
            try:
                n = 0
                for i in ids:
                    cur = conn.execute("DELETE FROM vec_memories WHERE id=?", (i,))
                    n += cur.rowcount
                conn.execute("COMMIT")
                return n
            except BaseException:
                conn.execute("ROLLBACK")
                raise


def open_sqlite_stores(
    path: str, dim: int, model_id: str
) -> tuple[SqliteRecordStore, SqliteVectorStore]:
    """Build a shared-backend (record, vector) pair — the single-file collapse."""
    backend = SqliteBackend(path, dim, model_id)
    return SqliteRecordStore(backend), SqliteVectorStore(backend)
