"""
Tier-1 Postgres + pgvector adapters: records + vectors in ONE database.

RecordStore (``memories`` + tsvector) and VectorStore (``memory_vectors`` with
``halfvec`` + HNSW cosine) share one asyncpg connection per unit-of-work, so the dual-write
collapses into a single transaction (``co_transactional = True``) — eliminating era-core's
Postgres-then-Milvus fail-fast 503 orphan path.

``halfvec(2048)`` is the locked production dimension (Qwen3-VL truncated); halfvec indexes
to 4000 dims, side-stepping pgvector's 2000-dim cap on the plain ``vector`` type. Embeddings
are bound as string literals cast to ``halfvec`` (no codec dependency). Lexical leg is native
``tsvector``/``ts_rank`` (no extension needed; RRF consumes rank, not raw score).

Needs the ``[postgres]`` extra (asyncpg). Requires the ``vector`` extension in the database.
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager

from ...errors import ConfigurationError, VectorStoreWriteError
from ...models import MemoryRecord, MemoryType, SearchFilters, VectorRecord
from ...ports import RecordStore, UnitOfWork, VectorStore

_TOKEN = re.compile(r"\w+")


def _require_asyncpg():
    try:
        import asyncpg
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Tier 1 Postgres needs the [postgres] extra: pip install era-memory[postgres]"
        ) from e
    return asyncpg


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


_SELECT_COLS = (
    "id, user_id, content, memory_type, status, importance_score, confidence, entities, "
    "topics, category, temporal_anchor, content_hash, session_id, source_memory_id, "
    "experience_id, metadata, created_at, updated_at, last_accessed_at, access_count"
)


def _record_from_row(r) -> MemoryRecord:
    return MemoryRecord(
        id=r["id"], user_id=r["user_id"], content=r["content"],
        memory_type=MemoryType(r["memory_type"]), status=r["status"],
        importance_score=r["importance_score"], confidence=r["confidence"],
        entities=json.loads(r["entities"]) if r["entities"] else [],
        topics=json.loads(r["topics"]) if r["topics"] else [],
        category=r["category"], temporal_anchor=r["temporal_anchor"],
        content_hash=r["content_hash"], session_id=r["session_id"],
        source_memory_id=r["source_memory_id"], experience_id=r["experience_id"],
        metadata=json.loads(r["metadata"]) if r["metadata"] else {},
        created_at=r["created_at"] or 0.0, updated_at=r["updated_at"] or 0.0,
        last_accessed_at=r["last_accessed_at"] or 0.0, access_count=r["access_count"] or 0,
    )


# ------------------------------------------------------------------------ shared backend


class PgBackend:
    def __init__(self, pool, dim: int, model_id: str) -> None:
        self.pool = pool
        self.dim = dim
        self.model_id = model_id

    async def close(self) -> None:
        await self.pool.close()

    @classmethod
    async def connect(cls, dsn: str, dim: int, model_id: str, *, reset: bool = False) -> "PgBackend":
        asyncpg = _require_asyncpg()
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=8)
        self = cls(pool, dim, model_id)
        await self._init_schema(reset=reset)
        return self

    async def _init_schema(self, *, reset: bool) -> None:
        async with self.pool.acquire() as c:
            await c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            if reset:
                await c.execute("DROP TABLE IF EXISTS memories, memory_vectors, meta CASCADE")
            await c.execute("CREATE TABLE IF NOT EXISTS meta (key text PRIMARY KEY, value text)")
            sig = json.dumps({"model": self.model_id, "dim": self.dim})
            row = await c.fetchrow("SELECT value FROM meta WHERE key='embedding'")
            if row is None:
                await c.execute("INSERT INTO meta(key,value) VALUES('embedding',$1)", sig)
            elif row["value"] != sig:
                raise ConfigurationError(
                    f"store was created with {row['value']} but embedder is {sig}; "
                    "single-tier-for-life — re-embed into a new database to change model/dim"
                )
            await c.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id text PRIMARY KEY, user_id text NOT NULL, content text NOT NULL,
                    content_tsv tsvector, memory_type text NOT NULL,
                    status text NOT NULL DEFAULT 'active',
                    importance_score double precision, confidence double precision,
                    entities jsonb, topics jsonb, category text, temporal_anchor text,
                    content_hash text, session_id text, source_memory_id text,
                    experience_id text, metadata jsonb,
                    created_at double precision, updated_at double precision,
                    last_accessed_at double precision, access_count integer DEFAULT 0
                )
                """
            )
            await c.execute(
                "CREATE INDEX IF NOT EXISTS ix_mem_user_hash ON memories(user_id, content_hash)"
            )
            await c.execute(
                "CREATE INDEX IF NOT EXISTS ix_mem_tsv ON memories USING gin (content_tsv)"
            )
            await c.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    id text PRIMARY KEY, user_id text NOT NULL,
                    embedding halfvec({self.dim}), memory_type text,
                    experience_id text, created_at bigint
                )
                """
            )
            await c.execute(
                "CREATE INDEX IF NOT EXISTS ix_vec_hnsw ON memory_vectors "
                "USING hnsw (embedding halfvec_cosine_ops)"
            )


# ------------------------------------------------------------------------ RecordStore


class _PgUoW(UnitOfWork):
    def __init__(self, conn) -> None:
        self.conn = conn


class PgRecordStore(RecordStore):
    def __init__(self, backend: PgBackend) -> None:
        self._b = backend

    @asynccontextmanager
    async def unit_of_work(self):
        async with self._b.pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                yield _PgUoW(conn)
                await tr.commit()
            except BaseException:
                await tr.rollback()
                raise

    async def insert_memory(self, uow, record: MemoryRecord) -> tuple[MemoryRecord, bool]:
        conn = uow.conn
        if record.content_hash:
            existing = await conn.fetchrow(
                f"SELECT {_SELECT_COLS} FROM memories "
                "WHERE user_id=$1 AND content_hash=$2 AND status='active'",
                record.user_id, record.content_hash,
            )
            if existing is not None:
                return _record_from_row(existing), False
        await conn.execute(
            """
            INSERT INTO memories (
                id,user_id,content,content_tsv,memory_type,status,importance_score,confidence,
                entities,topics,category,temporal_anchor,content_hash,session_id,
                source_memory_id,experience_id,metadata,created_at,updated_at,
                last_accessed_at,access_count
            ) VALUES (
                $1,$2,$3,to_tsvector('english',$3),$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10,$11,
                $12,$13,$14,$15,$16::jsonb,$17,$18,$19,$20
            )
            """,
            record.id, record.user_id, record.content, record.memory_type.value,
            record.status, record.importance_score, record.confidence,
            json.dumps(record.entities), json.dumps(record.topics), record.category,
            record.temporal_anchor, record.content_hash, record.session_id,
            record.source_memory_id, record.experience_id, json.dumps(record.metadata),
            record.created_at, record.updated_at, record.last_accessed_at, record.access_count,
        )
        return record, True

    async def fetch_by_ids(self, user_id, ids):
        if not ids:
            return []
        async with self._b.pool.acquire() as c:
            rows = await c.fetch(
                f"SELECT {_SELECT_COLS} FROM memories "
                "WHERE user_id=$1 AND status='active' AND id = ANY($2::text[])",
                user_id, list(ids),
            )
        return [_record_from_row(r) for r in rows]

    async def lexical_search(self, user_id, query, filters: SearchFilters, limit):
        if not _TOKEN.findall(query):
            return []
        clauses = ["m.user_id=$1", "m.status='active'", "m.content_tsv @@ q"]
        params: list = [user_id, query]
        n = 3
        if filters.memory_type is not None:
            clauses.append(f"m.memory_type=${n}")
            params.append(filters.memory_type.value)
            n += 1
        if filters.experience_id is not None:
            clauses.append(f"m.experience_id=${n}")
            params.append(filters.experience_id)
            n += 1
        params.append(limit)
        async with self._b.pool.acquire() as c:
            rows = await c.fetch(
                f"""
                SELECT m.id AS id, ts_rank(m.content_tsv, q) AS score
                FROM memories m, plainto_tsquery('english',$2) q
                WHERE {' AND '.join(clauses)}
                ORDER BY score DESC LIMIT ${n}
                """,
                *params,
            )
        return [(r["id"], r["score"]) for r in rows]

    async def soft_delete(self, user_id, memory_id):
        async with self._b.pool.acquire() as c:
            async with c.transaction():
                status = await c.execute(
                    "UPDATE memories SET status='archived' "
                    "WHERE user_id=$1 AND id=$2 AND status='active'",
                    user_id, memory_id,
                )
                affected = status.endswith("1")
                if affected:
                    await c.execute("DELETE FROM memory_vectors WHERE id=$1", memory_id)
                return affected


# ------------------------------------------------------------------------ VectorStore


class PgVectorStore(VectorStore):
    co_transactional = True

    def __init__(self, backend: PgBackend) -> None:
        self._b = backend

    async def _upsert(self, conn, records: list[VectorRecord]) -> int:
        for r in records:
            if not r.embedding:
                raise VectorStoreWriteError(f"empty embedding for {r.id}")
            if len(r.embedding) != self._b.dim:
                raise VectorStoreWriteError(
                    f"embedding dim {len(r.embedding)} != store dim {self._b.dim}"
                )
            await conn.execute(
                """
                INSERT INTO memory_vectors (id,user_id,embedding,memory_type,experience_id,created_at)
                VALUES ($1,$2,$3::halfvec,$4,$5,$6)
                ON CONFLICT (id) DO UPDATE SET embedding=EXCLUDED.embedding
                """,
                r.id, r.user_id, _vec_literal(r.embedding), r.memory_type.value,
                r.experience_id, r.created_at,
            )
        return len(records)

    async def insert(self, records):
        async with self._b.pool.acquire() as c:
            async with c.transaction():
                return await self._upsert(c, records)

    async def insert_in_uow(self, uow, records):
        return await self._upsert(uow.conn, records)

    async def search(self, user_id, embedding, limit, filters: SearchFilters):
        clauses = ["user_id=$2"]
        params: list = [_vec_literal(embedding), user_id]
        n = 3
        if filters.memory_type is not None:
            clauses.append(f"memory_type=${n}")
            params.append(filters.memory_type.value)
            n += 1
        if filters.experience_id is not None:
            clauses.append(f"experience_id=${n}")
            params.append(filters.experience_id)
            n += 1
        params.append(limit)
        async with self._b.pool.acquire() as c:
            rows = await c.fetch(
                f"""
                SELECT id, embedding <=> $1::halfvec AS distance
                FROM memory_vectors WHERE {' AND '.join(clauses)}
                ORDER BY distance LIMIT ${n}
                """,
                *params,
            )
        return [(r["id"], 1.0 - r["distance"]) for r in rows]

    async def delete(self, ids):
        if not ids:
            return 0
        async with self._b.pool.acquire() as c:
            status = await c.execute(
                "DELETE FROM memory_vectors WHERE id = ANY($1::text[])", list(ids)
            )
        return int(status.split()[-1]) if status.startswith("DELETE") else 0


async def open_pg_stores(
    dsn: str, dim: int, model_id: str, *, reset: bool = False
) -> tuple[PgRecordStore, PgVectorStore, PgBackend]:
    backend = await PgBackend.connect(dsn, dim, model_id, reset=reset)
    return PgRecordStore(backend), PgVectorStore(backend), backend
