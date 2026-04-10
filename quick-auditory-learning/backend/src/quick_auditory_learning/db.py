from __future__ import annotations

import json
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from decimal import Decimal
from pathlib import Path
from collections.abc import Iterable
from typing import Iterator, Mapping

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from pgvector import Vector

from quick_auditory_learning.models import EmbeddingModel, SessionCostItem
from quick_auditory_learning.settings import settings


COST_KIND_TABLES: dict[str, str] = {
    "search": "search_generation_costs",
    "embedding": "embedding_generation_costs",
    "explanation": "explanation_generation_costs",
    "audio": "audio_generation_costs",
    "keyword_generation": "keyword_generation_costs",
    "query_generation": "query_generation_costs",
    "prefetch": "prefetch_generation_costs",
}

COST_KIND_COLUMNS: dict[str, tuple[str, str]] = {
    kind: (f"{kind}_elapsed_ms", f"{kind}_cost_usd") for kind in COST_KIND_TABLES
}
VISIBLE_COST_KIND_TABLES: dict[str, str] = {kind: table for kind, table in COST_KIND_TABLES.items() if kind != "prefetch"}


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(settings.postgres_dsn, autocommit=True, row_factory=dict_row)
    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def normalize_identifier(value: str) -> str:
    cleaned = [ch if ch.isalnum() else "_" for ch in value.lower()]
    result = "".join(cleaned).strip("_")
    while "__" in result:
        result = result.replace("__", "_")
    return result or "model"


def model_table_name(model_name: str, model_version: str) -> str:
    digest = sha256(f"{model_name}:{model_version}".encode("utf-8")).hexdigest()[:12]
    return f"embedding_{normalize_identifier(model_name)}_{normalize_identifier(model_version)}_{digest}"


def generation_cost_rows(
    conn: psycopg.Connection,
    session_id: str,
    *,
    paper_id: str | None = None,
    kinds: Mapping[str, str] | None = None,
    include_prefetch: bool = True,
) -> list[Mapping[str, object]]:
    kinds = kinds or COST_KIND_TABLES
    if not kinds:
        return []
    query_parts: list[str] = []
    params: list[object] = []
    for kind, table in kinds.items():
        query_parts.append(
            f"SELECT '{kind}' AS kind, paper_id, created_at, started_at, finished_at, elapsed_ms, estimated_cost_usd, detail FROM {table} WHERE session_id = %s"
        )
        params.append(session_id)
        if paper_id is not None:
            query_parts[-1] += " AND paper_id = %s"
            params.append(paper_id)
    query = " UNION ALL ".join(query_parts) + " ORDER BY created_at ASC"
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
    if include_prefetch:
        return rows
    return [row for row in rows if _generation_scope_from_row(row) != "prefetch"]


def _row_elapsed_after_request_ms(row: Mapping[str, object], requested_at: datetime | None) -> int:
    elapsed_ms = _row_duration_ms(row)
    if elapsed_ms <= 0:
        return 0
    start_at, end_at = _row_time_bounds(row)
    if requested_at is None or not hasattr(requested_at, "timestamp") or start_at is None or end_at is None:
        return elapsed_ms
    if requested_at >= end_at:
        return 0
    return max(0, int((end_at - requested_at).total_seconds() * 1000))


def _row_cost_after_request_usd(row: Mapping[str, object], requested_at: datetime | None) -> Decimal:
    cost_usd = Decimal(str(row.get("estimated_cost_usd") or 0))
    start_at, end_at = _row_time_bounds(row)
    if start_at is None or end_at is None:
        return cost_usd
    if requested_at is None:
        return cost_usd
    if requested_at >= end_at:
        return Decimal("0")
    return cost_usd


def _row_time_bounds(row: Mapping[str, object]) -> tuple[datetime | None, datetime | None]:
    started_at = row.get("started_at")
    finished_at = row.get("finished_at")
    if hasattr(started_at, "timestamp") and hasattr(finished_at, "timestamp"):
        return started_at, finished_at
    created_at = row.get("created_at")
    elapsed_ms = int(row.get("elapsed_ms") or 0)
    if hasattr(started_at, "timestamp") and elapsed_ms > 0:
        return started_at, started_at + timedelta(milliseconds=elapsed_ms)
    if hasattr(finished_at, "timestamp") and elapsed_ms > 0:
        return finished_at - timedelta(milliseconds=elapsed_ms), finished_at
    if hasattr(created_at, "timestamp") and elapsed_ms > 0:
        return created_at, created_at + timedelta(milliseconds=elapsed_ms)
    if hasattr(started_at, "timestamp"):
        return started_at, started_at
    if hasattr(finished_at, "timestamp"):
        return finished_at, finished_at
    return None, None


def _row_duration_ms(row: Mapping[str, object]) -> int:
    start_at, end_at = _row_time_bounds(row)
    if start_at is not None and end_at is not None and hasattr(start_at, "timestamp") and hasattr(end_at, "timestamp"):
        return max(0, int((end_at - start_at).total_seconds() * 1000))
    return int(row.get("elapsed_ms") or 0)


def generation_cost_wall_elapsed_ms_from_rows(
    rows: Iterable[Mapping[str, object]],
    requested_at_by_paper_id: Mapping[str, datetime] | None = None,
) -> int:
    intervals: list[tuple[int, int]] = []
    fallback_total = 0
    for row in rows:
        elapsed_ms = _row_duration_ms(row)
        if elapsed_ms <= 0:
            continue
        start_at, end_at = _row_time_bounds(row)
        if start_at is None or end_at is None or not hasattr(start_at, "timestamp") or not hasattr(end_at, "timestamp"):
            fallback_total += elapsed_ms
            continue
        start_ms = int(start_at.timestamp() * 1000)
        end_ms = int(end_at.timestamp() * 1000)
        if requested_at_by_paper_id is not None:
            paper_id = str(row.get("paper_id") or "")
            requested_at = requested_at_by_paper_id.get(paper_id)
            if requested_at is not None and hasattr(requested_at, "timestamp"):
                requested_at_ms = int(requested_at.timestamp() * 1000)
                if requested_at_ms >= end_ms:
                    continue
                start_ms = requested_at_ms
        intervals.append((start_ms, end_ms))
    if not intervals:
        return fallback_total
    intervals.sort()
    total = 0
    current_start, current_end = intervals[0]
    for start_ms, end_ms in intervals[1:]:
        if start_ms <= current_end:
            current_end = max(current_end, end_ms)
            continue
        total += current_end - current_start
        current_start = start_ms
        current_end = end_ms
    total += current_end - current_start
    return total + fallback_total


def generation_cost_total_cost_usd_from_rows(
    rows: Iterable[Mapping[str, object]],
    requested_at_by_paper_id: Mapping[str, datetime] | None = None,
) -> float:
    total_cost_usd = Decimal("0")
    for row in rows:
        requested_at = None
        if requested_at_by_paper_id is not None:
            paper_id = str(row.get("paper_id") or "")
            requested_at = requested_at_by_paper_id.get(paper_id)
        total_cost_usd += _row_cost_after_request_usd(row, requested_at)
    return float(total_cost_usd)


def _generation_scope_from_row(row: Mapping[str, object]) -> str:
    detail = row.get("detail")
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except Exception:  # noqa: BLE001
            return ""
    if not isinstance(detail, Mapping):
        return ""
    return str(detail.get("generation_scope") or "")


def generation_cost_items_from_rows(
    rows: Iterable[Mapping[str, object]],
    kinds: Mapping[str, str] | None = None,
    requested_at_by_paper_id: Mapping[str, datetime] | None = None,
    *,
    missing_as_zero: bool = False,
) -> list[SessionCostItem]:
    kinds = kinds or VISIBLE_COST_KIND_TABLES
    grouped_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        kind = str(row.get("kind") or "")
        if kind:
            grouped_rows[kind].append(row)
    items: list[SessionCostItem] = []
    for kind in kinds:
        kind_rows = grouped_rows.get(kind, [])
        if not kind_rows:
            items.append(
                SessionCostItem(
                    kind=kind,
                    elapsed_ms=0 if missing_as_zero else None,
                    elapsed_ms_without_prefetch=0 if missing_as_zero else None,
                    estimated_cost_usd=0.0 if missing_as_zero else None,
                    status="calculated" if missing_as_zero else "pending",
                )
            )
            continue
        elapsed_ms = sum(_row_duration_ms(row) for row in kind_rows)
        elapsed_ms_without_prefetch = 0
        for row in kind_rows:
            paper_id = str(row.get("paper_id") or "")
            requested_at = None
            if requested_at_by_paper_id is not None:
                requested_at = requested_at_by_paper_id.get(paper_id)
            elapsed_ms_without_prefetch += _row_elapsed_after_request_ms(row, requested_at)
        cost_usd = sum(Decimal(str(row["estimated_cost_usd"])) for row in kind_rows)
        items.append(
            SessionCostItem(
                kind=kind,
                elapsed_ms=elapsed_ms,
                elapsed_ms_without_prefetch=elapsed_ms_without_prefetch,
                estimated_cost_usd=float(cost_usd),
                status="calculated",
            )
        )
    return items


def generation_cost_wall_elapsed_ms(
    conn: psycopg.Connection,
    session_id: str,
    *,
    paper_id: str | None = None,
    kinds: Mapping[str, str] | None = None,
    requested_at_by_paper_id: Mapping[str, datetime] | None = None,
    include_prefetch: bool = True,
) -> int:
    rows = generation_cost_rows(conn, session_id, paper_id=paper_id, kinds=kinds, include_prefetch=include_prefetch)
    return generation_cost_wall_elapsed_ms_from_rows(rows, requested_at_by_paper_id=requested_at_by_paper_id)


def generation_cost_total_cost_usd(
    conn: psycopg.Connection,
    session_id: str,
    *,
    paper_id: str | None = None,
    kinds: Mapping[str, str] | None = None,
    requested_at_by_paper_id: Mapping[str, datetime] | None = None,
    include_prefetch: bool = True,
) -> float:
    rows = generation_cost_rows(conn, session_id, paper_id=paper_id, kinds=kinds, include_prefetch=include_prefetch)
    return generation_cost_total_cost_usd_from_rows(rows, requested_at_by_paper_id=requested_at_by_paper_id)


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS arxiv_papers (
                id TEXT PRIMARY KEY,
                submitter TEXT,
                authors TEXT,
                title TEXT NOT NULL,
                comments TEXT,
                journal_ref TEXT,
                doi TEXT,
                abstract TEXT NOT NULL,
                report_no TEXT,
                categories JSONB NOT NULL DEFAULT '[]'::jsonb,
                versions JSONB NOT NULL DEFAULT '[]'::jsonb,
                raw JSONB NOT NULL DEFAULT '{}'::jsonb,
                search_text TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(title, '') || ' ' || coalesce(abstract, ''))
                ) STORED,
                imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS arxiv_papers_search_text_idx
            ON arxiv_papers
            USING GIN (search_text)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_models (
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                table_name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (model_name, model_version)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS favorites (
                paper_id TEXT PRIMARY KEY REFERENCES arxiv_papers(id) ON DELETE CASCADE,
                favorited_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS playback_transitions (
                id BIGSERIAL PRIMARY KEY,
                from_paper_id TEXT REFERENCES arxiv_papers(id) ON DELETE SET NULL,
                to_paper_id TEXT NOT NULL REFERENCES arxiv_papers(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS explanations (
                paper_id TEXT PRIMARY KEY REFERENCES arxiv_papers(id) ON DELETE CASCADE,
                model_name TEXT NOT NULL,
                explanation TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_memos (
                paper_id TEXT PRIMARY KEY REFERENCES arxiv_papers(id) ON DELETE CASCADE,
                memo TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS playback_sessions (
                session_id TEXT PRIMARY KEY,
                root_source_url TEXT NOT NULL,
                root_paper_id TEXT NOT NULL REFERENCES arxiv_papers(id) ON DELETE CASCADE,
                current_paper_id TEXT NOT NULL REFERENCES arxiv_papers(id) ON DELETE CASCADE,
                next_paper_id TEXT REFERENCES arxiv_papers(id) ON DELETE SET NULL,
                status TEXT NOT NULL,
                config JSONB NOT NULL,
                next_event_seq BIGINT NOT NULL DEFAULT 0,
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            ALTER TABLE playback_sessions
            ADD COLUMN IF NOT EXISTS next_paper_id TEXT REFERENCES arxiv_papers(id) ON DELETE SET NULL
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session_events (
                session_id TEXT NOT NULL REFERENCES playback_sessions(session_id) ON DELETE CASCADE,
                seq BIGINT NOT NULL,
                event_type TEXT NOT NULL,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (session_id, seq)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session_trail_items (
                session_id TEXT NOT NULL REFERENCES playback_sessions(session_id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                paper_id TEXT NOT NULL REFERENCES arxiv_papers(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (session_id, position),
                UNIQUE (session_id, paper_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session_queue_items (
                session_id TEXT NOT NULL REFERENCES playback_sessions(session_id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                paper_id TEXT NOT NULL REFERENCES arxiv_papers(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (session_id, position),
                UNIQUE (session_id, paper_id)
            )
            """
        )
        for table_name in COST_KIND_TABLES.values():
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT REFERENCES playback_sessions(session_id) ON DELETE CASCADE,
                    paper_id TEXT REFERENCES arxiv_papers(id) ON DELETE CASCADE,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    finished_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    elapsed_ms INTEGER NOT NULL,
                    estimated_cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
                    detail JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                f"""
                ALTER TABLE {table_name}
                ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ
                """
            )
            cursor.execute(
                f"""
                ALTER TABLE {table_name}
                ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ
                """
            )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session_generation_cost_totals (
                session_id TEXT PRIMARY KEY REFERENCES playback_sessions(session_id) ON DELETE CASCADE,
                search_elapsed_ms BIGINT NOT NULL DEFAULT 0,
                search_cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
                embedding_elapsed_ms BIGINT NOT NULL DEFAULT 0,
                embedding_cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
                explanation_elapsed_ms BIGINT NOT NULL DEFAULT 0,
                explanation_cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
                audio_elapsed_ms BIGINT NOT NULL DEFAULT 0,
                audio_cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
                keyword_generation_elapsed_ms BIGINT NOT NULL DEFAULT 0,
                keyword_generation_cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
                query_generation_elapsed_ms BIGINT NOT NULL DEFAULT 0,
                query_generation_cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
                prefetch_elapsed_ms BIGINT NOT NULL DEFAULT 0,
                prefetch_cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        for column_name in (
            "search_elapsed_ms",
            "search_cost_usd",
            "embedding_elapsed_ms",
            "embedding_cost_usd",
            "explanation_elapsed_ms",
            "explanation_cost_usd",
            "audio_elapsed_ms",
            "audio_cost_usd",
            "keyword_generation_elapsed_ms",
            "keyword_generation_cost_usd",
            "query_generation_elapsed_ms",
            "query_generation_cost_usd",
            "prefetch_elapsed_ms",
            "prefetch_cost_usd",
        ):
            column_type = "BIGINT NOT NULL DEFAULT 0" if column_name.endswith("_elapsed_ms") else "NUMERIC(12, 6) NOT NULL DEFAULT 0"
            cursor.execute(
                f"""
                ALTER TABLE session_generation_cost_totals
                ADD COLUMN IF NOT EXISTS {column_name} {column_type}
                """
            )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jsonl_import_sources (
                source_path TEXT PRIMARY KEY,
                source_mtime_ns BIGINT NOT NULL,
                source_size BIGINT NOT NULL,
                imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )


def ensure_embedding_table(conn: psycopg.Connection, model_name: str, model_version: str, dimension: int) -> str:
    table_name = model_table_name(model_name, model_version)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO embedding_models (model_name, model_version, dimension, table_name)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (model_name, model_version) DO UPDATE
            SET dimension = EXCLUDED.dimension,
                table_name = EXCLUDED.table_name
            """,
            (model_name, model_version, dimension, table_name),
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                paper_id TEXT PRIMARY KEY REFERENCES arxiv_papers(id) ON DELETE CASCADE,
                embedding vector({dimension}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx ON {table_name} USING hnsw (embedding vector_cosine_ops)"
        )
    return table_name


def get_embedding_model(
    conn: psycopg.Connection,
    model_name: str,
    model_version: str | None,
) -> EmbeddingModel | None:
    with conn.cursor() as cursor:
        if model_version is None:
            cursor.execute(
                """
                SELECT model_name, model_version, dimension, table_name
                FROM embedding_models
                WHERE model_name = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (model_name,),
            )
        else:
            cursor.execute(
                """
                SELECT model_name, model_version, dimension, table_name
                FROM embedding_models
                WHERE model_name = %s AND model_version = %s
                """,
                (model_name, model_version),
            )
        row = cursor.fetchone()
        if row is None:
            return None
        return EmbeddingModel(**row)


def list_embedding_models(conn: psycopg.Connection, model_name: str) -> list[EmbeddingModel]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT model_name, model_version, dimension, table_name
            FROM embedding_models
            WHERE model_name = %s
            ORDER BY created_at DESC
            """,
            (model_name,),
        )
        return [EmbeddingModel(**row) for row in cursor.fetchall()]


def upsert_paper(conn: psycopg.Connection, paper: dict[str, object]) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO arxiv_papers (
                id, submitter, authors, title, comments, journal_ref, doi, abstract, report_no,
                categories, versions, raw, updated_at
            ) VALUES (
                %(id)s, %(submitter)s, %(authors)s, %(title)s, %(comments)s, %(journal_ref)s, %(doi)s, %(abstract)s,
                %(report_no)s, %(categories)s::jsonb, %(versions)s::jsonb, %(raw)s::jsonb, NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                submitter = EXCLUDED.submitter,
                authors = EXCLUDED.authors,
                title = EXCLUDED.title,
                comments = EXCLUDED.comments,
                journal_ref = EXCLUDED.journal_ref,
                doi = EXCLUDED.doi,
                abstract = EXCLUDED.abstract,
                report_no = EXCLUDED.report_no,
                categories = EXCLUDED.categories,
                versions = EXCLUDED.versions,
                raw = EXCLUDED.raw,
                updated_at = NOW()
            """,
            paper,
        )


def list_papers_without_embeddings(conn: psycopg.Connection, table_name: str) -> list[str]:
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT id
            FROM arxiv_papers
            WHERE id NOT IN (SELECT paper_id FROM {table_name})
            ORDER BY id ASC
            """
        )
        return [row["id"] for row in cursor.fetchall()]


def count_papers(conn: psycopg.Connection) -> int:
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS count FROM arxiv_papers")
        row = cursor.fetchone()
    if row is None:
        return 0
    return int(row["count"])


def store_embedding(conn: psycopg.Connection, table_name: str, paper_id: str, embedding: list[float]) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO {table_name} (paper_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (paper_id) DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            (paper_id, Vector(embedding)),
        )


def get_explanation(conn: psycopg.Connection, paper_id: str) -> str | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT explanation
            FROM explanations
            WHERE paper_id = %s
            """,
            (paper_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return row["explanation"]


def upsert_explanation(conn: psycopg.Connection, paper_id: str, model_name: str, explanation: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO explanations (paper_id, model_name, explanation, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (paper_id) DO UPDATE SET
                model_name = EXCLUDED.model_name,
                explanation = EXCLUDED.explanation,
                updated_at = NOW()
            """,
            (paper_id, model_name, explanation),
        )


@dataclass(frozen=True)
class ImportResult:
    imported: int
    updated: int


@dataclass(frozen=True)
class JsonlImportState:
    source_path: str
    source_mtime_ns: int
    source_size: int


@dataclass(frozen=True)
class PlaybackSession:
    session_id: str
    root_source_url: str
    root_paper_id: str
    current_paper_id: str
    next_paper_id: str | None
    status: str
    config: dict[str, object]
    next_event_seq: int
    started_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class SessionEvent:
    session_id: str
    seq: int
    event_type: str
    payload: dict[str, object]
    created_at: datetime


@dataclass(frozen=True)
class PlaybackSessionSummary:
    session_id: str
    status: str
    root_source_url: str
    root_paper_id: str
    root_paper_title: str | None
    current_paper_id: str
    current_paper_title: str | None
    config: dict[str, object]
    next_event_seq: int
    started_at: datetime
    updated_at: datetime
    total_generation_elapsed_ms: int = 0
    total_wall_elapsed_ms: int = 0
    total_generation_cost_usd: float = 0.0


@dataclass(frozen=True)
class SessionGenerationCostSummary:
    session_id: str
    search_elapsed_ms: int
    search_cost_usd: float
    embedding_elapsed_ms: int
    embedding_cost_usd: float
    explanation_elapsed_ms: int
    explanation_cost_usd: float
    audio_elapsed_ms: int
    audio_cost_usd: float
    keyword_generation_elapsed_ms: int
    keyword_generation_cost_usd: float
    query_generation_elapsed_ms: int
    query_generation_cost_usd: float
    prefetch_elapsed_ms: int
    prefetch_cost_usd: float
    total_elapsed_ms: int
    total_wall_elapsed_ms: int
    total_cost_usd: float
    updated_at: datetime


def jsonl_import_state(conn: psycopg.Connection, source_path: str) -> JsonlImportState | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT source_path, source_mtime_ns, source_size
            FROM jsonl_import_sources
            WHERE source_path = %s
            """,
            (source_path,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return JsonlImportState(
        source_path=row["source_path"],
        source_mtime_ns=row["source_mtime_ns"],
        source_size=row["source_size"],
    )


def upsert_jsonl_import_state(
    conn: psycopg.Connection,
    source_path: str,
    source_mtime_ns: int,
    source_size: int,
) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO jsonl_import_sources (source_path, source_mtime_ns, source_size)
            VALUES (%s, %s, %s)
            ON CONFLICT (source_path) DO UPDATE SET
                source_mtime_ns = EXCLUDED.source_mtime_ns,
                source_size = EXCLUDED.source_size,
                imported_at = NOW()
            """,
            (source_path, source_mtime_ns, source_size),
        )


def create_playback_session(
    conn: psycopg.Connection,
    session_id: str,
    root_source_url: str,
    root_paper_id: str,
    current_paper_id: str,
    next_paper_id: str | None,
    config: dict[str, object],
) -> PlaybackSession:
    config_json = json.dumps(config, ensure_ascii=False)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO playback_sessions (
                session_id, root_source_url, root_paper_id, current_paper_id, next_paper_id, status, config, next_event_seq
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, 0)
            ON CONFLICT (session_id) DO UPDATE SET
                root_source_url = EXCLUDED.root_source_url,
                root_paper_id = EXCLUDED.root_paper_id,
                current_paper_id = EXCLUDED.current_paper_id,
                next_paper_id = EXCLUDED.next_paper_id,
                status = EXCLUDED.status,
                config = EXCLUDED.config,
                next_event_seq = EXCLUDED.next_event_seq,
                updated_at = NOW()
            """,
            (session_id, root_source_url, root_paper_id, current_paper_id, next_paper_id, "active", config_json),
        )
    session = get_playback_session(conn, session_id)
    if session is None:
        raise ValueError(f"failed to create playback session: {session_id}")
    return session


def get_playback_session(conn: psycopg.Connection, session_id: str) -> PlaybackSession | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT session_id, root_source_url, root_paper_id, current_paper_id, next_paper_id, status, config,
                   next_event_seq, started_at, updated_at
            FROM playback_sessions
            WHERE session_id = %s
            """,
            (session_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    config_value = row["config"]
    if isinstance(config_value, str):
        config = json.loads(config_value)
    else:
        config = dict(config_value)
    return PlaybackSession(
        session_id=row["session_id"],
        root_source_url=row["root_source_url"],
        root_paper_id=row["root_paper_id"],
        current_paper_id=row["current_paper_id"],
        next_paper_id=row["next_paper_id"],
        status=row["status"],
        config=config,
        next_event_seq=int(row["next_event_seq"]),
        started_at=row["started_at"],
        updated_at=row["updated_at"],
    )


def list_playback_sessions(conn: psycopg.Connection, limit: int = 20) -> list[PlaybackSessionSummary]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                ps.session_id,
                ps.status,
                ps.root_source_url,
                ps.root_paper_id,
                root_paper.title AS root_paper_title,
                ps.current_paper_id,
                current_paper.title AS current_paper_title,
                ps.config,
                ps.next_event_seq,
                ps.started_at,
                ps.updated_at,
                COALESCE(costs.search_elapsed_ms, 0) AS search_elapsed_ms,
                COALESCE(costs.search_cost_usd, 0) AS search_cost_usd,
                COALESCE(costs.embedding_elapsed_ms, 0) AS embedding_elapsed_ms,
                COALESCE(costs.embedding_cost_usd, 0) AS embedding_cost_usd,
                COALESCE(costs.explanation_elapsed_ms, 0) AS explanation_elapsed_ms,
                COALESCE(costs.explanation_cost_usd, 0) AS explanation_cost_usd,
                COALESCE(costs.audio_elapsed_ms, 0) AS audio_elapsed_ms,
                COALESCE(costs.audio_cost_usd, 0) AS audio_cost_usd,
                COALESCE(costs.keyword_generation_elapsed_ms, 0) AS keyword_generation_elapsed_ms,
                COALESCE(costs.keyword_generation_cost_usd, 0) AS keyword_generation_cost_usd,
                COALESCE(costs.query_generation_elapsed_ms, 0) AS query_generation_elapsed_ms,
                COALESCE(costs.query_generation_cost_usd, 0) AS query_generation_cost_usd,
                COALESCE(costs.prefetch_elapsed_ms, 0) AS prefetch_elapsed_ms,
                COALESCE(costs.prefetch_cost_usd, 0) AS prefetch_cost_usd,
                COALESCE(costs.search_elapsed_ms, 0)
                  + COALESCE(costs.embedding_elapsed_ms, 0)
                  + COALESCE(costs.explanation_elapsed_ms, 0)
                  + COALESCE(costs.audio_elapsed_ms, 0)
                  + COALESCE(costs.keyword_generation_elapsed_ms, 0)
                  + COALESCE(costs.query_generation_elapsed_ms, 0) AS total_generation_elapsed_ms,
                CASE
                    WHEN ps.status = 'active' THEN EXTRACT(EPOCH FROM (NOW() - ps.started_at)) * 1000
                    ELSE EXTRACT(EPOCH FROM (ps.updated_at - ps.started_at)) * 1000
                END AS total_wall_elapsed_ms,
                COALESCE(costs.search_cost_usd, 0)
                  + COALESCE(costs.embedding_cost_usd, 0)
                  + COALESCE(costs.explanation_cost_usd, 0)
                  + COALESCE(costs.audio_cost_usd, 0)
                  + COALESCE(costs.keyword_generation_cost_usd, 0)
                  + COALESCE(costs.query_generation_cost_usd, 0) AS total_generation_cost_usd
            FROM playback_sessions AS ps
            LEFT JOIN arxiv_papers AS root_paper
                ON root_paper.id = ps.root_paper_id
            LEFT JOIN arxiv_papers AS current_paper
                ON current_paper.id = ps.current_paper_id
            LEFT JOIN session_generation_cost_totals AS costs
                ON costs.session_id = ps.session_id
            ORDER BY ps.updated_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall()
    sessions: list[PlaybackSessionSummary] = []
    for row in rows:
        total_generation_elapsed_ms = generation_cost_wall_elapsed_ms(conn, row["session_id"])
        total_generation_cost_usd = generation_cost_total_cost_usd(conn, row["session_id"])
        config_value = row["config"]
        if isinstance(config_value, str):
            config = json.loads(config_value)
        else:
            config = dict(config_value)
        sessions.append(
            PlaybackSessionSummary(
                session_id=row["session_id"],
                status=row["status"],
                root_source_url=row["root_source_url"],
                root_paper_id=row["root_paper_id"],
                root_paper_title=row["root_paper_title"],
                current_paper_id=row["current_paper_id"],
                current_paper_title=row["current_paper_title"],
                config=config,
                next_event_seq=int(row["next_event_seq"]),
                started_at=row["started_at"],
                updated_at=row["updated_at"],
                total_generation_elapsed_ms=total_generation_elapsed_ms,
                total_wall_elapsed_ms=int(row["total_wall_elapsed_ms"]),
                total_generation_cost_usd=total_generation_cost_usd,
            )
        )
    return sessions


def _generation_cost_table_name(kind: str) -> str:
    table_name = COST_KIND_TABLES.get(kind)
    if table_name is None:
        raise ValueError(f"unknown generation cost kind: {kind}")
    return table_name


def record_generation_cost(
    conn: psycopg.Connection,
    kind: str,
    *,
    session_id: str | None,
    paper_id: str | None,
    started_at: datetime,
    finished_at: datetime,
    elapsed_ms: int,
    estimated_cost_usd: float | Decimal,
    detail: dict[str, object] | None = None,
) -> None:
    table_name = _generation_cost_table_name(kind)
    detail_json = json.dumps(detail or {}, ensure_ascii=False)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO {table_name} (session_id, paper_id, started_at, finished_at, elapsed_ms, estimated_cost_usd, detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                session_id,
                paper_id,
                started_at,
                finished_at,
                elapsed_ms,
                estimated_cost_usd if isinstance(estimated_cost_usd, Decimal) else Decimal(str(estimated_cost_usd)),
                detail_json,
            ),
        )
        if session_id is None:
            return
        elapsed_column, cost_column = COST_KIND_COLUMNS[kind]
        totals = {
            column: 0 for column in (f"{name}_elapsed_ms" for name in COST_KIND_TABLES)
        }
        totals.update({column: Decimal("0") for column in (f"{name}_cost_usd" for name in COST_KIND_TABLES)})
        totals[elapsed_column] = elapsed_ms
        totals[cost_column] = estimated_cost_usd if isinstance(estimated_cost_usd, Decimal) else Decimal(str(estimated_cost_usd))
        columns = ["session_id", *totals.keys()]
        values = [session_id, *totals.values()]
        set_clause = ", ".join(f"{column} = session_generation_cost_totals.{column} + EXCLUDED.{column}" for column in totals)
        cursor.execute(
            f"""
            INSERT INTO session_generation_cost_totals ({", ".join(columns)}, updated_at)
            VALUES ({", ".join(["%s"] * len(columns))}, NOW())
            ON CONFLICT (session_id) DO UPDATE SET
                {set_clause},
                updated_at = NOW()
            """,
            values,
        )


def get_paper_generation_costs(conn: psycopg.Connection, session_id: str, paper_id: str) -> list[tuple[str, int, float]]:
    """Return [(kind, elapsed_ms, cost_usd)] for the given session+paper, excluding zero rows."""
    union_parts = " UNION ALL ".join(
        f"SELECT '{kind}' AS kind, COALESCE(SUM(elapsed_ms), 0) AS elapsed_ms, COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd FROM {table} WHERE session_id = %s AND paper_id = %s"
        for kind, table in VISIBLE_COST_KIND_TABLES.items()
    )
    params: list[object] = []
    for _ in VISIBLE_COST_KIND_TABLES:
        params.extend([session_id, paper_id])
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT kind, elapsed_ms, cost_usd FROM ({union_parts}) t WHERE elapsed_ms > 0 OR cost_usd > 0",
            params,
        )
        rows = cursor.fetchall()
    return [(row["kind"], int(row["elapsed_ms"]), float(row["cost_usd"])) for row in rows]


def get_session_generation_costs(conn: psycopg.Connection, session_id: str) -> SessionGenerationCostSummary | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT costs.session_id,
                   costs.search_elapsed_ms, costs.search_cost_usd,
                   costs.embedding_elapsed_ms, costs.embedding_cost_usd,
                   costs.explanation_elapsed_ms, costs.explanation_cost_usd,
                   costs.audio_elapsed_ms, costs.audio_cost_usd,
                   costs.keyword_generation_elapsed_ms, costs.keyword_generation_cost_usd,
                   costs.query_generation_elapsed_ms, costs.query_generation_cost_usd,
                   costs.prefetch_elapsed_ms, costs.prefetch_cost_usd,
                   costs.search_elapsed_ms + costs.embedding_elapsed_ms + costs.explanation_elapsed_ms + costs.audio_elapsed_ms + costs.keyword_generation_elapsed_ms + costs.query_generation_elapsed_ms AS total_elapsed_ms,
                   CASE
                       WHEN ps.status = 'active' THEN EXTRACT(EPOCH FROM (NOW() - ps.started_at)) * 1000
                       ELSE EXTRACT(EPOCH FROM (ps.updated_at - ps.started_at)) * 1000
                   END AS total_wall_elapsed_ms,
                   costs.search_cost_usd + costs.embedding_cost_usd + costs.explanation_cost_usd + costs.audio_cost_usd + costs.keyword_generation_cost_usd + costs.query_generation_cost_usd AS total_cost_usd,
                   costs.updated_at
            FROM session_generation_cost_totals AS costs
            JOIN playback_sessions AS ps
              ON ps.session_id = costs.session_id
            WHERE costs.session_id = %s
            """,
            (session_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    total_elapsed_ms = generation_cost_wall_elapsed_ms(conn, session_id)
    total_cost_usd = generation_cost_total_cost_usd(conn, session_id)
    return SessionGenerationCostSummary(
        session_id=row["session_id"],
        search_elapsed_ms=int(row["search_elapsed_ms"]),
        search_cost_usd=float(row["search_cost_usd"]),
        embedding_elapsed_ms=int(row["embedding_elapsed_ms"]),
        embedding_cost_usd=float(row["embedding_cost_usd"]),
        explanation_elapsed_ms=int(row["explanation_elapsed_ms"]),
        explanation_cost_usd=float(row["explanation_cost_usd"]),
        audio_elapsed_ms=int(row["audio_elapsed_ms"]),
        audio_cost_usd=float(row["audio_cost_usd"]),
        keyword_generation_elapsed_ms=int(row["keyword_generation_elapsed_ms"]),
        keyword_generation_cost_usd=float(row["keyword_generation_cost_usd"]),
        query_generation_elapsed_ms=int(row["query_generation_elapsed_ms"]),
        query_generation_cost_usd=float(row["query_generation_cost_usd"]),
        prefetch_elapsed_ms=int(row["prefetch_elapsed_ms"]),
        prefetch_cost_usd=float(row["prefetch_cost_usd"]),
        total_elapsed_ms=total_elapsed_ms,
        total_wall_elapsed_ms=int(row["total_wall_elapsed_ms"]),
        total_cost_usd=total_cost_usd,
        updated_at=row["updated_at"],
    )


def update_playback_session(
    conn: psycopg.Connection,
    session_id: str,
    *,
    current_paper_id: str | None = None,
    next_paper_id: str | None = None,
    status: str | None = None,
) -> None:
    updates: list[str] = []
    values: list[object] = []
    if current_paper_id is not None:
        updates.append("current_paper_id = %s")
        values.append(current_paper_id)
    if next_paper_id is not None:
        updates.append("next_paper_id = %s")
        values.append(next_paper_id)
    if status is not None:
        updates.append("status = %s")
        values.append(status)
    if not updates:
        return
    updates.append("updated_at = NOW()")
    values.append(session_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"UPDATE playback_sessions SET {', '.join(updates)} WHERE session_id = %s",
            values,
        )


def append_session_event(
    conn: psycopg.Connection,
    session_id: str,
    event_type: str,
    payload: dict[str, object],
) -> int:
    payload_json = json.dumps(payload, ensure_ascii=False)
    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT next_event_seq
                FROM playback_sessions
                WHERE session_id = %s
                FOR UPDATE
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"playback session not found: {session_id}")
            seq = int(row["next_event_seq"]) + 1
            cursor.execute(
                """
                INSERT INTO session_events (session_id, seq, event_type, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (session_id, seq, event_type, payload_json),
            )
            cursor.execute(
                """
                UPDATE playback_sessions
                SET next_event_seq = %s,
                    updated_at = NOW()
                WHERE session_id = %s
                """,
                (seq, session_id),
            )
    return seq


def list_session_events(
    conn: psycopg.Connection,
    session_id: str,
    after_seq: int = 0,
) -> list[SessionEvent]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT session_id, seq, event_type, payload, created_at
            FROM session_events
            WHERE session_id = %s AND seq > %s
            ORDER BY seq ASC
            """,
            (session_id, after_seq),
        )
        rows = cursor.fetchall()
    events: list[SessionEvent] = []
    for row in rows:
        payload_value = row["payload"]
        if isinstance(payload_value, str):
            payload = json.loads(payload_value)
        else:
            payload = dict(payload_value)
        events.append(
            SessionEvent(
                session_id=row["session_id"],
                seq=int(row["seq"]),
                event_type=row["event_type"],
                payload=payload,
                created_at=row["created_at"],
            )
        )
    return events


def session_requested_at_by_paper_id(conn: psycopg.Connection, session_id: str) -> dict[str, datetime]:
    session = get_playback_session(conn, session_id)
    request_times: dict[str, datetime] = {}
    if session is not None:
        request_times[session.root_paper_id] = session.started_at
    for event in list_session_events(conn, session_id):
        paper_id = ""
        if event.event_type == "session_started":
            root_paper = event.payload.get("root_paper")
            if isinstance(root_paper, Mapping):
                paper_id = str(root_paper.get("id") or "")
        elif event.event_type == "session_next_requested":
            paper_id = str(event.payload.get("to_paper_id") or "")
        elif event.event_type == "session_regenerated":
            paper_id = str(event.payload.get("paper_id") or "")
        if paper_id and paper_id not in request_times:
            request_times[paper_id] = event.created_at
    return request_times


def append_session_trail_item(conn: psycopg.Connection, session_id: str, paper_id: str) -> int:
    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT COALESCE(MAX(position), 0) AS position
                FROM session_trail_items
                WHERE session_id = %s
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            position = int(row["position"]) + 1 if row is not None else 1
            cursor.execute(
                """
                INSERT INTO session_trail_items (session_id, position, paper_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id, paper_id) DO NOTHING
                """,
                (session_id, position, paper_id),
            )
    return position


def list_session_trail_paper_ids(conn: psycopg.Connection, session_id: str) -> list[str]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT paper_id
            FROM session_trail_items
            WHERE session_id = %s
            ORDER BY position ASC
            """,
            (session_id,),
        )
        return [row["paper_id"] for row in cursor.fetchall()]


def set_session_queue_item(conn: psycopg.Connection, session_id: str, paper_id: str) -> int:
    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM session_queue_items
                WHERE session_id = %s
                """,
                (session_id,),
            )
            cursor.execute(
                """
                INSERT INTO session_queue_items (session_id, position, paper_id)
                VALUES (%s, 1, %s)
                ON CONFLICT (session_id, paper_id) DO UPDATE
                SET position = EXCLUDED.position
                """,
                (session_id, paper_id),
            )
    return 1


def remove_session_queue_item(conn: psycopg.Connection, session_id: str, paper_id: str) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM session_queue_items
            WHERE session_id = %s AND paper_id = %s
            """,
            (session_id, paper_id),
        )
        return cursor.rowcount > 0


def pop_session_queue_item(conn: psycopg.Connection, session_id: str) -> str | None:
    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT paper_id
                FROM session_queue_items
                WHERE session_id = %s
                ORDER BY position ASC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            paper_id = row["paper_id"]
            cursor.execute(
                """
                DELETE FROM session_queue_items
                WHERE session_id = %s AND paper_id = %s
                """,
                (session_id, paper_id),
            )
    return paper_id


def list_session_queue_paper_ids(conn: psycopg.Connection, session_id: str) -> list[str]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT paper_id
            FROM session_queue_items
            WHERE session_id = %s
            ORDER BY position ASC
            """,
            (session_id,),
        )
        return [row["paper_id"] for row in cursor.fetchall()]
