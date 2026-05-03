from __future__ import annotations

import threading
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol
from uuid import uuid4

import psycopg
from psycopg.rows import tuple_row
from psycopg.types.json import Json


DEFAULT_MODEL_NAME = "gpt-5.4-mini"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ConnectionFactory(Protocol):
    def __call__(self, dsn: str) -> psycopg.Connection[tuple]: ...


class Repository:
    def __init__(self, dsn: str, connect: ConnectionFactory = psycopg.connect) -> None:
        self._dsn = dsn
        self._connect_factory = connect
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> psycopg.Connection[tuple]:
        connection = self._connect_factory(self._dsn)
        connection.row_factory = tuple_row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS papers (
                      paper_id TEXT PRIMARY KEY,
                      source_url TEXT NOT NULL UNIQUE,
                      page_num INTEGER NOT NULL,
                      created_at TIMESTAMPTZ NOT NULL,
                      updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                      session_id TEXT PRIMARY KEY,
                      paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                      current_page INTEGER NOT NULL DEFAULT 1,
                      prompt_text TEXT NOT NULL DEFAULT '',
                      model_name TEXT NOT NULL DEFAULT 'gpt-5.4-mini',
                      total_generation_count INTEGER NOT NULL DEFAULT 0,
                      total_generation_elapsed_ms INTEGER NOT NULL DEFAULT 0,
                      total_input_tokens INTEGER NOT NULL DEFAULT 0,
                      total_output_tokens INTEGER NOT NULL DEFAULT 0,
                      total_cost_usd NUMERIC(18, 6) NOT NULL DEFAULT 0,
                      created_at TIMESTAMPTZ NOT NULL,
                      updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_results (
                      result_id TEXT PRIMARY KEY,
                      paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                      session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                      page_num INTEGER NOT NULL,
                      prompt_text TEXT NOT NULL DEFAULT '',
                      model_name TEXT NOT NULL DEFAULT 'gpt-5.4-mini',
                      explanation TEXT NOT NULL,
                      audio_status TEXT NOT NULL DEFAULT 'ready',
                      audio_error TEXT,
                      created_at TIMESTAMPTZ NOT NULL,
                      updated_at TIMESTAMPTZ NOT NULL,
                      UNIQUE (session_id, page_num, prompt_text, model_name)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_usage_records (
                      usage_id TEXT PRIMARY KEY,
                      session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                      paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                      result_id TEXT REFERENCES session_results(result_id) ON DELETE SET NULL,
                      kind TEXT NOT NULL DEFAULT 'explanation',
                      page_num INTEGER NOT NULL,
                      prompt_text TEXT NOT NULL DEFAULT '',
                      model_name TEXT NOT NULL DEFAULT 'gpt-5.4-mini',
                      elapsed_ms INTEGER NOT NULL DEFAULT 0,
                      input_tokens INTEGER,
                      output_tokens INTEGER,
                      cost_usd NUMERIC(18, 6) NOT NULL DEFAULT 0,
                      detail JSONB NOT NULL DEFAULT '{}'::jsonb,
                      created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS favorites (
                      paper_id TEXT PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
                      favorited_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE sessions
                    ADD COLUMN IF NOT EXISTS total_generation_count INTEGER NOT NULL DEFAULT 0
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE sessions
                    ADD COLUMN IF NOT EXISTS total_generation_elapsed_ms INTEGER NOT NULL DEFAULT 0
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE sessions
                    ADD COLUMN IF NOT EXISTS total_input_tokens INTEGER NOT NULL DEFAULT 0
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE sessions
                    ADD COLUMN IF NOT EXISTS total_output_tokens INTEGER NOT NULL DEFAULT 0
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE sessions
                    ADD COLUMN IF NOT EXISTS total_cost_usd NUMERIC(18, 6) NOT NULL DEFAULT 0
                    """
                )
            connection.commit()

    def create_session_id(self) -> str:
        return str(uuid4())

    def get_or_create_request_id(self, source_url: str) -> str:
        # 互換のため残す。session 単位の実行 ID を新規作成する。
        return self.create_session_id()

    def _upsert_paper(self, source_url: str, page_num: int) -> str:
        paper_id = source_url
        now = _utc_now()
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO papers (paper_id, source_url, page_num, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source_url)
                    DO UPDATE SET
                      page_num = EXCLUDED.page_num,
                      updated_at = EXCLUDED.updated_at
                    RETURNING paper_id
                    """,
                    (paper_id, source_url, page_num, now, now),
                )
                resolved_paper_id = cursor.fetchone()[0]
            connection.commit()
        return str(resolved_paper_id)

    def _upsert_session(
        self,
        session_id: str,
        paper_id: str,
        page_num: int,
        current_page: int = 1,
        prompt_text: str = "",
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        now = _utc_now()
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sessions (
                      session_id,
                      paper_id,
                      current_page,
                      prompt_text,
                      model_name,
                      total_generation_count,
                      total_generation_elapsed_ms,
                      total_input_tokens,
                      total_output_tokens,
                      total_cost_usd,
                      created_at,
                      updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, 0, 0, 0, 0, 0, %s, %s)
                    ON CONFLICT (session_id)
                    DO UPDATE SET
                      paper_id = EXCLUDED.paper_id,
                      current_page = EXCLUDED.current_page,
                      prompt_text = EXCLUDED.prompt_text,
                      model_name = EXCLUDED.model_name,
                      updated_at = EXCLUDED.updated_at
                    """,
                    (session_id, paper_id, current_page, prompt_text, model_name, now, now),
                )
            connection.commit()

    def upsert_document(
        self,
        request_id: str,
        source_url: str,
        page_num: int,
        current_page: int = 1,
        prompt_text: str = "",
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        paper_id = self._upsert_paper(source_url, page_num)
        self._upsert_session(
            request_id,
            paper_id,
            page_num,
            current_page=current_page,
            prompt_text=prompt_text,
            model_name=model_name,
        )

    def _fetch_session_row(self, request_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                      s.session_id,
                      s.paper_id,
                      p.source_url,
                      p.page_num,
                      s.current_page,
                      s.prompt_text,
                      s.model_name,
                      s.total_generation_count,
                      s.total_generation_elapsed_ms,
                      s.total_input_tokens,
                      s.total_output_tokens,
                      s.total_cost_usd,
                      s.created_at,
                      s.updated_at
                    FROM sessions s
                    JOIN papers p ON p.paper_id = s.paper_id
                    WHERE s.session_id = %s
                    """,
                    (request_id,),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return {
            "request_id": str(row[0]),
            "paper_id": str(row[1]),
            "source_url": str(row[2]),
            "page_num": row[3],
            "current_page": row[4],
            "prompt_text": str(row[5]),
            "model_name": str(row[6]),
            "total_generation_count": row[7],
            "total_generation_elapsed_ms": row[8],
            "total_input_tokens": row[9],
            "total_output_tokens": row[10],
            "total_cost_usd": float(row[11]),
            "created_at": row[12],
            "updated_at": row[13],
        }

    def update_current_page(self, request_id: str, current_page: int) -> None:
        now = _utc_now()
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE sessions
                    SET current_page = %s, updated_at = %s
                    WHERE session_id = %s
                    """,
                    (current_page, now, request_id),
                )
            connection.commit()

    def get_document(self, request_id: str) -> dict[str, object] | None:
        row = self._fetch_session_row(request_id)
        if row is None:
            return None
        return {
            "request_id": row["request_id"],
            "paper_id": row["paper_id"],
            "source_url": row["source_url"],
            "page_num": row["page_num"],
            "current_page": row["current_page"],
            "prompt_text": row["prompt_text"],
            "model_name": row["model_name"],
            "total_generation_count": row["total_generation_count"],
            "total_generation_elapsed_ms": row["total_generation_elapsed_ms"],
            "total_input_tokens": row["total_input_tokens"],
            "total_output_tokens": row["total_output_tokens"],
            "total_cost_usd": row["total_cost_usd"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_documents(self, limit: int = 20) -> list[dict[str, object]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                      s.session_id,
                      s.paper_id,
                      p.source_url,
                      p.page_num,
                      s.current_page,
                      s.prompt_text,
                      s.model_name,
                      s.total_generation_count,
                      s.total_generation_elapsed_ms,
                      s.total_input_tokens,
                      s.total_output_tokens,
                      s.total_cost_usd,
                      s.created_at,
                      s.updated_at
                    FROM sessions s
                    JOIN papers p ON p.paper_id = s.paper_id
                    ORDER BY s.updated_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
        return [
            {
                "request_id": str(row[0]),
                "paper_id": str(row[1]),
                "source_url": str(row[2]),
                "page_num": row[3],
                "current_page": row[4],
                "prompt_text": str(row[5]),
                "model_name": str(row[6]),
                "total_generation_count": row[7],
                "total_generation_elapsed_ms": row[8],
                "total_input_tokens": row[9],
                "total_output_tokens": row[10],
                "total_cost_usd": float(row[11]),
                "created_at": row[12],
                "updated_at": row[13],
            }
            for row in rows
        ]

    def update_session_settings(
        self,
        request_id: str,
        *,
        prompt_text: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, object] | None:
        current = self.get_document(request_id)
        if current is None:
            return None

        next_prompt_text = current["prompt_text"] if prompt_text is None else prompt_text
        next_model_name = current["model_name"] if model_name is None else model_name
        now = _utc_now()
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE sessions
                    SET prompt_text = %s, model_name = %s, updated_at = %s
                    WHERE session_id = %s
                    """,
                    (next_prompt_text, next_model_name, now, request_id),
                )
            connection.commit()
        return self.get_document(request_id)

    def record_session_usage(
        self,
        session_id: str,
        *,
        paper_id: str,
        result_id: str | None,
        kind: str,
        page_num: int,
        prompt_text: str,
        model_name: str,
        elapsed_ms: int,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: Decimal,
        detail: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        current = self.get_document(session_id)
        if current is None:
            return None
        now = _utc_now()
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO session_usage_records (
                      usage_id,
                      session_id,
                      paper_id,
                      result_id,
                      kind,
                      page_num,
                      prompt_text,
                      model_name,
                      elapsed_ms,
                      input_tokens,
                      output_tokens,
                      cost_usd,
                      detail,
                      created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid4()),
                        session_id,
                        paper_id,
                        result_id,
                        kind,
                        page_num,
                        prompt_text,
                        model_name,
                        elapsed_ms,
                        input_tokens,
                        output_tokens,
                        cost_usd,
                        Json(detail or {}),
                        now,
                    ),
                )
                cursor.execute(
                    """
                    UPDATE sessions
                    SET
                      total_generation_count = total_generation_count + 1,
                      total_generation_elapsed_ms = total_generation_elapsed_ms + %s,
                      total_input_tokens = total_input_tokens + %s,
                      total_output_tokens = total_output_tokens + %s,
                      total_cost_usd = total_cost_usd + %s,
                      updated_at = %s
                    WHERE session_id = %s
                    """,
                    (
                        elapsed_ms,
                        input_tokens or 0,
                        output_tokens or 0,
                        cost_usd,
                        now,
                        session_id,
                    ),
                )
            connection.commit()
        return self.get_document(session_id)

    def toggle_favorite(self, request_id: str) -> bool:
        current = self.get_document(request_id)
        if current is None:
            raise KeyError(request_id)

        paper_id = str(current["paper_id"])
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1
                    FROM favorites
                    WHERE paper_id = %s
                    """,
                    (paper_id,),
                )
                is_favorited = cursor.fetchone() is not None
                if is_favorited:
                    cursor.execute(
                        """
                        DELETE FROM favorites
                        WHERE paper_id = %s
                        """,
                        (paper_id,),
                    )
                    connection.commit()
                    return False
                cursor.execute(
                    """
                    INSERT INTO favorites (paper_id, favorited_at)
                    VALUES (%s, %s)
                    """,
                    (paper_id, _utc_now()),
                )
            connection.commit()
            return True

    def is_favorited(self, request_id: str) -> bool:
        current = self.get_document(request_id)
        if current is None:
            return False
        paper_id = str(current["paper_id"])
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1
                    FROM favorites
                    WHERE paper_id = %s
                    """,
                    (paper_id,),
                )
                return cursor.fetchone() is not None

    def list_favorites(self, limit: int = 20) -> list[dict[str, object]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT DISTINCT ON (p.paper_id)
                      s.session_id,
                      s.paper_id,
                      p.source_url,
                      p.page_num,
                      s.current_page,
                      s.prompt_text,
                      s.model_name,
                      s.total_generation_count,
                      s.total_generation_elapsed_ms,
                      s.total_input_tokens,
                      s.total_output_tokens,
                      s.total_cost_usd,
                      s.created_at,
                      s.updated_at
                    FROM favorites f
                    JOIN papers p ON p.paper_id = f.paper_id
                    JOIN sessions s ON s.paper_id = p.paper_id
                    ORDER BY p.paper_id, s.updated_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
        return [
            {
                "request_id": str(row[0]),
                "paper_id": str(row[1]),
                "source_url": str(row[2]),
                "page_num": row[3],
                "current_page": row[4],
                "prompt_text": str(row[5]),
                "model_name": str(row[6]),
                "total_generation_count": row[7],
                "total_generation_elapsed_ms": row[8],
                "total_input_tokens": row[9],
                "total_output_tokens": row[10],
                "total_cost_usd": float(row[11]),
                "created_at": row[12],
                "updated_at": row[13],
                "is_favorited": True,
            }
            for row in rows
        ]

    def upsert_result(
        self,
        request_id: str,
        page_num: int,
        explanation: str,
        *,
        prompt_text: str = "",
        model_name: str = DEFAULT_MODEL_NAME,
        audio_status: str = "ready",
        audio_error: str | None = None,
    ) -> dict[str, object] | None:
        current = self.get_document(request_id)
        if current is None:
            return None

        result_id = str(uuid4())
        now = _utc_now()
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO session_results (
                      result_id, paper_id, session_id, page_num, prompt_text, model_name, explanation, audio_status, audio_error, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id, page_num, prompt_text, model_name)
                    DO UPDATE SET
                      explanation = EXCLUDED.explanation,
                      audio_status = EXCLUDED.audio_status,
                      audio_error = EXCLUDED.audio_error,
                      updated_at = EXCLUDED.updated_at
                    RETURNING result_id
                    """,
                    (
                        result_id,
                        current["paper_id"],
                        request_id,
                        page_num,
                        prompt_text,
                        model_name,
                        explanation,
                        audio_status,
                        audio_error,
                        now,
                        now,
                    ),
                )
                resolved_result_id = cursor.fetchone()[0]
            connection.commit()
        return {
            "result_id": str(resolved_result_id),
            "request_id": request_id,
            "paper_id": current["paper_id"],
            "page_num": page_num,
            "prompt_text": prompt_text,
            "model_name": model_name,
            "explanation": explanation,
            "audio_status": audio_status,
            "audio_error": audio_error,
            "created_at": now,
            "updated_at": now,
        }

    def get_result(
        self,
        request_id: str,
        page_num: int,
        *,
        prompt_text: str = "",
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> dict[str, object] | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                      result_id,
                      paper_id,
                      session_id,
                      page_num,
                      prompt_text,
                      model_name,
                      explanation,
                      audio_status,
                      audio_error,
                      created_at,
                      updated_at
                    FROM session_results
                    WHERE session_id = %s AND page_num = %s AND prompt_text = %s AND model_name = %s
                    """,
                    (request_id, page_num, prompt_text, model_name),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return {
            "result_id": str(row[0]),
            "paper_id": str(row[1]),
            "request_id": str(row[2]),
            "page_num": row[3],
            "prompt_text": str(row[4]),
            "model_name": str(row[5]),
            "explanation": str(row[6]),
            "audio_status": str(row[7]),
            "audio_error": row[8],
            "created_at": row[9],
            "updated_at": row[10],
        }
