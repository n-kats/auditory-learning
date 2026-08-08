from __future__ import annotations

import threading
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol
from uuid import uuid4

import psycopg
from psycopg.rows import tuple_row
from psycopg.types.json import Json


DEFAULT_MODEL_NAME = "gpt-5.6-luna"


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
                      prompt_explain_text TEXT NOT NULL DEFAULT '',
                      prompt_speak_text TEXT NOT NULL DEFAULT '',
                      model_name TEXT NOT NULL DEFAULT 'gpt-5.6-luna',
                      reasoning_effort TEXT NOT NULL DEFAULT '',
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
                      prompt_explain_text TEXT NOT NULL DEFAULT '',
                      prompt_speak_text TEXT NOT NULL DEFAULT '',
                      model_name TEXT NOT NULL DEFAULT 'gpt-5.6-luna',
                      explanation TEXT NOT NULL,
                      speech_text TEXT NOT NULL DEFAULT '',
                      audio_status TEXT NOT NULL DEFAULT 'ready',
                      audio_error TEXT,
                      created_at TIMESTAMPTZ NOT NULL,
                      updated_at TIMESTAMPTZ NOT NULL,
                      UNIQUE (session_id, page_num, prompt_explain_text, prompt_speak_text, model_name)
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
                      model_name TEXT NOT NULL DEFAULT 'gpt-5.6-luna',
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
                    CREATE TABLE IF NOT EXISTS favorite_pages (
                      session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                      page_num INTEGER NOT NULL,
                      favorited_at TIMESTAMPTZ NOT NULL,
                      PRIMARY KEY (session_id, page_num)
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
                    ADD COLUMN IF NOT EXISTS prompt_explain_text TEXT NOT NULL DEFAULT ''
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE sessions
                    ADD COLUMN IF NOT EXISTS prompt_speak_text TEXT NOT NULL DEFAULT ''
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
                cursor.execute(
                    """
                    ALTER TABLE session_results
                    ADD COLUMN IF NOT EXISTS prompt_explain_text TEXT NOT NULL DEFAULT ''
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE session_results
                    ADD COLUMN IF NOT EXISTS prompt_speak_text TEXT NOT NULL DEFAULT ''
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE session_results
                    ADD COLUMN IF NOT EXISTS speech_text TEXT NOT NULL DEFAULT ''
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE sessions
                    ADD COLUMN IF NOT EXISTS reasoning_effort TEXT NOT NULL DEFAULT ''
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE sessions
                    DROP COLUMN IF EXISTS prompt_text
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE session_results
                    DROP COLUMN IF EXISTS prompt_text
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE session_usage_records
                    DROP COLUMN IF EXISTS prompt_text
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
        prompt_explain_text: str = "",
        prompt_speak_text: str = "",
        model_name: str = DEFAULT_MODEL_NAME,
        reasoning_effort: str = "",
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
                      prompt_explain_text,
                      prompt_speak_text,
                      model_name,
                      reasoning_effort,
                      total_generation_count,
                      total_generation_elapsed_ms,
                      total_input_tokens,
                      total_output_tokens,
                      total_cost_usd,
                      created_at,
                      updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id)
                    DO UPDATE SET
                      paper_id = EXCLUDED.paper_id,
                      current_page = EXCLUDED.current_page,
                      prompt_explain_text = EXCLUDED.prompt_explain_text,
                      prompt_speak_text = EXCLUDED.prompt_speak_text,
                      model_name = EXCLUDED.model_name,
                      reasoning_effort = EXCLUDED.reasoning_effort,
                      updated_at = EXCLUDED.updated_at
                    """,
                    (
                        session_id,
                        paper_id,
                        current_page,
                        prompt_explain_text,
                        prompt_speak_text,
                        model_name,
                        reasoning_effort,
                        0,
                        0,
                        0,
                        0,
                        0,
                        now,
                        now,
                    ),
                )
            connection.commit()

    def upsert_document(
        self,
        request_id: str,
        source_url: str,
        page_num: int,
        current_page: int = 1,
        prompt_explain_text: str = "",
        prompt_speak_text: str = "",
        model_name: str = DEFAULT_MODEL_NAME,
        reasoning_effort: str = "",
    ) -> None:
        paper_id = self._upsert_paper(source_url, page_num)
        self._upsert_session(
            request_id,
            paper_id,
            page_num,
            current_page=current_page,
            prompt_explain_text=prompt_explain_text,
            prompt_speak_text=prompt_speak_text,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
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
                      s.prompt_explain_text,
                      s.prompt_speak_text,
                      s.model_name,
                      s.reasoning_effort,
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
            "prompt_explain_text": str(row[5]),
            "prompt_speak_text": str(row[6]),
            "model_name": str(row[7]),
            "reasoning_effort": str(row[8]),
            "total_generation_count": row[9],
            "total_generation_elapsed_ms": row[10],
            "total_input_tokens": row[11],
            "total_output_tokens": row[12],
            "total_cost_usd": float(row[13]),
            "created_at": row[14],
            "updated_at": row[15],
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
            "prompt_explain_text": row["prompt_explain_text"],
            "prompt_speak_text": row["prompt_speak_text"],
            "model_name": row["model_name"],
            "reasoning_effort": row["reasoning_effort"],
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
                      s.prompt_explain_text,
                      s.prompt_speak_text,
                      s.model_name,
                      s.reasoning_effort,
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
                "prompt_explain_text": str(row[5]),
                "prompt_speak_text": str(row[6]),
                "model_name": str(row[7]),
                "reasoning_effort": str(row[8]),
                "total_generation_count": row[9],
                "total_generation_elapsed_ms": row[10],
                "total_input_tokens": row[11],
                "total_output_tokens": row[12],
                "total_cost_usd": float(row[13]),
                "created_at": row[14],
                "updated_at": row[15],
            }
            for row in rows
        ]

    def update_session_settings(
        self,
        request_id: str,
        *,
        prompt_explain_text: str | None = None,
        prompt_speak_text: str | None = None,
        model_name: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, object] | None:
        current = self.get_document(request_id)
        if current is None:
            return None

        next_prompt_explain_text = current["prompt_explain_text"] if prompt_explain_text is None else prompt_explain_text
        next_prompt_speak_text = current["prompt_speak_text"] if prompt_speak_text is None else prompt_speak_text
        next_model_name = current["model_name"] if model_name is None else model_name
        next_reasoning_effort = current["reasoning_effort"] if reasoning_effort is None else reasoning_effort
        now = _utc_now()
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE sessions
                    SET prompt_explain_text = %s, prompt_speak_text = %s, model_name = %s, reasoning_effort = %s, updated_at = %s
                    WHERE session_id = %s
                    """,
                    (next_prompt_explain_text, next_prompt_speak_text, next_model_name, next_reasoning_effort, now, request_id),
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
                      model_name,
                      elapsed_ms,
                      input_tokens,
                      output_tokens,
                      cost_usd,
                      detail,
                      created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid4()),
                        session_id,
                        paper_id,
                        result_id,
                        kind,
                        page_num,
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

    def toggle_favorite(self, request_id: str, page_num: int | None = None) -> bool:
        current = self.get_document(request_id)
        if current is None:
            raise KeyError(request_id)

        session_id = str(current["request_id"])
        target_page_num = int(page_num if page_num is not None else current["current_page"])
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1
                    FROM favorite_pages
                    WHERE session_id = %s AND page_num = %s
                    """,
                    (session_id, target_page_num),
                )
                is_favorited = cursor.fetchone() is not None
                if is_favorited:
                    cursor.execute(
                        """
                        DELETE FROM favorite_pages
                        WHERE session_id = %s AND page_num = %s
                        """,
                        (session_id, target_page_num),
                    )
                    connection.commit()
                    return False
                cursor.execute(
                    """
                    INSERT INTO favorite_pages (session_id, page_num, favorited_at)
                    VALUES (%s, %s, %s)
                    """,
                    (session_id, target_page_num, _utc_now()),
                )
            connection.commit()
            return True

    def is_favorited(self, request_id: str, page_num: int | None = None) -> bool:
        current = self.get_document(request_id)
        if current is None:
            return False
        session_id = str(current["request_id"])
        target_page_num = int(page_num if page_num is not None else current["current_page"])
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1
                    FROM favorite_pages
                    WHERE session_id = %s AND page_num = %s
                    """,
                    (session_id, target_page_num),
                )
                return cursor.fetchone() is not None

    def list_favorites(self, limit: int = 20) -> list[dict[str, object]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                      f.session_id,
                      f.page_num,
                      f.favorited_at,
                      s.paper_id,
                      p.source_url,
                      p.page_num,
                      s.current_page,
                      s.prompt_explain_text,
                      s.prompt_speak_text,
                      s.model_name,
                      s.reasoning_effort,
                      s.total_generation_count,
                      s.total_generation_elapsed_ms,
                      s.total_input_tokens,
                      s.total_output_tokens,
                      s.total_cost_usd,
                      s.created_at,
                      s.updated_at
                    FROM favorite_pages f
                    JOIN sessions s ON s.session_id = f.session_id
                    JOIN papers p ON p.paper_id = s.paper_id
                    ORDER BY f.favorited_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
        return [
            {
                "request_id": str(row[0]),
                "favorite_page_num": row[1],
                "favorited_at": row[2],
                "paper_id": str(row[3]),
                "source_url": str(row[4]),
                "page_num": row[5],
                "current_page": row[6],
                "prompt_explain_text": str(row[7]),
                "prompt_speak_text": str(row[8]),
                "model_name": str(row[9]),
                "reasoning_effort": str(row[10]),
                "total_generation_count": row[11],
                "total_generation_elapsed_ms": row[12],
                "total_input_tokens": row[13],
                "total_output_tokens": row[14],
                "total_cost_usd": float(row[15]),
                "created_at": row[16],
                "updated_at": row[17],
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
        speech_text: str = "",
        prompt_explain_text: str = "",
        prompt_speak_text: str = "",
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
                      result_id, paper_id, session_id, page_num, prompt_explain_text, prompt_speak_text, model_name, explanation, speech_text, audio_status, audio_error, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id, page_num, prompt_explain_text, prompt_speak_text, model_name)
                    DO UPDATE SET
                      explanation = EXCLUDED.explanation,
                      speech_text = EXCLUDED.speech_text,
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
                        prompt_explain_text,
                        prompt_speak_text,
                        model_name,
                        explanation,
                        speech_text,
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
            "prompt_explain_text": prompt_explain_text,
            "prompt_speak_text": prompt_speak_text,
            "model_name": model_name,
            "explanation": explanation,
            "speech_text": speech_text,
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
        prompt_explain_text: str = "",
        prompt_speak_text: str = "",
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
                      prompt_explain_text,
                      prompt_speak_text,
                      model_name,
                      explanation,
                      speech_text,
                      audio_status,
                      audio_error,
                      created_at,
                      updated_at
                    FROM session_results
                    WHERE session_id = %s AND page_num = %s AND prompt_explain_text = %s AND prompt_speak_text = %s AND model_name = %s
                    """,
                    (request_id, page_num, prompt_explain_text, prompt_speak_text, model_name),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return {
            "result_id": str(row[0]),
            "paper_id": str(row[1]),
            "request_id": str(row[2]),
            "page_num": row[3],
            "prompt_explain_text": str(row[4]),
            "prompt_speak_text": str(row[5]),
            "model_name": str(row[6]),
            "explanation": str(row[7]),
            "speech_text": str(row[8]),
            "audio_status": str(row[9]),
            "audio_error": row[10],
            "created_at": row[11],
            "updated_at": row[12],
        }
