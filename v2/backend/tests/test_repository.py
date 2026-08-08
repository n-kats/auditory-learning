from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from v2_auditory_learning import db as repository_module


class FakeCursor:
    def __init__(self, state: dict[str, dict[str, tuple]], inputs: list[tuple[str, tuple]]) -> None:
        self.state = state
        self.inputs = inputs
        self._result: tuple | list[tuple] | None = None

    def execute(self, query: str, params: tuple | None = None) -> None:
        self.inputs.append((query, tuple() if params is None else params))
        normalized = " ".join(query.split())

        if normalized.startswith("CREATE TABLE IF NOT EXISTS papers"):
            self._result = None
            return
        if normalized.startswith("CREATE TABLE IF NOT EXISTS sessions"):
            self._result = None
            return
        if normalized.startswith("CREATE TABLE IF NOT EXISTS session_results"):
            self._result = None
            return
        if normalized.startswith("CREATE TABLE IF NOT EXISTS session_usage_records"):
            self._result = None
            return
        if normalized.startswith("CREATE TABLE IF NOT EXISTS favorite_pages"):
            self._result = None
            return
        if normalized.startswith("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS"):
            self._result = None
            return
        if normalized.startswith("ALTER TABLE sessions DROP COLUMN IF EXISTS"):
            self._result = None
            return
        if normalized.startswith("ALTER TABLE session_results ADD COLUMN IF NOT EXISTS"):
            self._result = None
            return
        if normalized.startswith("ALTER TABLE session_results DROP COLUMN IF EXISTS"):
            self._result = None
            return
        if normalized.startswith("ALTER TABLE session_usage_records DROP COLUMN IF EXISTS"):
            self._result = None
            return

        if normalized.startswith("INSERT INTO papers"):
            paper_id, source_url, page_num, created_at, updated_at = params
            existing = self.state["papers"].get(source_url)
            if existing is None:
                self.state["papers"][source_url] = (paper_id, source_url, page_num, created_at, updated_at)
                self.state["papers_by_id"][paper_id] = (paper_id, source_url, page_num, created_at, updated_at)
                self._result = (paper_id,)
            else:
                self.state["papers"][source_url] = (existing[0], source_url, page_num, existing[3], updated_at)
                self.state["papers_by_id"][existing[0]] = (existing[0], source_url, page_num, existing[3], updated_at)
                self._result = (existing[0],)
            return

        if normalized.startswith("INSERT INTO sessions ("):
            (
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
                updated_at,
            ) = params
            self.state["sessions"][session_id] = (
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
                Decimal(str(total_cost_usd)),
                created_at,
                updated_at,
            )
            self._result = None
            return

        if normalized.startswith("UPDATE sessions SET current_page ="):
            current_page, updated_at, session_id = params
            session = self.state["sessions"][session_id]
            self.state["sessions"][session_id] = (
                session[0],
                session[1],
                current_page,
                session[3],
                session[4],
                session[5],
                session[6],
                session[7],
                session[8],
                session[9],
                session[10],
                session[11],
                session[12],
                updated_at,
            )
            self._result = None
            return

        if normalized.startswith("UPDATE sessions SET prompt_explain_text = %s, prompt_speak_text = %s, model_name = %s, reasoning_effort = %s, updated_at = %s"):
            prompt_explain_text, prompt_speak_text, model_name, reasoning_effort, updated_at, session_id = params
            session = self.state["sessions"][session_id]
            self.state["sessions"][session_id] = (
                session[0],
                session[1],
                session[2],
                prompt_explain_text,
                prompt_speak_text,
                model_name,
                reasoning_effort,
                session[7],
                session[8],
                session[9],
                session[10],
                session[11],
                session[12],
                updated_at,
            )
            self._result = None
            return

        if normalized.startswith("SELECT 1 FROM favorite_pages WHERE session_id ="):
            session_id, page_num = params
            self._result = (1,) if (session_id, page_num) in self.state["favorite_pages"] else None
            return

        if normalized.startswith("INSERT INTO favorite_pages (session_id, page_num, favorited_at)"):
            session_id, page_num, favorited_at = params
            self.state["favorite_pages"][(session_id, page_num)] = (favorited_at,)
            self._result = None
            return

        if normalized.startswith("DELETE FROM favorite_pages WHERE session_id ="):
            session_id, page_num = params
            self.state["favorite_pages"].pop((session_id, page_num), None)
            self._result = None
            return

        if normalized.startswith("SELECT s.session_id, s.paper_id, p.source_url, p.page_num, s.current_page, s.prompt_explain_text, s.prompt_speak_text, s.model_name, s.reasoning_effort, s.total_generation_count, s.total_generation_elapsed_ms, s.total_input_tokens, s.total_output_tokens, s.total_cost_usd, s.created_at, s.updated_at FROM sessions s JOIN papers p ON p.paper_id = s.paper_id WHERE s.session_id ="):
            session_id = params[0]
            session = self.state["sessions"].get(session_id)
            if session is None:
                self._result = None
                return
            paper = self.state["papers_by_id"][session[1]]
            self._result = (
                session[0],
                session[1],
                paper[1],
                paper[2],
                session[2],
                session[3],
                session[4],
                session[5],
                session[6],
                session[7],
                session[8],
                session[9],
                session[10],
                session[11],
                session[12],
                session[13],
            )
            return

        if normalized.startswith("SELECT s.session_id, s.paper_id, p.source_url, p.page_num, s.current_page, s.prompt_explain_text, s.prompt_speak_text, s.model_name, s.reasoning_effort, s.total_generation_count, s.total_generation_elapsed_ms, s.total_input_tokens, s.total_output_tokens, s.total_cost_usd, s.created_at, s.updated_at FROM sessions s JOIN papers p ON p.paper_id = s.paper_id ORDER BY s.updated_at DESC LIMIT"):
            rows = []
            for session in sorted(self.state["sessions"].values(), key=lambda row: row[13], reverse=True):
                paper = self.state["papers_by_id"][session[1]]
                rows.append(
                    (
                        session[0],
                        session[1],
                        paper[1],
                        paper[2],
                        session[2],
                        session[3],
                        session[4],
                        session[5],
                        session[6],
                        session[7],
                        session[8],
                        session[9],
                        session[10],
                        session[11],
                        session[12],
                        session[13],
                    )
                )
            self._result = rows
            return

        if normalized.startswith("SELECT f.session_id, f.page_num, f.favorited_at, s.paper_id, p.source_url, p.page_num, s.current_page"):
            rows = []
            favorite_rows = sorted(self.state["favorite_pages"].items(), key=lambda item: item[1][0], reverse=True)
            for (session_id, favorite_page_num), favorite_row in favorite_rows:
                session = self.state["sessions"][session_id]
                paper = self.state["papers_by_id"][session[1]]
                rows.append(
                    (
                        session[0],
                        favorite_page_num,
                        favorite_row[0],
                        session[1],
                        paper[1],
                        paper[2],
                        session[2],
                        session[3],
                        session[4],
                        session[5],
                        session[6],
                        session[7],
                        session[8],
                        session[9],
                        session[10],
                        session[11],
                        session[12],
                        session[13],
                    )
                )
            self._result = rows
            return

        if normalized.startswith("INSERT INTO session_results"):
            (
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
                updated_at,
            ) = params
            key = (session_id, page_num, prompt_explain_text, prompt_speak_text, model_name)
            self.state["session_results"][key] = (
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
                updated_at,
            )
            self._result = (result_id,)
            return

        if normalized.startswith("SELECT result_id, paper_id, session_id, page_num, prompt_explain_text, prompt_speak_text, model_name, explanation, speech_text, audio_status, audio_error, created_at, updated_at FROM session_results WHERE session_id ="):
            session_id, page_num, prompt_explain_text, prompt_speak_text, model_name = params
            self._result = self.state["session_results"].get((session_id, page_num, prompt_explain_text, prompt_speak_text, model_name))
            return

        if normalized.startswith("INSERT INTO session_usage_records"):
            (
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
                created_at,
            ) = params
            self.state["session_usage_records"].append(
                (
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
                    created_at,
                )
            )
            self._result = None
            return

        if normalized.startswith("UPDATE sessions SET total_generation_count = total_generation_count +"):
            elapsed_ms, input_tokens, output_tokens, cost_usd, updated_at, session_id = params
            session = self.state["sessions"][session_id]
            self.state["sessions"][session_id] = (
                session[0],
                session[1],
                session[2],
                session[3],
                session[4],
                session[5],
                session[6],
                session[7] + 1,
                session[8] + elapsed_ms,
                session[9] + input_tokens,
                session[10] + output_tokens,
                session[11] + Decimal(str(cost_usd)),
                session[12],
                updated_at,
            )
            self._result = None
            return

        raise AssertionError(f"unexpected query: {query}")

    def fetchone(self):
        if isinstance(self._result, list):
            return self._result[0] if self._result else None
        return self._result

    def fetchall(self):
        if isinstance(self._result, list):
            return self._result
        return [] if self._result is None else [self._result]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, state: dict[str, dict[str, tuple]]) -> None:
        self.state = state
        self.inputs: list[tuple[str, tuple]] = []
        self.row_factory = None

    def cursor(self):
        return FakeCursor(self.state, self.inputs)

    def commit(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_repository_persists_session_document_result_and_usage(monkeypatch) -> None:
    state = {"papers": {}, "papers_by_id": {}, "sessions": {}, "favorite_pages": {}, "session_results": {}, "session_usage_records": []}

    def fake_connect(dsn: str):
        assert dsn == "postgresql://example"
        return FakeConnection(state)

    monkeypatch.setattr(repository_module, "uuid4", lambda: UUID("12345678-1234-5678-1234-567812345678"))

    repository = repository_module.Repository("postgresql://example", connect=fake_connect)

    session_id = repository.create_session_id()
    repository.upsert_document(
        session_id,
        "https://arxiv.org/pdf/2604.16347",
        12,
        prompt_explain_text="prompt-1",
        prompt_speak_text="prompt-speak-1",
        model_name="model-1",
    )

    snapshot = repository.get_document(session_id)
    assert snapshot is not None
    assert snapshot["request_id"] == session_id
    assert snapshot["source_url"] == "https://arxiv.org/pdf/2604.16347"
    assert snapshot["page_num"] == 12
    assert snapshot["current_page"] == 1
    assert snapshot["prompt_explain_text"] == "prompt-1"
    assert snapshot["prompt_speak_text"] == "prompt-speak-1"
    assert snapshot["model_name"] == "model-1"
    assert snapshot["total_generation_count"] == 0
    assert snapshot["total_cost_usd"] == 0.0

    repository.update_current_page(session_id, 7)
    snapshot = repository.get_document(session_id)
    assert snapshot is not None
    assert snapshot["current_page"] == 7

    assert repository.is_favorited(session_id) is False
    assert repository.toggle_favorite(session_id) is True
    assert repository.is_favorited(session_id) is True
    assert repository.toggle_favorite(session_id) is False
    assert repository.is_favorited(session_id) is False

    repository.update_current_page(session_id, 7)
    assert repository.toggle_favorite(session_id) is True
    assert repository.toggle_favorite(session_id, page_num=3) is True
    favorites = repository.list_favorites(limit=10)
    assert len(favorites) == 2
    assert favorites[0]["request_id"] == session_id
    assert {item["favorite_page_num"] for item in favorites} == {7, 3}

    updated = repository.update_session_settings(
        session_id,
        prompt_explain_text="prompt-2",
        prompt_speak_text="prompt-speak-2",
        model_name="model-2",
    )
    assert updated is not None
    assert updated["prompt_explain_text"] == "prompt-2"
    assert updated["prompt_speak_text"] == "prompt-speak-2"
    assert updated["model_name"] == "model-2"

    documents = repository.list_documents(limit=10)
    assert len(documents) == 1
    assert documents[0]["request_id"] == session_id
    assert documents[0]["source_url"] == "https://arxiv.org/pdf/2604.16347"
    assert documents[0]["current_page"] == 7
    assert documents[0]["prompt_explain_text"] == "prompt-2"
    assert documents[0]["prompt_speak_text"] == "prompt-speak-2"
    assert documents[0]["model_name"] == "model-2"

    result = repository.upsert_result(
        session_id,
        7,
        "説明文",
        speech_text="読み上げ文",
        prompt_explain_text="prompt-2",
        prompt_speak_text="prompt-speak-2",
        model_name="model-2",
        audio_status="ready",
        audio_error=None,
    )
    assert result is not None
    fetched_result = repository.get_result(
        session_id,
        7,
        prompt_explain_text="prompt-2",
        prompt_speak_text="prompt-speak-2",
        model_name="model-2",
    )
    assert fetched_result is not None
    assert fetched_result["explanation"] == "説明文"
    assert fetched_result["speech_text"] == "読み上げ文"

    usage_row = repository.record_session_usage(
        session_id,
        paper_id=str(snapshot["paper_id"]),
        result_id=str(result["result_id"]),
        kind="explanation",
        page_num=7,
        model_name="model-2",
        elapsed_ms=1200,
        input_tokens=100,
        output_tokens=200,
        cost_usd=Decimal("0.123456"),
        detail={"kind": "explanation"},
    )
    assert usage_row is not None

    snapshot = repository.get_document(session_id)
    assert snapshot is not None
    assert snapshot["total_generation_count"] == 1
    assert snapshot["total_generation_elapsed_ms"] == 1200
    assert snapshot["total_input_tokens"] == 100
    assert snapshot["total_output_tokens"] == 200
    assert snapshot["total_cost_usd"] == float(Decimal("0.123456"))
