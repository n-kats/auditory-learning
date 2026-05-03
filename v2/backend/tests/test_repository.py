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
        if normalized.startswith("CREATE TABLE IF NOT EXISTS favorites"):
            self._result = None
            return
        if normalized.startswith("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS"):
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
            session_id, paper_id, current_page, prompt_text, model_name, created_at, updated_at = params
            self.state["sessions"][session_id] = (
                session_id,
                paper_id,
                current_page,
                prompt_text,
                model_name,
                0,
                0,
                0,
                0,
                Decimal("0"),
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
                updated_at,
            )
            self._result = None
            return

        if normalized.startswith("UPDATE sessions SET prompt_text ="):
            prompt_text, model_name, updated_at, session_id = params
            session = self.state["sessions"][session_id]
            self.state["sessions"][session_id] = (
                session[0],
                session[1],
                session[2],
                prompt_text,
                model_name,
                session[5],
                session[6],
                session[7],
                session[8],
                session[9],
                session[10],
                updated_at,
            )
            self._result = None
            return

        if normalized.startswith("SELECT 1 FROM favorites WHERE paper_id ="):
            paper_id = params[0]
            self._result = (1,) if paper_id in self.state["favorites"] else None
            return

        if normalized.startswith("INSERT INTO favorites (paper_id, favorited_at)"):
            paper_id, favorited_at = params
            self.state["favorites"][paper_id] = (favorited_at,)
            self._result = None
            return

        if normalized.startswith("DELETE FROM favorites WHERE paper_id ="):
            paper_id = params[0]
            self.state["favorites"].pop(paper_id, None)
            self._result = None
            return

        if normalized.startswith("SELECT s.session_id, s.paper_id, p.source_url, p.page_num, s.current_page, s.prompt_text, s.model_name, s.total_generation_count, s.total_generation_elapsed_ms, s.total_input_tokens, s.total_output_tokens, s.total_cost_usd, s.created_at, s.updated_at FROM sessions s JOIN papers p ON p.paper_id = s.paper_id WHERE s.session_id ="):
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
            )
            return

        if normalized.startswith("SELECT s.session_id, s.paper_id, p.source_url, p.page_num, s.current_page, s.prompt_text, s.model_name, s.total_generation_count, s.total_generation_elapsed_ms, s.total_input_tokens, s.total_output_tokens, s.total_cost_usd, s.created_at, s.updated_at FROM sessions s JOIN papers p ON p.paper_id = s.paper_id ORDER BY s.updated_at DESC LIMIT"):
            rows = []
            for session in sorted(self.state["sessions"].values(), key=lambda row: row[11], reverse=True):
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
                    )
                )
            self._result = rows
            return

        if normalized.startswith("SELECT DISTINCT ON (p.paper_id)"):
            rows = []
            for paper_id in sorted(self.state["favorites"].keys()):
                favorite_sessions = [row for row in self.state["sessions"].values() if row[1] == paper_id]
                session = sorted(favorite_sessions, key=lambda row: row[11], reverse=True)[0]
                paper = self.state["papers_by_id"][paper_id]
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
                    )
                )
            self._result = rows
            return

        if normalized.startswith("INSERT INTO session_results"):
            result_id, paper_id, session_id, page_num, prompt_text, model_name, explanation, audio_status, audio_error, created_at, updated_at = params
            key = (session_id, page_num, prompt_text, model_name)
            self.state["session_results"][key] = (
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
                updated_at,
            )
            self._result = (result_id,)
            return

        if normalized.startswith("SELECT result_id, paper_id, session_id, page_num, prompt_text, model_name, explanation, audio_status, audio_error, created_at, updated_at FROM session_results WHERE session_id ="):
            session_id, page_num, prompt_text, model_name = params
            self._result = self.state["session_results"].get((session_id, page_num, prompt_text, model_name))
            return

        if normalized.startswith("INSERT INTO session_usage_records"):
            (
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
                    prompt_text,
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
                session[5] + 1,
                session[6] + elapsed_ms,
                session[7] + input_tokens,
                session[8] + output_tokens,
                session[9] + Decimal(str(cost_usd)),
                session[10],
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
    state = {"papers": {}, "papers_by_id": {}, "sessions": {}, "favorites": {}, "session_results": {}, "session_usage_records": []}

    def fake_connect(dsn: str):
        assert dsn == "postgresql://example"
        return FakeConnection(state)

    monkeypatch.setattr(repository_module, "uuid4", lambda: UUID("12345678-1234-5678-1234-567812345678"))

    repository = repository_module.Repository("postgresql://example", connect=fake_connect)

    session_id = repository.create_session_id()
    repository.upsert_document(session_id, "https://arxiv.org/pdf/2604.16347", 12, prompt_text="prompt-1", model_name="model-1")

    snapshot = repository.get_document(session_id)
    assert snapshot is not None
    assert snapshot["request_id"] == session_id
    assert snapshot["source_url"] == "https://arxiv.org/pdf/2604.16347"
    assert snapshot["page_num"] == 12
    assert snapshot["current_page"] == 1
    assert snapshot["prompt_text"] == "prompt-1"
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

    repository.toggle_favorite(session_id)
    favorites = repository.list_favorites(limit=10)
    assert len(favorites) == 1
    assert favorites[0]["request_id"] == session_id

    updated = repository.update_session_settings(session_id, prompt_text="prompt-2", model_name="model-2")
    assert updated is not None
    assert updated["prompt_text"] == "prompt-2"
    assert updated["model_name"] == "model-2"

    documents = repository.list_documents(limit=10)
    assert len(documents) == 1
    assert documents[0]["request_id"] == session_id
    assert documents[0]["source_url"] == "https://arxiv.org/pdf/2604.16347"
    assert documents[0]["current_page"] == 7
    assert documents[0]["prompt_text"] == "prompt-2"
    assert documents[0]["model_name"] == "model-2"

    result = repository.upsert_result(
        session_id,
        7,
        "説明文",
        prompt_text="prompt-2",
        model_name="model-2",
        audio_status="ready",
        audio_error=None,
    )
    assert result is not None
    fetched_result = repository.get_result(session_id, 7, prompt_text="prompt-2", model_name="model-2")
    assert fetched_result is not None
    assert fetched_result["explanation"] == "説明文"

    usage_row = repository.record_session_usage(
        session_id,
        paper_id=str(snapshot["paper_id"]),
        result_id=str(result["result_id"]),
        kind="explanation",
        page_num=7,
        prompt_text="prompt-2",
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
