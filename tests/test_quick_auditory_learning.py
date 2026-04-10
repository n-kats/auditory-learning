from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = ROOT / "quick-auditory-learning" / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from quick_auditory_learning.db import (
    ensure_schema,
    generation_cost_items_from_rows,
    generation_cost_total_cost_usd_from_rows,
    generation_cost_wall_elapsed_ms_from_rows,
    model_table_name,
    normalize_identifier,
)
from quick_auditory_learning.costs import estimate_completion_cost_usd, estimate_embedding_cost_usd
from quick_auditory_learning.arxiv_source import parse_arxiv_identifier, strip_arxiv_version
from quick_auditory_learning.importer import jsonl_import_is_stale, normalize_paper_record
from quick_auditory_learning.repository import get_paper_memo, upsert_paper_memo
from quick_auditory_learning.search import merge_scores
from quick_auditory_learning.session_flow import build_followup_query
from quick_auditory_learning.session_selection import latest_event_payload, restore_next_paper_id, sort_search_modes
from quick_auditory_learning.embeddings import EmbeddingBatchResult
from quick_auditory_learning.voice import RandomVoiceVoxSpeaker, build_voicevox_speaker, split_text


def load_backend_main(monkeypatch):
    from quick_auditory_learning import logging_config as logging_config_module

    monkeypatch.setattr(logging_config_module, "configure_logging", lambda log_dir: log_dir / "backend.log")
    main = importlib.import_module("quick_auditory_learning.main")
    monkeypatch.setattr(main, "bootstrap_background_jobs", lambda: None)
    monkeypatch.setattr(main.database_ready_event, "set", lambda: None)
    return main


def test_normalize_identifier() -> None:
    assert normalize_identifier("text-embedding-3-small") == "text_embedding_3_small"


def test_model_table_name_changes_with_version() -> None:
    table_a = model_table_name("text-embedding-3-small", "v1")
    table_b = model_table_name("text-embedding-3-small", "v2")
    assert table_a != table_b
    assert table_a.startswith("embedding_text_embedding_3_small_v1_")


def test_normalize_paper_record_preserves_raw_aliases() -> None:
    payload = {
        "id": "0704.0001",
        "title": "Example",
        "abstract": "Abstract",
        "categories": ["hep-ph"],
        "versions": ["v1", "v2"],
        "journal-ref": "Journal",
        "report-no": "R-1",
    }
    normalized = normalize_paper_record(payload)
    assert normalized["journal_ref"] == "Journal"
    assert normalized["report_no"] == "R-1"
    assert json.loads(normalized["categories"]) == ["hep-ph"]
    assert json.loads(normalized["versions"]) == ["v1", "v2"]


def test_merge_scores_is_deterministic_with_seed() -> None:
    route1 = {"a": 0.8, "b": 0.4}
    route2 = {"a": 0.6, "c": 0.9}
    first = merge_scores(route1, route2, 0.5, 0.5, seed=7)
    second = merge_scores(route1, route2, 0.5, 0.5, seed=7)
    assert first == second
    assert [item[0] for item in first] == ["a", "c", "b"]


def test_search_papers_falls_back_to_db_random_candidates(monkeypatch) -> None:
    import importlib
    from quick_auditory_learning.models import Paper

    search = importlib.import_module("quick_auditory_learning.search")

    calls = {}

    class FakeModel:
        table_name = "embedding_table"
        model_name = "text-embedding-3-large"
        dimension = 3

    monkeypatch.setattr(search, "count_papers", lambda conn: 1)
    monkeypatch.setattr(search, "search_by_keyword", lambda *args, **kwargs: [])
    monkeypatch.setattr(search, "ensure_search_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(search, "generate_embeddings_for_paper_ids", lambda *args, **kwargs: SimpleNamespace(input_tokens=0, generated_count=0))
    monkeypatch.setattr(search, "list_embedding_models", lambda *args, **kwargs: [])
    monkeypatch.setattr(search, "search_by_vector_for_ids", lambda *args, **kwargs: [])
    monkeypatch.setattr(search, "search_by_vector", lambda *args, **kwargs: [])

    def fake_random_paper_ids(conn, limit, exclude_paper_ids):
        calls["random"] = (limit, tuple(exclude_paper_ids))
        return ["p-random"]

    monkeypatch.setattr(search, "random_paper_ids", fake_random_paper_ids)
    monkeypatch.setattr(search, "get_paper_rows", lambda *args, **kwargs: {"p-random": Paper(id="p-random", title="Random", abstract="Random abstract")})

    result = search.search_papers(
        conn=object(),
        client=object(),
        request=SimpleNamespace(
            query="missing terms",
            model_name="text-embedding-3-large",
            include_old_vectors=False,
            exclude_paper_ids=["p-excluded"],
            limit=5,
            route1_weight=0.55,
            route2_weight=0.45,
            seed=None,
        ),
        query_embedding=[0.1, 0.2, 0.3],
    )

    assert calls["random"] == (10, ("p-excluded",))
    assert len(result.hits) == 1
    assert result.hits[0].paper.id == "p-random"
    assert result.fallback_used is True


def test_search_by_vector_fetches_rows_before_cursor_closes(monkeypatch) -> None:
    import importlib

    search = importlib.import_module("quick_auditory_learning.search")

    class FakeCursor:
        def __init__(self) -> None:
            self.closed = False
            self.executed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.closed = True
            return False

        def execute(self, *args, **kwargs):
            self.executed = True

        def fetchall(self):
            assert self.executed
            assert not self.closed
            return [{"paper_id": "p-1", "score": 0.75}]

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    result = search.search_by_vector(
        FakeConnection(),
        SimpleNamespace(table_name="embedding_table"),
        [0.1, 0.2, 0.3],
        5,
        ["p-excluded"],
    )

    assert result == [("p-1", 0.75)]


def test_weighted_choice_hit_skips_trail_ids_and_uses_rank_weights(monkeypatch) -> None:
    import importlib
    from quick_auditory_learning import logging_config as logging_config_module

    monkeypatch.setattr(logging_config_module, "configure_logging", lambda log_dir: log_dir / "backend.log")
    main = importlib.import_module("quick_auditory_learning.main")

    recorded = {}

    def fake_choices(population, weights, k):
        recorded["population"] = population
        recorded["weights"] = weights
        recorded["k"] = k
        return [population[-1]]

    monkeypatch.setattr("quick_auditory_learning.main.random.choices", fake_choices)
    hits = [
        {"paper": {"id": "p-1", "title": "One"}},
        {"paper": {"id": "p-2", "title": "Two"}},
        {"paper": {"id": "p-3", "title": "Three"}},
    ]

    paper_id, paper = main.weighted_choice_hit(hits, {"p-2"})

    assert paper_id == "p-3"
    assert paper["id"] == "p-3"
    assert recorded["population"] == [
        {"id": "p-1", "title": "One"},
        {"id": "p-3", "title": "Three"},
    ]
    assert recorded["weights"] == [1.0, 1.0 / 3.0]
    assert recorded["k"] == 1


def test_sort_search_modes_orders_by_priority_and_deduplicates() -> None:
    assert sort_search_modes(["fulltext_query", "simple", "keyword_list", "simple", "unknown"]) == [
        "simple",
        "keyword_list",
        "fulltext_query",
        "unknown",
    ]


def test_latest_event_payload_returns_most_recent_matching_payload() -> None:
    events = [
        SimpleNamespace(event_type="paper_ready", payload={"paper": {"id": "p-1"}}),
        SimpleNamespace(event_type="session_queued", payload={"next_paper_id": "p-2"}),
        SimpleNamespace(event_type="session_next_requested", payload={"to_paper_id": "p-3"}),
        SimpleNamespace(event_type="paper_ready", payload={"paper": {"id": "p-3"}}),
    ]

    assert latest_event_payload(events, "paper_ready") == {"paper": {"id": "p-3"}}
    assert latest_event_payload(events, "session_queued") == {"next_paper_id": "p-2"}
    assert latest_event_payload(events, "session_next_requested") == {"to_paper_id": "p-3"}
    assert latest_event_payload(events, "missing") is None


def test_session_requested_at_by_paper_id_uses_next_requested_event(monkeypatch) -> None:
    import importlib

    db = importlib.import_module("quick_auditory_learning.db")

    session = SimpleNamespace(
        root_paper_id="p-root",
        started_at=datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc),
    )

    events = [
        SimpleNamespace(
            event_type="session_next_requested",
            payload={"from_paper_id": "p-root", "to_paper_id": "p-next"},
            created_at=datetime(2026, 4, 10, 0, 0, 1, tzinfo=timezone.utc),
        ),
        SimpleNamespace(
            event_type="session_regenerated",
            payload={"paper_id": "p-next"},
            created_at=datetime(2026, 4, 10, 0, 0, 2, tzinfo=timezone.utc),
        ),
    ]

    monkeypatch.setattr(db, "get_playback_session", lambda *args, **kwargs: session)
    monkeypatch.setattr(db, "list_session_events", lambda *args, **kwargs: events)

    requested_at = db.session_requested_at_by_paper_id(object(), "session-1")

    assert requested_at["p-root"] == datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)
    assert requested_at["p-next"] == datetime(2026, 4, 10, 0, 0, 1, tzinfo=timezone.utc)


def test_restore_next_paper_id_uses_last_paper_ready_event(monkeypatch) -> None:
    import importlib
    from quick_auditory_learning import logging_config as logging_config_module

    monkeypatch.setattr(logging_config_module, "configure_logging", lambda log_dir: log_dir / "backend.log")
    main = importlib.import_module("quick_auditory_learning.session_selection")

    calls = {}

    def fake_list_session_events(conn, session_id):
        calls["list_session_events"] = (conn, session_id)
        return [
            SimpleNamespace(event_type="session_started", payload={"session_id": session_id}),
            SimpleNamespace(
                event_type="paper_ready",
                payload={
                    "search": {"hits": [{"paper": {"id": "p-1"}}, {"paper": {"id": "p-2"}}]},
                    "trail_paper_ids": ["p-root"],
                },
            ),
        ]

    monkeypatch.setattr(main, "list_session_events", fake_list_session_events)
    monkeypatch.setattr(main, "weighted_choice_hit", lambda hits, trail_ids: ("p-2", {"id": "p-2"}))

    assert restore_next_paper_id(object(), "session-1") == "p-2"
    assert calls["list_session_events"][1] == "session-1"


def test_advance_session_uses_existing_next_paper_id(monkeypatch, tmp_path) -> None:
    import importlib
    from quick_auditory_learning import logging_config as logging_config_module

    monkeypatch.setattr(logging_config_module, "configure_logging", lambda log_dir: log_dir / "backend.log")
    main = importlib.import_module("quick_auditory_learning.main")

    calls = {}

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_connection():
        return FakeConnection()

    def fake_get_playback_session(conn, session_id):
        return SimpleNamespace(
            session_id=session_id,
            current_paper_id="p-current",
            next_paper_id="p-next",
            config={"model_name": "text-embedding-3-large", "include_old_vectors": False, "limit": 5, "route1_weight": 0.55, "route2_weight": 0.45, "seed": None, "search_modes": []},
        )

    def fake_get_paper(conn, paper_id):
        return SimpleNamespace(id=paper_id, title=f"Title {paper_id}", abstract=f"Abstract {paper_id}")

    def fake_pop_session_queue_item(conn, session_id):
        calls["pop_session_queue_item"] = (session_id,)
        return None

    def fake_set_session_next_paper_id(conn, session_id, paper_id):
        calls["set_session_next_paper_id"] = (session_id, paper_id)

    def fake_append_session_trail_item(conn, session_id, paper_id):
        calls.setdefault("append_session_trail_item", []).append((session_id, paper_id))

    def fake_update_playback_session(conn, session_id, **kwargs):
        calls["update_playback_session"] = (session_id, kwargs)

    def fake_record_transition(conn, from_paper_id, to_paper_id):
        calls["record_transition"] = (from_paper_id, to_paper_id)

    def fake_list_session_trail_paper_ids(conn, session_id):
        return ["p-root", "p-current"]

    def fake_paper_ready_payload(conn, client, session_id, paper, **kwargs):
        calls["paper_ready_payload"] = (session_id, paper.id, kwargs["origin"], kwargs["from_paper_id"], kwargs["trail_paper_ids"])
        return {"type": "paper_ready", "session_id": session_id, "paper": {"id": paper.id}}

    def fake_append_session_event_message(conn, session_id, event_type, payload):
        return {"session_id": session_id, "type": event_type, **payload}

    monkeypatch.setattr(main, "connection", fake_connection)
    monkeypatch.setattr(main, "require_openai_client", lambda operation: object())
    monkeypatch.setattr(main, "get_playback_session", fake_get_playback_session)
    monkeypatch.setattr(main, "get_paper", fake_get_paper)
    monkeypatch.setattr(main, "pop_session_queue_item", fake_pop_session_queue_item)
    monkeypatch.setattr(main, "_set_session_next_paper_id", fake_set_session_next_paper_id)
    monkeypatch.setattr(main, "append_session_trail_item", fake_append_session_trail_item)
    monkeypatch.setattr(main, "update_playback_session", fake_update_playback_session)
    monkeypatch.setattr(main, "record_transition", fake_record_transition)
    monkeypatch.setattr(main, "list_session_trail_paper_ids", fake_list_session_trail_paper_ids)
    monkeypatch.setattr(main, "_paper_ready_payload", fake_paper_ready_payload)
    monkeypatch.setattr(main, "_append_session_event_message", fake_append_session_event_message)

    events = main._advance_session(SimpleNamespace(session_id="session-1"))

    assert calls["pop_session_queue_item"] == ("session-1",)
    assert calls["update_playback_session"] == ("session-1", {"current_paper_id": "p-next"})
    assert calls["record_transition"] == ("p-current", "p-next")
    assert calls["paper_ready_payload"] == ("session-1", "p-next", "search", "p-current", ["p-root", "p-current"])
    assert [event["type"] for event in events] == ["session_next_requested", "session_advanced", "paper_ready"]
    assert events[0]["from_paper_id"] == "p-current"
    assert events[0]["to_paper_id"] == "p-next"


def test_generate_explanation_does_not_fetch_arxiv_for_missing_paper_id(monkeypatch) -> None:
    import importlib
    from quick_auditory_learning import logging_config as logging_config_module

    monkeypatch.setattr(logging_config_module, "configure_logging", lambda log_dir: log_dir / "backend.log")
    main = importlib.import_module("quick_auditory_learning.main")

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_connection():
        return FakeConnection()

    def fake_get_paper(conn, paper_id):
        return None

    def fake_get_explanation(conn, paper_id):
        return None

    monkeypatch.setattr(main, "connection", fake_connection)
    monkeypatch.setattr(main, "get_paper", fake_get_paper)
    monkeypatch.setattr(main, "get_explanation", fake_get_explanation)
    monkeypatch.setattr(main, "_ensure_explanation_audio", lambda *args, **kwargs: None)

    response = main.app.router.routes  # keep import side effects exercised
    assert response is not None

    try:
        main.generate_explanation("cond-mat/0104435")
    except main.HTTPException as exc:
        assert exc.status_code == 404
        assert exc.detail == "paper not found"
    else:
        raise AssertionError("expected HTTPException")


def test_split_text_prefers_separator_boundary() -> None:
    chunks = split_text("一文目。二文目。三文目。", 6, ["。"])
    assert chunks == ["一文目。", "二文目。", "三文目。"]


def test_random_voicevox_speaker_uses_random_candidate(monkeypatch) -> None:
    monkeypatch.setattr("quick_auditory_learning.voice._resolve_random_speaker_ids", lambda url: ("1", "2", "3"))
    monkeypatch.setattr("quick_auditory_learning.voice.random.choice", lambda items: items[-1])
    speaker = RandomVoiceVoxSpeaker(url="http://voicevox:50021", fallback_speaker_id="99")
    assert speaker._choose_speaker_id() == "3"


def test_random_voicevox_speaker_falls_back_when_no_candidates(monkeypatch) -> None:
    monkeypatch.setattr("quick_auditory_learning.voice._resolve_random_speaker_ids", lambda url: ())
    speaker = RandomVoiceVoxSpeaker(url="http://voicevox:50021", fallback_speaker_id="99")
    assert speaker._choose_speaker_id() == "99"


def test_build_voicevox_speaker_is_deterministic_per_key(monkeypatch) -> None:
    monkeypatch.setattr("quick_auditory_learning.voice._resolve_random_speaker_ids", lambda url: ("1", "2", "3"))
    first = build_voicevox_speaker(url="http://voicevox:50021", fallback_speaker_id="99", key="paper-a")
    second = build_voicevox_speaker(url="http://voicevox:50021", fallback_speaker_id="99", key="paper-a")
    third = build_voicevox_speaker(url="http://voicevox:50021", fallback_speaker_id="99", key="paper-b")
    assert first.speaker_id == second.speaker_id
    assert first.speaker_id in {"1", "2", "3"}
    assert third.speaker_id in {"1", "2", "3"}


def test_generate_embeddings_for_paper_ids_returns_batch_result_when_all_exist(monkeypatch) -> None:
    from quick_auditory_learning import embeddings as embeddings_module

    class FakeCursor:
        def __init__(self):
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params=None):
            self.executed.append((query, params))

        def fetchall(self):
            return [{"paper_id": "p-1"}]

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(embeddings_module, "embed_text", lambda client, model, text: None)
    result = embeddings_module.generate_embeddings_for_paper_ids(
        FakeConnection(),
        object(),
        SimpleNamespace(table_name="embedding_table"),
        ["p-1"],
    )
    assert isinstance(result, EmbeddingBatchResult)
    assert result.generated_count == 0


def test_cost_estimators_return_positive_values() -> None:
    embedding_cost = estimate_embedding_cost_usd("text-embedding-3-large", 1000)
    completion_cost = estimate_completion_cost_usd("gpt-5-mini", 1200, 300)
    assert embedding_cost > 0
    assert completion_cost > 0


def test_generation_cost_wall_elapsed_ms_from_rows_merges_overlaps() -> None:
    rows = [
        {"created_at": datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc), "elapsed_ms": 100},
        {"created_at": datetime(2026, 4, 10, 0, 0, 0, 50000, tzinfo=timezone.utc), "elapsed_ms": 100},
        {"created_at": datetime(2026, 4, 10, 0, 0, 0, 160000, tzinfo=timezone.utc), "elapsed_ms": 40},
    ]

    assert generation_cost_wall_elapsed_ms_from_rows(rows) == 190


def test_generation_cost_total_cost_usd_from_rows_sums_all_rows() -> None:
    rows = [
        {"estimated_cost_usd": 0.1},
        {"estimated_cost_usd": 0.2},
        {"estimated_cost_usd": 9.99},
    ]

    assert generation_cost_total_cost_usd_from_rows(rows) == 10.29


def test_generation_cost_items_from_rows_marks_missing_rows_pending() -> None:
    rows = [
        {
            "kind": "search",
            "paper_id": "p-1",
            "created_at": datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc),
            "elapsed_ms": 12,
            "estimated_cost_usd": 0.1,
            "detail": {},
        },
        {
            "kind": "search",
            "paper_id": "p-1",
            "created_at": datetime(2026, 4, 10, 0, 0, 0, 20000, tzinfo=timezone.utc),
            "elapsed_ms": 8,
            "estimated_cost_usd": 0.05,
            "detail": {},
        },
        {
            "kind": "audio",
            "paper_id": "p-2",
            "created_at": datetime(2026, 4, 10, 0, 0, 1, tzinfo=timezone.utc),
            "elapsed_ms": 34,
            "estimated_cost_usd": 0.2,
            "detail": {},
        },
    ]

    items = generation_cost_items_from_rows(
        rows,
        requested_at_by_paper_id={"p-1": datetime(2026, 4, 10, 0, 0, 0, 10000, tzinfo=timezone.utc)},
    )
    by_kind = {item.kind: item for item in items}

    assert by_kind["search"].status == "calculated"
    assert by_kind["search"].elapsed_ms == 20
    assert by_kind["search"].elapsed_ms_without_prefetch == 10
    assert by_kind["search"].estimated_cost_usd == 0.15
    assert by_kind["embedding"].status == "pending"
    assert by_kind["embedding"].elapsed_ms is None
    assert by_kind["embedding"].elapsed_ms_without_prefetch is None
    assert by_kind["embedding"].estimated_cost_usd is None


def test_generation_cost_items_from_rows_excludes_prefetch_from_subtotal() -> None:
    rows = [
        {
            "kind": "explanation",
            "paper_id": "p-1",
            "created_at": datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc),
            "elapsed_ms": 100,
            "estimated_cost_usd": 0.4,
            "detail": {},
        },
        {
            "kind": "explanation",
            "paper_id": "p-1",
            "created_at": datetime(2026, 4, 10, 0, 0, 0, 50000, tzinfo=timezone.utc),
            "elapsed_ms": 25,
            "estimated_cost_usd": 0.05,
            "detail": {},
        },
    ]

    items = generation_cost_items_from_rows(
        rows,
        requested_at_by_paper_id={"p-1": datetime(2026, 4, 10, 0, 0, 0, 20000, tzinfo=timezone.utc)},
    )
    explanation = {item.kind: item for item in items}["explanation"]
    assert explanation.elapsed_ms == 125
    assert explanation.elapsed_ms_without_prefetch == 105


def test_generation_cost_items_from_rows_zeroes_finished_before_request() -> None:
    rows = [
        {
            "kind": "audio",
            "paper_id": "p-1",
            "started_at": datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc),
            "finished_at": datetime(2026, 4, 10, 0, 0, 0, 50000, tzinfo=timezone.utc),
            "created_at": datetime(2026, 4, 10, 0, 0, 0, 50000, tzinfo=timezone.utc),
            "elapsed_ms": 50,
            "estimated_cost_usd": 0.05,
            "detail": {},
        }
    ]

    items = generation_cost_items_from_rows(
        rows,
        requested_at_by_paper_id={"p-1": datetime(2026, 4, 10, 0, 0, 0, 100000, tzinfo=timezone.utc)},
    )
    audio = {item.kind: item for item in items}["audio"]

    assert audio.elapsed_ms == 50
    assert audio.elapsed_ms_without_prefetch == 0
    assert audio.estimated_cost_usd == 0.05


def test_generation_cost_items_from_rows_counts_from_request_until_finish() -> None:
    rows = [
        {
            "kind": "audio",
            "paper_id": "p-1",
            "started_at": datetime(2026, 4, 10, 0, 0, 0, 50000, tzinfo=timezone.utc),
            "finished_at": datetime(2026, 4, 10, 0, 0, 0, 100000, tzinfo=timezone.utc),
            "created_at": datetime(2026, 4, 10, 0, 0, 0, 100000, tzinfo=timezone.utc),
            "elapsed_ms": 50,
            "estimated_cost_usd": 0.05,
            "detail": {},
        }
    ]

    items = generation_cost_items_from_rows(
        rows,
        requested_at_by_paper_id={"p-1": datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)},
    )
    audio = {item.kind: item for item in items}["audio"]

    assert audio.elapsed_ms == 50
    assert audio.elapsed_ms_without_prefetch == 100
    assert audio.estimated_cost_usd == 0.05


def test_generation_cost_items_from_rows_can_treat_missing_kinds_as_zero() -> None:
    rows = [
        {
            "kind": "audio",
            "paper_id": "p-1",
            "started_at": datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc),
            "finished_at": datetime(2026, 4, 10, 0, 0, 0, 50000, tzinfo=timezone.utc),
            "created_at": datetime(2026, 4, 10, 0, 0, 0, 50000, tzinfo=timezone.utc),
            "elapsed_ms": 50,
            "estimated_cost_usd": 0.05,
            "detail": {},
        }
    ]

    items = generation_cost_items_from_rows(rows, missing_as_zero=True)
    by_kind = {item.kind: item for item in items}

    assert by_kind["audio"].status == "calculated"
    assert by_kind["audio"].elapsed_ms == 50
    assert by_kind["search"].status == "calculated"
    assert by_kind["search"].elapsed_ms == 0
    assert by_kind["search"].elapsed_ms_without_prefetch == 0
    assert by_kind["search"].estimated_cost_usd == 0.0


def test_generation_cost_wall_elapsed_ms_from_rows_skips_intervals_finished_before_request() -> None:
    rows = [
        {
            "kind": "audio",
            "paper_id": "p-1",
            "started_at": datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc),
            "finished_at": datetime(2026, 4, 10, 0, 0, 0, 50000, tzinfo=timezone.utc),
            "created_at": datetime(2026, 4, 10, 0, 0, 0, 50000, tzinfo=timezone.utc),
            "elapsed_ms": 50,
            "estimated_cost_usd": 0.05,
            "detail": {},
        },
        {
            "kind": "audio",
            "paper_id": "p-1",
            "started_at": datetime(2026, 4, 10, 0, 0, 0, 150000, tzinfo=timezone.utc),
            "finished_at": datetime(2026, 4, 10, 0, 0, 0, 200000, tzinfo=timezone.utc),
            "created_at": datetime(2026, 4, 10, 0, 0, 0, 200000, tzinfo=timezone.utc),
            "elapsed_ms": 50,
            "estimated_cost_usd": 0.05,
            "detail": {},
        },
    ]

    elapsed_ms = generation_cost_wall_elapsed_ms_from_rows(
        rows,
        requested_at_by_paper_id={"p-1": datetime(2026, 4, 10, 0, 0, 0, 100000, tzinfo=timezone.utc)},
    )

    assert elapsed_ms == 50


def test_generation_cost_wall_elapsed_ms_from_rows_counts_from_request_until_finish() -> None:
    rows = [
        {
            "kind": "audio",
            "paper_id": "p-1",
            "started_at": datetime(2026, 4, 10, 0, 0, 0, 50000, tzinfo=timezone.utc),
            "finished_at": datetime(2026, 4, 10, 0, 0, 0, 200000, tzinfo=timezone.utc),
            "created_at": datetime(2026, 4, 10, 0, 0, 0, 200000, tzinfo=timezone.utc),
            "elapsed_ms": 150,
            "estimated_cost_usd": 0.05,
            "detail": {},
        }
    ]

    elapsed_ms = generation_cost_wall_elapsed_ms_from_rows(
        rows,
        requested_at_by_paper_id={"p-1": datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)},
    )

    assert elapsed_ms == 200


def test_generation_cost_rows_can_exclude_prefetch(monkeypatch) -> None:
    import importlib

    db = importlib.import_module("quick_auditory_learning.db")

    class FakeCursor:
        def __init__(self):
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            self.executed.append((query, params))

        def fetchall(self):
            return [
                {
                    "kind": "search",
                    "created_at": datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc),
                    "elapsed_ms": 10,
                    "estimated_cost_usd": 0.1,
                    "detail": {},
                },
                {
                    "kind": "explanation",
                    "created_at": datetime(2026, 4, 10, 0, 0, 1, tzinfo=timezone.utc),
                    "elapsed_ms": 20,
                    "estimated_cost_usd": 0.2,
                    "detail": {"generation_scope": "prefetch"},
                },
            ]

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    rows = db.generation_cost_rows(FakeConnection(), "s-1", include_prefetch=False)
    assert [row["kind"] for row in rows] == ["search"]


def test_generation_cost_rows_selects_paper_id_for_request_time_grouping() -> None:
    import importlib

    db = importlib.import_module("quick_auditory_learning.db")

    class FakeCursor:
        def __init__(self):
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            self.executed.append((query, params))

        def fetchall(self):
            return []

    class FakeConnection:
        def __init__(self):
            self.cursor_obj = FakeCursor()

        def cursor(self):
            return self.cursor_obj

    conn = FakeConnection()
    db.generation_cost_rows(conn, "s-1")

    assert "paper_id" in conn.cursor_obj.executed[0][0]


def test_get_session_generation_costs_uses_wall_elapsed_ms_and_includes_prefetch_cost(monkeypatch) -> None:
    import importlib

    db = importlib.import_module("quick_auditory_learning.db")

    class FakeCursor:
        def __init__(self, row):
            self.row = row
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            self.executed.append((query, params))
            self.query = query

        def fetchone(self):
            return self.row

        def fetchall(self):
            return [
                {"kind": "search", "created_at": datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc), "elapsed_ms": 100, "estimated_cost_usd": 0.1},
                {"kind": "embedding", "created_at": datetime(2026, 4, 10, 0, 0, 0, 50000, tzinfo=timezone.utc), "elapsed_ms": 100, "estimated_cost_usd": 0.2},
                {"kind": "explanation", "created_at": datetime(2026, 4, 10, 0, 0, 0, 200000, tzinfo=timezone.utc), "elapsed_ms": 50, "estimated_cost_usd": 0.3},
            ]

    class FakeConnection:
        def __init__(self, row):
            self.cursor_obj = FakeCursor(row)

        def cursor(self):
            return self.cursor_obj

    row = {
        "session_id": "s-1",
        "search_elapsed_ms": 10,
        "search_cost_usd": 0.1,
        "embedding_elapsed_ms": 20,
        "embedding_cost_usd": 0.2,
        "explanation_elapsed_ms": 30,
        "explanation_cost_usd": 0.3,
        "audio_elapsed_ms": 40,
        "audio_cost_usd": 0.4,
        "keyword_generation_elapsed_ms": 50,
        "keyword_generation_cost_usd": 0.5,
        "query_generation_elapsed_ms": 60,
        "query_generation_cost_usd": 0.6,
        "prefetch_elapsed_ms": 999,
        "prefetch_cost_usd": 9.99,
        "total_elapsed_ms": 0,
        "total_wall_elapsed_ms": 321,
        "total_cost_usd": 0,
        "updated_at": "2026-04-10T00:00:00Z",
    }

    monkeypatch.setattr(db, "generation_cost_wall_elapsed_ms", lambda *args, **kwargs: 250)
    monkeypatch.setattr(db, "generation_cost_total_cost_usd", lambda *args, **kwargs: 12.09)
    summary = db.get_session_generation_costs(FakeConnection(row), "s-1")

    assert summary is not None
    assert summary.total_elapsed_ms == 250
    assert summary.total_cost_usd == 12.09


def test_session_cost_payload_keeps_current_totals_even_when_active(monkeypatch, tmp_path) -> None:
    from quick_auditory_learning import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "log_dir", tmp_path / "logs")
    main = load_backend_main(monkeypatch)

    monkeypatch.setattr(main, "get_session_generation_costs", lambda *args, **kwargs: SimpleNamespace(session_id="s-1", total_elapsed_ms=333, total_wall_elapsed_ms=444, total_cost_usd=1.23))
    monkeypatch.setattr(
        main,
        "generation_cost_rows",
        lambda *args, **kwargs: [
            {
                "kind": "search",
                "paper_id": "p-1",
                "created_at": datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc),
                "elapsed_ms": 100,
                "estimated_cost_usd": 0.1,
                "detail": {},
            },
            {
                "kind": "search",
                "paper_id": "p-1",
                "created_at": datetime(2026, 4, 10, 0, 0, 0, 50000, tzinfo=timezone.utc),
                "elapsed_ms": 20,
                "estimated_cost_usd": 0.02,
                "detail": {},
            },
        ],
    )
    monkeypatch.setattr(main, "generation_cost_items_from_rows", generation_cost_items_from_rows)
    monkeypatch.setattr(main, "_session_audio_duration_ms", lambda *args, **kwargs: 900)
    monkeypatch.setattr(main, "get_playback_session", lambda *args, **kwargs: SimpleNamespace(status="active"))
    monkeypatch.setattr(main, "session_requested_at_by_paper_id", lambda *args, **kwargs: {"p-1": datetime(2026, 4, 10, 0, 0, 0, 20000, tzinfo=timezone.utc)})
    monkeypatch.setattr(
        main,
        "generation_cost_wall_elapsed_ms",
        lambda *args, **kwargs: 222 if kwargs.get("requested_at_by_paper_id") is None else 111,
    )
    monkeypatch.setattr(
        main,
        "generation_cost_total_cost_usd",
        lambda *args, **kwargs: 1.23 if kwargs.get("requested_at_by_paper_id") is None else 1.11,
    )

    payload = main._session_cost_payload(object(), "s-1")

    assert payload is not None
    assert payload.total_elapsed_ms == 333
    assert payload.total_cost_usd == 1.23
    assert payload.total_elapsed_ms_without_prefetch == 111
    assert payload.total_cost_usd_without_prefetch == 1.11
    assert payload.items[0].elapsed_ms_without_prefetch == 100


def test_generate_explanation_records_zero_audio_cost_on_audio_cache_hit(monkeypatch, tmp_path: Path) -> None:
    main = load_backend_main(monkeypatch)

    class FakeConnection:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(main, "connection", lambda: FakeConnection())
    monkeypatch.setattr(
        main,
        "get_paper",
        lambda conn, paper_id: SimpleNamespace(id=paper_id, title="Title", abstract="Abstract"),
    )
    monkeypatch.setattr(main, "get_explanation", lambda conn, paper_id: "cached explanation")
    monkeypatch.setattr(main, "explanation_audio_path", lambda paper_id: tmp_path / f"{paper_id}.wav")
    monkeypatch.setattr(main, "_read_explanation_audio_speaker_id", lambda paper_id: "speaker-1")
    monkeypatch.setattr(main, "build_voicevox_speaker", lambda **kwargs: SimpleNamespace(speaker_id="speaker-1"))
    monkeypatch.setattr(main, "explanation_audio_chunk_texts", lambda explanation: ["chunk"])
    monkeypatch.setattr(main, "explanation_audio_chunk_url", lambda paper_id, index: f"/audio/{paper_id}/chunks/{index:04d}")
    monkeypatch.setattr(main, "_wav_duration_ms", lambda path: 1000)

    audio_path = tmp_path / "p-1.wav"
    audio_path.write_bytes(b"RIFF")

    calls = []

    response = main.generate_explanation(
        "p-1",
        cost_recorder=lambda kind, started_at, finished_at, elapsed_ms, estimated_cost_usd, detail: calls.append(
            (kind, elapsed_ms, estimated_cost_usd, detail)
        ),
    )

    assert response.paper_id == "p-1"
    assert [call[0] for call in calls] == ["explanation", "audio"]
    assert calls[1][1] == 0
    assert calls[1][2] == 0.0
    assert calls[1][3]["cache_hit"] is True


def test_schedule_next_paper_prefetch_records_generation_cost(monkeypatch) -> None:
    from quick_auditory_learning import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "log_dir", Path("/tmp") / "quick-auditory-learning-logs")
    main = load_backend_main(monkeypatch)

    calls = {}

    class FakeConnection:
        def __enter__(self):
            calls["connection_entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            calls["connection_exited"] = True
            return False

    class FakeExecutor:
        def submit(self, fn):
            calls["executor_submit"] = True
            fn()

    monkeypatch.setattr(main, "connection", lambda: FakeConnection())
    monkeypatch.setattr(main, "PREFETCH_EXECUTOR", FakeExecutor())
    monkeypatch.setattr(main, "_prefetch_target_is_current", lambda session_id, paper_id: True)
    monkeypatch.setattr(main, "_record_generation_cost_and_notify", lambda *args, **kwargs: calls.setdefault("cost_calls", []).append((args, kwargs)))
    monkeypatch.setattr(
        main,
        "generate_explanation",
        lambda paper_id, should_continue=None, cost_recorder=None: (
            cost_recorder("explanation", datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc), datetime(2026, 4, 10, 0, 0, 0, 100000, tzinfo=timezone.utc), 100, 0.5, {"paper_id": paper_id}) if cost_recorder else None,
            cost_recorder("audio", datetime(2026, 4, 10, 0, 0, 0, 100000, tzinfo=timezone.utc), datetime(2026, 4, 10, 0, 0, 0, 150000, tzinfo=timezone.utc), 50, 0.0, {"paper_id": paper_id}) if cost_recorder else None,
        ),
    )

    main._schedule_next_paper_prefetch("session-1", "p-next")

    assert calls["executor_submit"] is True
    assert calls["connection_entered"] is True
    assert calls["connection_exited"] is True
    assert len(calls["cost_calls"]) == 2


def test_parse_arxiv_identifier_accepts_abs_and_pdf_urls() -> None:
    assert parse_arxiv_identifier("https://arxiv.org/abs/0704.0001v2") == ("0704.0001v2", "0704.0001")
    assert parse_arxiv_identifier("https://arxiv.org/pdf/0704.0001v2.pdf") == ("0704.0001v2", "0704.0001")


def test_strip_arxiv_version_removes_suffix() -> None:
    assert strip_arxiv_version("0704.0001v2") == "0704.0001"
    assert strip_arxiv_version("hep-th/9911001v3") == "hep-th/9911001"


def test_jsonl_import_is_stale() -> None:
    class State:
        source_mtime_ns = 10
        source_size = 20

    assert jsonl_import_is_stale(None, 1, 2) is True
    assert jsonl_import_is_stale(State(), 10, 20) is False
    assert jsonl_import_is_stale(State(), 11, 20) is True


def test_build_followup_query_uses_distinct_tokens() -> None:
    query = build_followup_query(
        "Transformer-Based Approach for Graph Attention",
        "This approach studies graph attention with transformer-based models for data analysis.",
    )
    tokens = query.split()
    assert "transformer" in tokens
    assert "graph" in tokens
    assert "attention" in tokens
    assert len(query.split()) <= 8


def test_paper_ready_payload_keeps_search_and_force_flag(monkeypatch, tmp_path) -> None:
    from quick_auditory_learning import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "log_dir", tmp_path / "logs")
    from quick_auditory_learning import main

    captured = {}

    def fake_build_followup_query(title: str, abstract: str) -> str:
        return f"{title} / {abstract}"

    def fake_generate_search_keyword(client, model_name: str, title: str, abstract: str):
        captured["generate_search_keyword"] = (client, model_name, title, abstract)
        return SimpleNamespace(search_keyword="llm keyword", elapsed_ms=123, input_tokens=8, output_tokens=4)

    def fake_generate_fulltext_query(client, model_name: str, title: str, abstract: str):
        captured["generate_fulltext_query"] = (client, model_name, title, abstract)
        return SimpleNamespace(search_query='llm fulltext', elapsed_ms=234, input_tokens=9, output_tokens=5)

    def fake_embed_text(client, model_name: str, query: str) -> list[float]:
        captured["embed"] = (client, model_name, query)
        return SimpleNamespace(embedding=[0.1], input_tokens=10)

    def fake_search_papers(conn, client, request, query_embedding, cost_recorder=None):
        captured.setdefault("search_requests", []).append(request.query)
        captured.setdefault("query_embeddings", []).append(query_embedding)
        if cost_recorder is not None:
            started_at = datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)
            finished_at = datetime(2026, 4, 10, 0, 0, 1, tzinfo=timezone.utc)
            cost_recorder("search", started_at, finished_at, 1000, 0.01, {"query": request.query})

        class FakeSearchResponse:
            def __init__(self, query: str) -> None:
                self.query = query

            def model_dump(self, mode: str) -> dict[str, object]:
                assert mode == "json"
                if self.query == "llm fulltext":
                    return {
                        "hits": [{"paper": {"id": "p-fulltext", "title": "Fulltext paper"}, "score": 0.95, "route1_score": 0.5, "route2_score": 0.4}],
                        "rejected_candidates": [{"paper_id": "p-fulltext-reject", "title": "Fulltext rejected", "score": 0.3}],
                        "fallback_used": False,
                    }
                if self.query == "llm keyword":
                    return {
                        "hits": [{"paper": {"id": "p-llm", "title": "LLM paper"}, "score": 0.9, "route1_score": 0.5, "route2_score": 0.4}],
                        "rejected_candidates": [{"paper_id": "p-llm-reject", "title": "LLM rejected", "score": 0.25}],
                        "fallback_used": False,
                    }
                return {
                    "hits": [{"paper": {"id": "p-next", "title": "Next paper"}, "score": 0.8, "route1_score": 0.6, "route2_score": 0.2}],
                    "rejected_candidates": [{"paper_id": "p-reject", "title": "Rejected", "score": 0.1}],
                    "fallback_used": False,
                }

        return FakeSearchResponse(request.query)

    def fake_generate_explanation(paper_id: str, force: bool = False, *, cost_recorder=None, should_continue=None, notice_recorder=None):
        captured["generate_explanation"] = (paper_id, force)
        return SimpleNamespace(
            explanation="generated explanation",
            audio_url="/audio/paper",
            audio_urls=["/audio/paper"],
            audio_duration_ms=1234,
            notices=[],
        )

    def fake_record_generation_cost(*args, **kwargs):
        captured.setdefault("cost_calls", []).append((args, kwargs))

    def fake_get_session_generation_costs(conn, session_id):
        return None

    monkeypatch.setattr(main, "build_followup_query", fake_build_followup_query)
    monkeypatch.setattr(main, "generate_search_keyword", fake_generate_search_keyword)
    monkeypatch.setattr(main, "generate_fulltext_query", fake_generate_fulltext_query)
    monkeypatch.setattr(main, "make_client", lambda api_key: object())
    monkeypatch.setattr(main, "embed_text", fake_embed_text)
    monkeypatch.setattr(main, "search_papers", fake_search_papers)
    monkeypatch.setattr(main, "generate_explanation", fake_generate_explanation)
    monkeypatch.setattr(main, "record_generation_cost", fake_record_generation_cost)
    monkeypatch.setattr(main, "get_session_generation_costs", fake_get_session_generation_costs)
    monkeypatch.setattr(main, "_set_session_next_paper_id", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "weighted_choice_hit", lambda *args, **kwargs: ("p-next", {"id": "p-next"}))
    monkeypatch.setattr(main, "_paper_cost_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "list_session_queue_paper_ids", lambda *args, **kwargs: [])
    monkeypatch.setattr(main, "get_paper_memo", lambda *args, **kwargs: None)

    paper = SimpleNamespace(
        id="p-current",
        title="Current paper",
        abstract="Current abstract",
        model_dump=lambda mode: {"id": "p-current", "title": "Current paper", "abstract": "Current abstract"},
    )

    result = main._paper_ready_payload(
        conn=object(),
        client=object(),
        session_id="session-1",
        paper=paper,
        origin="regenerate",
        from_paper_id="p-current",
        trail_paper_ids=["p-previous", "p-current"],
        config={
            "model_name": "text-embedding-3-large",
            "include_old_vectors": False,
            "limit": 5,
            "route1_weight": 0.55,
            "route2_weight": 0.45,
            "seed": None,
            "search_modes": ["simple", "keyword_list", "fulltext_query"],
        },
        force_explanation=True,
    )

    assert result["origin"] == "regenerate"
    assert result["trail_paper_ids"] == ["p-previous", "p-current"]
    assert [hit["paper"]["id"] for hit in result["search"]["hits"]] == ["p-fulltext", "p-llm", "p-next"]
    assert result["search"]["hits"][0]["source_modes"] == ["fulltext_query"]
    assert result["search"]["hits"][1]["source_modes"] == ["keyword_list"]
    assert result["search"]["hits"][2]["source_modes"] == ["simple"]
    assert result["simple_search_query"] == "Current paper / Current abstract"
    assert result["keyword_search_query"] == "llm keyword"
    assert result["fulltext_search_query"] == "llm fulltext"
    assert captured["generate_explanation"] == ("p-current", True)
    assert captured["search_requests"] == ["Current paper / Current abstract", "llm keyword", "llm fulltext"]
    search_related_paper_ids = [
        kwargs["paper_id"]
        for args, kwargs in captured["cost_calls"]
        if len(args) >= 2 and args[1] in {"keyword_generation", "query_generation", "embedding", "search"}
    ]
    assert search_related_paper_ids
    assert set(search_related_paper_ids) == {"p-next"}


def test_paper_ready_payload_collects_api_failure_notices(monkeypatch, tmp_path) -> None:
    from quick_auditory_learning import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "log_dir", tmp_path / "logs")
    from quick_auditory_learning import main

    def fake_build_followup_query(title: str, abstract: str) -> str:
        return f"{title} / {abstract}"

    def fake_generate_search_keyword(client, model_name: str, title: str, abstract: str):
        raise RuntimeError("Connection error")

    def fake_generate_explanation(paper_id: str, force: bool = False, *, cost_recorder=None, should_continue=None, notice_recorder=None):
        return SimpleNamespace(
            explanation="generated explanation",
            audio_url="/audio/paper",
            audio_urls=["/audio/paper"],
            audio_duration_ms=1234,
            notices=[],
        )

    def fake_get_session_generation_costs(conn, session_id):
        return None

    monkeypatch.setattr(main, "build_followup_query", fake_build_followup_query)
    monkeypatch.setattr(main, "generate_search_keyword", fake_generate_search_keyword)
    monkeypatch.setattr(main, "generate_fulltext_query", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "make_client", lambda api_key: object())
    monkeypatch.setattr(main, "embed_text", lambda *args, **kwargs: SimpleNamespace(embedding=[0.1], input_tokens=10))
    monkeypatch.setattr(
        main,
        "search_papers",
        lambda *args, **kwargs: SimpleNamespace(
            model_dump=lambda mode: {
                "hits": [{"paper": {"id": "p-next", "title": "Next paper"}, "score": 0.8, "route1_score": 0.6, "route2_score": 0.2}],
                "rejected_candidates": [],
                "fallback_used": True,
            }
        ),
    )
    monkeypatch.setattr(main, "generate_explanation", fake_generate_explanation)
    monkeypatch.setattr(main, "record_generation_cost", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "get_session_generation_costs", fake_get_session_generation_costs)
    monkeypatch.setattr(main, "_set_session_next_paper_id", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "_paper_cost_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "list_session_queue_paper_ids", lambda *args, **kwargs: [])
    monkeypatch.setattr(main, "get_paper_memo", lambda *args, **kwargs: None)

    paper = SimpleNamespace(
        id="p-current",
        title="Current paper",
        abstract="Current abstract",
        model_dump=lambda mode: {"id": "p-current", "title": "Current paper", "abstract": "Current abstract"},
    )

    result = main._paper_ready_payload(
        conn=object(),
        client=object(),
        session_id="session-1",
        paper=paper,
        origin="search",
        from_paper_id=None,
        trail_paper_ids=["p-root"],
        config={
            "model_name": "text-embedding-3-large",
            "include_old_vectors": False,
            "limit": 5,
            "route1_weight": 0.55,
            "route2_weight": 0.45,
            "seed": None,
            "search_modes": ["simple", "keyword_list"],
        },
        force_explanation=False,
    )

    assert result["notices"] == ["検索キーワードの生成に失敗しました。API を利用できませんでした。"]


def test_paper_memo_repository_roundtrip() -> None:
    class FakeCursor:
        def __init__(self, row=None):
            self.row = row
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            self.executed.append((query, params))

        def fetchone(self):
            return self.row

    class FakeConnection:
        def __init__(self, row=None):
            self.cursor_obj = FakeCursor(row)

        def cursor(self):
            return self.cursor_obj

    read_conn = FakeConnection({"paper_id": "p-1", "memo": "note", "updated_at": "2026-04-04T00:00:00Z"})
    memo = get_paper_memo(read_conn, "p-1")
    assert memo == {"paper_id": "p-1", "memo": "note", "updated_at": "2026-04-04T00:00:00Z"}
    assert read_conn.cursor_obj.executed[0][1] == ("p-1",)

    write_conn = FakeConnection({"paper_id": "p-1", "memo": "updated", "updated_at": "2026-04-04T00:01:00Z"})
    saved = upsert_paper_memo(write_conn, "p-1", "updated")
    assert saved == {"paper_id": "p-1", "memo": "updated", "updated_at": "2026-04-04T00:01:00Z"}
    assert write_conn.cursor_obj.executed[0][1] == ("p-1", "updated")


def test_paper_memo_route_accepts_slash_in_paper_id(monkeypatch) -> None:
    main = load_backend_main(monkeypatch)
    captured = {}

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(main, "connection", lambda: FakeConnection())

    def fake_get_paper_memo(conn, paper_id):
        captured["paper_id"] = paper_id
        return None

    monkeypatch.setattr(main, "get_paper_memo", fake_get_paper_memo)

    client = TestClient(main.app)
    response = client.get("/papers/cond-mat%2F0104435/memo")

    assert response.status_code == 200
    assert response.json() == {"paper_id": "cond-mat/0104435", "memo": "", "updated_at": None}
    assert captured["paper_id"] == "cond-mat/0104435"


def test_paper_memo_put_route_accepts_slash_in_paper_id(monkeypatch) -> None:
    main = load_backend_main(monkeypatch)
    captured = {}
    
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(main, "connection", lambda: FakeConnection())
    monkeypatch.setattr(main, "_ensure_paper_available", lambda conn, paper_id: None)

    def fake_upsert_paper_memo(conn, paper_id, memo):
        captured["args"] = (paper_id, memo)
        return {"paper_id": paper_id, "memo": memo, "updated_at": "2026-04-10T00:00:00Z"}

    monkeypatch.setattr(main, "upsert_paper_memo", fake_upsert_paper_memo)

    client = TestClient(main.app)
    response = client.put("/papers/cond-mat%2F0104435/memo", json={"memo": "note"})

    assert response.status_code == 200
    assert response.json()["paper_id"] == "cond-mat/0104435"
    assert response.json()["memo"] == "note"
    assert captured["args"] == ("cond-mat/0104435", "note")


def test_paper_memo_websocket_route_accepts_slash_in_paper_id(monkeypatch) -> None:
    main = load_backend_main(monkeypatch)
    monkeypatch.setattr(main, "_load_paper_memo_snapshot", lambda paper_id: {"paper_id": paper_id, "memo": "memo", "updated_at": None})

    client = TestClient(main.app)
    with client.websocket_connect("/papers/cond-mat%2F0104435/memo/ws") as websocket:
        snapshot = websocket.receive_json()
        assert snapshot == {"paper_id": "cond-mat/0104435", "memo": "memo", "updated_at": None}


def test_audio_route_accepts_slash_in_paper_id(monkeypatch, tmp_path) -> None:
    main = load_backend_main(monkeypatch)
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio-bytes")

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(main, "explanation_audio_path", lambda paper_id: audio_path)
    monkeypatch.setattr(main, "connection", lambda: FakeConnection())

    client = TestClient(main.app)
    response = client.get("/audio/cond-mat%2F0104435")

    assert response.status_code == 200
    assert response.content == b"audio-bytes"


def test_audio_chunk_route_accepts_slash_in_paper_id(monkeypatch, tmp_path) -> None:
    main = load_backend_main(monkeypatch)
    audio_path = tmp_path / "chunk.mp3"
    audio_path.write_bytes(b"chunk-bytes")

    monkeypatch.setattr(main, "explanation_audio_chunk_path", lambda paper_id, chunk_index: audio_path)

    client = TestClient(main.app)
    response = client.get("/audio/cond-mat%2F0104435/chunks/0000")

    assert response.status_code == 200
    assert response.content == b"chunk-bytes"


def test_ensure_schema_adds_new_cost_columns() -> None:
    class FakeCursor:
        def __init__(self):
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params=None):
            self.executed.append((query, params))

    class FakeConnection:
        def __init__(self):
            self.cursor_obj = FakeCursor()

        def cursor(self):
            return self.cursor_obj

    conn = FakeConnection()
    ensure_schema(conn)
    queries = [query for query, _ in conn.cursor_obj.executed]
    assert any("ALTER TABLE session_generation_cost_totals" in query and "keyword_generation_elapsed_ms" in query for query in queries)
    assert any("ALTER TABLE session_generation_cost_totals" in query and "query_generation_elapsed_ms" in query for query in queries)
