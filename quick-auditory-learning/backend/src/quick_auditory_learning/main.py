import asyncio
import json
import logging
import random
from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
import shutil
import threading
import time
from time import perf_counter
from uuid import uuid4

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from psycopg import OperationalError
from psycopg.errors import UndefinedTable
from uvicorn.protocols.utils import ClientDisconnected

from quick_auditory_learning.db import (
    append_session_event,
    append_session_trail_item,
    connection,
    create_playback_session,
    get_paper_generation_costs,
    get_session_generation_costs,
    ensure_schema,
    get_explanation,
    get_playback_session,
    generation_cost_items_from_rows,
    generation_cost_total_cost_usd,
    generation_cost_wall_elapsed_ms,
    generation_cost_rows,
    list_session_events,
    list_playback_sessions,
    list_session_trail_paper_ids,
    pop_session_next_candidate,
    record_generation_cost,
    set_session_next_candidate,
    session_requested_at_by_paper_id,
    upsert_explanation,
    update_playback_session,
)
from quick_auditory_learning.arxiv_source import resolve_paper_from_source
from quick_auditory_learning.costs import estimate_completion_cost_usd, estimate_embedding_cost_usd
from quick_auditory_learning.embeddings import embed_text, make_client
from quick_auditory_learning.importer import sync_jsonl
from quick_auditory_learning.models import (
    ExplanationResponse,
    FavoriteToggleResponse,
    FavoriteListResponse,
    FavoritePaperItem,
    HistoryTransition,
    PaperMemoResponse,
    PaperMemoUpdateRequest,
    SessionCostsResponse,
    SessionListItem,
    SessionListResponse,
    SessionClientMessage,
    SessionSnapshot,
    PaperResolveRequest,
    PaperResolveResponse,
    SearchRequest,
)
from quick_auditory_learning.repository import get_paper, list_favorite_items, list_favorites, recent_transitions, record_transition, toggle_favorite
from quick_auditory_learning.repository import get_paper_memo, upsert_paper_memo
from quick_auditory_learning.search import search_papers
from quick_auditory_learning.session_selection import (
    latest_event_payload,
    pick_next_paper_from_search_payload,
    restore_next_paper_id,
    sort_search_modes,
    weighted_choice_hit,
)
from quick_auditory_learning.session_flow import build_followup_query, generate_fulltext_query, generate_search_keyword
from quick_auditory_learning.logging_config import configure_logging
from quick_auditory_learning.settings import settings
from quick_auditory_learning.voice import build_voicevox_speaker, chunk_text, merge_wav_files

log_path = configure_logging(settings.log_dir)
app = FastAPI(title="quick-auditory-learning")
logger = logging.getLogger(__name__)
database_ready_event = threading.Event()
memo_room_queues: dict[str, set[asyncio.Queue[dict[str, object]]]] = defaultdict(set)
memo_room_lock = threading.Lock()
session_room_queues: dict[str, set[tuple[asyncio.Queue[dict[str, object]], asyncio.AbstractEventLoop]]] = defaultdict(set)
session_room_lock = threading.Lock()
session_room_pending_events: dict[str, list[dict[str, object]]] = defaultdict(list)
session_room_pending_lock = threading.Lock()
CostRecorder = Callable[[str, datetime, datetime, int, float, dict[str, object]], None]
PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quick-prefetch")
PREFETCH_LOCK = threading.Lock()
PREFETCH_SESSION_TARGETS: dict[str, str] = {}
PREFETCH_INFLIGHT: set[tuple[str, str]] = set()
SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quick-search")
SEARCH_LOCK = threading.Lock()
SEARCH_SESSION_TARGETS: dict[str, str] = {}
SEARCH_INFLIGHT: set[tuple[str, str]] = set()
SEARCH_PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quick-search-prefetch")
SEARCH_PREFETCH_LOCK = threading.Lock()
SEARCH_PREFETCH_SESSION_TARGETS: dict[str, str] = {}
SEARCH_PREFETCH_INFLIGHT: set[tuple[str, str]] = set()
SEARCH_PREFETCH_CACHE: dict[tuple[str, str], dict[str, object]] = {}


class PrefetchCancelled(RuntimeError):
    pass
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url] if settings.frontend_url else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("logging to %s", log_path)
    logger.info(
        "voicevox configured: url=%s random_speakers=%s fallback_speaker_id=%s",
        settings.voicevox_url,
        ["ずんだもん", "四国めたん", "春日部つむぎ"],
        settings.voicevox_speaker_id,
    )
    thread = threading.Thread(target=bootstrap_background_jobs, daemon=True)
    thread.start()


def bootstrap_background_jobs() -> None:
    wait_for_database_ready()
    if settings.jsonl_path is not None:
        sync_jsonl_in_background(settings.jsonl_path)


def wait_for_database_ready() -> None:
    last_error: Exception | None = None
    for attempt in range(60):
        try:
            with connection() as conn:
                ensure_schema(conn)
            database_ready_event.set()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("database is not ready yet: attempt=%s error=%s", attempt + 1, exc)
            time.sleep(2)
    if last_error is not None:
        raise last_error


def sync_jsonl_in_background(path: Path) -> None:
    if not path.exists():
        logger.warning("jsonl source not found: %s", path)
        return
    logger.info("jsonl sync started: path=%s", path)
    try:
        result = sync_jsonl(path)
    except Exception:
        logger.exception("jsonl sync failed: %s", path)
        return
    if result is not None:
        logger.info("jsonl synced: imported=%s updated=%s path=%s", result.imported, result.updated, path)
    else:
        logger.info("jsonl already up to date: %s", path)


def require_openai_client(operation: str):
    if not settings.openai_api_key:
        logger.warning("openai api key is not configured: operation=%s", operation)
        raise HTTPException(
            status_code=503,
            detail=f"OPENAI_API_KEY が未設定です。{operation} を使うには OPENAI_API_KEY が必要です。",
        )
    return make_client(settings.openai_api_key)


def _ensure_paper_available(conn, paper_id: str):
    paper = get_paper(conn, paper_id)
    if paper is not None:
        return paper
    raise HTTPException(status_code=404, detail="paper not found")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(
        "http exception: method=%s path=%s status=%s detail=%s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(OperationalError)
async def database_unavailable_handler(request: Request, exc: OperationalError) -> JSONResponse:
    logger.warning(
        "database unavailable: method=%s path=%s error=%s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=503,
        content={"detail": "データベースがまだ利用できません。しばらくしてから再試行してください。"},
    )


@app.exception_handler(UndefinedTable)
async def database_schema_missing_handler(request: Request, exc: UndefinedTable) -> JSONResponse:
    logger.warning(
        "database schema missing: method=%s path=%s error=%s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=503,
        content={"detail": "データベースを初期化中です。しばらくしてから再試行してください。"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: BLE001
    logger.exception("unhandled exception: method=%s path=%s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "予期しないエラーが発生しました。backend.log を確認してください。"},
    )


@app.get("/health")
def health() -> dict[str, str | bool | int]:
    return {
        "status": "ok",
        "database_ready": database_ready_event.is_set(),
        "session_websocket_connections": _session_websocket_connection_count(),
    }


@app.get("/config")
def config() -> dict[str, str | bool | None]:
    return {
        "data_dir": str(settings.data_dir),
        "cache_dir": str(settings.cache_dir),
        "postgres_dsn": settings.postgres_dsn,
        "jsonl_path": str(settings.jsonl_path) if settings.jsonl_path is not None else None,
        "embedding_model_name": settings.embedding_model_name,
        "voicevox_url": settings.voicevox_url,
        "voicevox_speaker_id": settings.voicevox_speaker_id,
        "openai_api_key_configured": bool(settings.openai_api_key),
        "jsonl_path_exists": settings.jsonl_path.exists() if settings.jsonl_path is not None else None,
    }


@app.get("/embedding-models")
def embedding_models(model_name: str) -> dict[str, list[dict[str, str | int]]]:
    from quick_auditory_learning.db import list_embedding_models

    with connection() as conn:
        models = list_embedding_models(conn, model_name)
    return {
        "models": [
            {
                "model_name": model.model_name,
                "model_version": model.model_version,
                "dimension": model.dimension,
                "table_name": model.table_name,
            }
            for model in models
        ]
    }


@app.post("/search")
def search(request: SearchRequest):
    client = require_openai_client("search")
    query_embedding = embed_text(client, request.model_name, request.query).embedding
    with connection() as conn:
        try:
            response = search_papers(conn, client, request, query_embedding)
        except ValueError as exc:
            detail = str(exc)
            if detail.startswith("no papers imported"):
                status_code = 404
            else:
                status_code = 500
            raise HTTPException(status_code=status_code, detail=detail) from exc
    return response


@app.post("/favorites/{paper_id:path}/toggle", response_model=FavoriteToggleResponse)
def favorite_toggle(paper_id: str) -> FavoriteToggleResponse:
    with connection() as conn:
        try:
            _ensure_paper_available(conn, paper_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        favorited = toggle_favorite(conn, paper_id)
    return FavoriteToggleResponse(paper_id=paper_id, favorited=favorited)


@app.get("/favorites", response_model=FavoriteListResponse)
def favorites() -> FavoriteListResponse:
    with connection() as conn:
        items = list_favorite_items(conn)
    return FavoriteListResponse(
        paper_ids=[item["paper_id"] for item in items],
        items=[FavoritePaperItem(paper_id=item["paper_id"], title=item["title"]) for item in items],
    )


@app.post("/history/transition")
def history_transition(payload: HistoryTransition) -> dict[str, str | None]:
    with connection() as conn:
        record_transition(conn, payload.from_paper_id, payload.to_paper_id)
    return {"from_paper_id": payload.from_paper_id, "to_paper_id": payload.to_paper_id}


@app.post("/papers/resolve", response_model=PaperResolveResponse)
def resolve_paper(payload: PaperResolveRequest) -> PaperResolveResponse:
    with connection() as conn:
        try:
            paper, source = resolve_paper_from_source(conn, payload.source_url)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    return PaperResolveResponse(paper=paper, source=source)


@app.get("/papers/{paper_id:path}/memo", response_model=PaperMemoResponse)
def paper_memo_get(paper_id: str) -> PaperMemoResponse:
    with connection() as conn:
        memo = get_paper_memo(conn, paper_id)
    if memo is None:
        return PaperMemoResponse(paper_id=paper_id, memo="", updated_at=None)
    return PaperMemoResponse(
        paper_id=paper_id,
        memo=str(memo["memo"]),
        updated_at=memo["updated_at"],
    )


def _load_paper_memo_snapshot(paper_id: str) -> dict[str, object]:
    with connection() as conn:
        memo = get_paper_memo(conn, paper_id)
    if memo is None:
        return {"paper_id": paper_id, "memo": "", "updated_at": None}
    return {
        "paper_id": paper_id,
        "memo": str(memo["memo"]),
        "updated_at": memo["updated_at"].isoformat() if memo.get("updated_at") else None,
    }


async def _memo_broadcast(paper_id: str, payload: dict[str, object]) -> None:
    with memo_room_lock:
        queues = list(memo_room_queues.get(paper_id, set()))
    for queue in queues:
        with suppress(asyncio.QueueFull):
            queue.put_nowait(payload)


async def _memo_sender(websocket: WebSocket, queue: asyncio.Queue[dict[str, object]]) -> None:
    while True:
        payload = await queue.get()
        try:
            await websocket.send_json(payload)
        except (WebSocketDisconnect, ClientDisconnected):
            return


async def _session_sender(websocket: WebSocket, queue: asyncio.Queue[dict[str, object]]) -> None:
    while True:
        payload = await queue.get()
        try:
            await websocket.send_json(payload)
        except (WebSocketDisconnect, ClientDisconnected):
            return


def _session_room_bind(session_id: str, queue: asyncio.Queue[dict[str, object]], loop: asyncio.AbstractEventLoop) -> None:
    with session_room_lock:
        session_room_queues[session_id].add((queue, loop))


def _session_room_unbind(session_id: str, queue: asyncio.Queue[dict[str, object]], loop: asyncio.AbstractEventLoop) -> None:
    with session_room_lock:
        listeners = session_room_queues.get(session_id)
        if listeners is None:
            return
        listeners.discard((queue, loop))
        if not listeners:
            session_room_queues.pop(session_id, None)


def _session_room_drain_pending(session_id: str) -> list[dict[str, object]]:
    with session_room_pending_lock:
        pending = session_room_pending_events.pop(session_id, [])
    return list(pending)


def _session_websocket_connection_count() -> int:
    with session_room_lock:
        return sum(len(listeners) for listeners in session_room_queues.values())


def _session_websocket_connection_counts() -> dict[str, int]:
    with session_room_lock:
        return {session_id: len(listeners) for session_id, listeners in session_room_queues.items()}


def _session_room_buffer_pending(session_id: str, payload: dict[str, object]) -> None:
    with session_room_pending_lock:
        session_room_pending_events[session_id].append(payload)


def _session_room_broadcast(session_id: str, payload: dict[str, object]) -> None:
    with session_room_lock:
        listeners = list(session_room_queues.get(session_id, set()))
    if not listeners:
        _session_room_buffer_pending(session_id, payload)
        return
    for queue, loop in listeners:
        def _enqueue(target_queue: asyncio.Queue[dict[str, object]] = queue) -> None:
            with suppress(asyncio.QueueFull):
                target_queue.put_nowait(payload)

        try:
            loop.call_soon_threadsafe(_enqueue)
        except RuntimeError:
            continue


def _should_broadcast_session_command(command_type: str) -> bool:
    return command_type in {"next", "set_next_candidate", "stop", "regenerate", "playback_started"}


@app.websocket("/papers/{paper_id:path}/memo/ws")
async def paper_memo_stream(websocket: WebSocket, paper_id: str) -> None:
    await websocket.accept()
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=16)
    with memo_room_lock:
        memo_room_queues[paper_id].add(queue)
    sender_task = asyncio.create_task(_memo_sender(websocket, queue))
    initial = await anyio.to_thread.run_sync(_load_paper_memo_snapshot, paper_id)
    await queue.put(initial)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        return
    finally:
        sender_task.cancel()
        with suppress(asyncio.CancelledError):
            await sender_task
        with memo_room_lock:
            queues = memo_room_queues.get(paper_id)
            if queues is not None:
                queues.discard(queue)
                if not queues:
                    memo_room_queues.pop(paper_id, None)


@app.put("/papers/{paper_id:path}/memo", response_model=PaperMemoResponse)
async def paper_memo_put(paper_id: str, payload: PaperMemoUpdateRequest) -> PaperMemoResponse:
    with connection() as conn:
        try:
            _ensure_paper_available(conn, paper_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        memo = upsert_paper_memo(conn, paper_id, payload.memo)
    response = PaperMemoResponse(
        paper_id=paper_id,
        memo=str(memo["memo"]),
        updated_at=memo["updated_at"],
    )
    await _memo_broadcast(paper_id, response.model_dump(mode="json"))
    return response


@app.get("/history/recent")
def history_recent(limit: int = 20) -> dict[str, list[dict[str, str | None]]]:
    with connection() as conn:
        return {"transitions": recent_transitions(conn, limit=limit)}


@app.get("/sessions/recent", response_model=SessionListResponse)
def sessions_recent(limit: int = 20) -> SessionListResponse:
    with connection() as conn:
        sessions = list_playback_sessions(conn, limit=limit)
    session_websocket_connection_counts = _session_websocket_connection_counts()
    return SessionListResponse(
        sessions=[
            SessionListItem(
                session_id=session.session_id,
                status=session.status,
                session_websocket_connections=session_websocket_connection_counts.get(session.session_id, 0),
                root_source_url=session.root_source_url,
                root_paper_id=session.root_paper_id,
                root_paper_title=session.root_paper_title,
                current_paper_id=session.current_paper_id,
                current_paper_title=session.current_paper_title,
                next_event_seq=session.next_event_seq,
                config=session.config,
                started_at=session.started_at,
                updated_at=session.updated_at,
                total_generation_elapsed_ms=session.total_generation_elapsed_ms,
                total_wall_elapsed_ms=session.total_wall_elapsed_ms,
                total_generation_cost_usd=session.total_generation_cost_usd,
            )
            for session in sessions
        ]
    )


@app.get("/sessions/{session_id}", response_model=SessionSnapshot)
def session_snapshot(session_id: str) -> SessionSnapshot:
    with connection() as conn:
        session = get_playback_session(conn, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    if session.status == "active" and session.next_paper_id:
        _schedule_next_paper_prefetch(session.session_id, session.next_paper_id)
    return SessionSnapshot(
        session_id=session.session_id,
        status=session.status,
        root_source_url=session.root_source_url,
        root_paper_id=session.root_paper_id,
        current_paper_id=session.current_paper_id,
        next_paper_id=session.next_paper_id,
        next_event_seq=session.next_event_seq,
        config=session.config,
    )


@app.get("/sessions/{session_id}/events")
def session_events(session_id: str, after_seq: int = 0) -> dict[str, list[dict[str, object]]]:
    with connection() as conn:
        events = list_session_events(conn, session_id, after_seq=after_seq)
    return {
        "events": [
            _session_event_message(event.session_id, event.seq, event.event_type, event.payload)
            for event in events
        ]
    }


def _session_event_message(session_id: str, seq: int, event_type: str, payload: dict[str, object]) -> dict[str, object]:
    return {"session_id": session_id, "seq": seq, "type": event_type, **payload}

def _prefetch_target_is_current(session_id: str, paper_id: str) -> bool:
    with PREFETCH_LOCK:
        return PREFETCH_SESSION_TARGETS.get(session_id) == paper_id


def _schedule_next_paper_prefetch(session_id: str, paper_id: str) -> None:
    if not paper_id:
        return
    key = (session_id, paper_id)
    with PREFETCH_LOCK:
        PREFETCH_SESSION_TARGETS[session_id] = paper_id
        if key in PREFETCH_INFLIGHT:
            return
        PREFETCH_INFLIGHT.add(key)

    def _run() -> None:
        try:
            if not _prefetch_target_is_current(session_id, paper_id):
                raise PrefetchCancelled()
            with connection() as conn:
                def _prefetch_cost_recorder(
                    kind: str,
                    started_at: datetime,
                    finished_at: datetime,
                    elapsed_ms: int,
                    estimated_cost_usd: float,
                    detail: dict[str, object],
                ) -> None:
                    _record_generation_cost_and_notify(
                        conn,
                        kind,
                        session_id=session_id,
                        paper_id=paper_id,
                        started_at=started_at,
                        finished_at=finished_at,
                        elapsed_ms=elapsed_ms,
                        estimated_cost_usd=estimated_cost_usd,
                        detail={**detail, "generation_scope": "prefetch"},
                    )

                generate_explanation(
                    paper_id,
                    should_continue=lambda: _prefetch_target_is_current(session_id, paper_id),
                    cost_recorder=_prefetch_cost_recorder,
                )
        except PrefetchCancelled:
            return
        except Exception:  # noqa: BLE001
            logger.warning("prefetch failed: paper_id=%s", paper_id, exc_info=True)
        finally:
            with PREFETCH_LOCK:
                PREFETCH_INFLIGHT.discard(key)

    PREFETCH_EXECUTOR.submit(_run)


def _set_session_next_paper_id(conn, session_id: str, next_paper_id: str) -> None:
    update_playback_session(conn, session_id, next_paper_id=next_paper_id)
    _schedule_next_paper_prefetch(session_id, next_paper_id)


def _clear_session_prefetch(session_id: str) -> None:
    with PREFETCH_LOCK:
        PREFETCH_SESSION_TARGETS.pop(session_id, None)
    _clear_session_search_prefetch(session_id)


def _latest_playback_started_paper_id(conn, session_id: str) -> str | None:
    events = list_session_events(conn, session_id)
    payload = latest_event_payload(events, "session_playback_started")
    if payload is None:
        return None
    paper_id = str(payload.get("paper_id") or "")
    return paper_id or None


def _search_target_is_current(session_id: str, paper_id: str) -> bool:
    with SEARCH_LOCK:
        return SEARCH_SESSION_TARGETS.get(session_id) == paper_id


def _clear_session_search(session_id: str) -> None:
    with SEARCH_LOCK:
        SEARCH_SESSION_TARGETS.pop(session_id, None)


def _search_prefetch_target_is_current(session_id: str, paper_id: str) -> bool:
    with SEARCH_PREFETCH_LOCK:
        return SEARCH_PREFETCH_SESSION_TARGETS.get(session_id) == paper_id


def _clear_session_search_prefetch(session_id: str) -> None:
    with SEARCH_PREFETCH_LOCK:
        SEARCH_PREFETCH_SESSION_TARGETS.pop(session_id, None)
        for key in [key for key in SEARCH_PREFETCH_CACHE if key[0] == session_id]:
            SEARCH_PREFETCH_CACHE.pop(key, None)


def _consume_session_search_prefetch(session_id: str, paper_id: str) -> dict[str, object] | None:
    key = (session_id, paper_id)
    with SEARCH_PREFETCH_LOCK:
        return SEARCH_PREFETCH_CACHE.pop(key, None)


def _schedule_next_paper_search_prefetch(session_id: str, paper_id: str) -> None:
    if not paper_id:
        return
    key = (session_id, paper_id)
    with SEARCH_PREFETCH_LOCK:
        SEARCH_PREFETCH_SESSION_TARGETS[session_id] = paper_id
        SEARCH_PREFETCH_CACHE.pop(key, None)
        if key in SEARCH_PREFETCH_INFLIGHT:
            return
        SEARCH_PREFETCH_INFLIGHT.add(key)

    def _run() -> None:
        try:
            if not _search_prefetch_target_is_current(session_id, paper_id):
                raise PrefetchCancelled()
            with connection() as conn:
                session = get_playback_session(conn, session_id)
                paper = get_paper(conn, paper_id)
                if session is None or paper is None:
                    raise PrefetchCancelled()

                def _prefetch_cost_recorder(
                    kind: str,
                    started_at: datetime,
                    finished_at: datetime,
                    elapsed_ms: int,
                    estimated_cost_usd: float,
                    detail: dict[str, object],
                ) -> None:
                    _record_generation_cost_and_notify(
                        conn,
                        kind,
                        session_id=session_id,
                        paper_id=paper_id,
                        started_at=started_at,
                        finished_at=finished_at,
                        elapsed_ms=elapsed_ms,
                        estimated_cost_usd=estimated_cost_usd,
                        detail={**detail, "generation_scope": "prefetch"},
                    )

                trail_ids = list_session_trail_paper_ids(conn, session_id)
                search_result = _paper_search_payload(
                    conn,
                    require_openai_client("next paper search prefetch"),
                    session_id,
                    paper,
                    trail_paper_ids=[*trail_ids, paper_id],
                    config=session.config,
                    cost_recorder=_prefetch_cost_recorder,
                )
                if not _search_prefetch_target_is_current(session_id, paper_id):
                    raise PrefetchCancelled()
                with SEARCH_PREFETCH_LOCK:
                    if SEARCH_PREFETCH_SESSION_TARGETS.get(session_id) != paper_id:
                        raise PrefetchCancelled()
                    SEARCH_PREFETCH_CACHE[key] = search_result
                if not _search_prefetch_target_is_current(session_id, paper_id):
                    raise PrefetchCancelled()
                next_paper_id = str(search_result.get("next_paper_id") or "")
                if next_paper_id:
                    _set_session_next_paper_id(conn, session_id, next_paper_id)
        except PrefetchCancelled:
            return
        except Exception:  # noqa: BLE001
            logger.warning("search prefetch failed: session_id=%s paper_id=%s", session_id, paper_id, exc_info=True)
        finally:
            with SEARCH_PREFETCH_LOCK:
                SEARCH_PREFETCH_INFLIGHT.discard(key)

    SEARCH_PREFETCH_EXECUTOR.submit(_run)


def _maybe_schedule_next_paper_search_prefetch(conn, session_id: str) -> None:
    session = get_playback_session(conn, session_id)
    if session is None or session.status != "active" or not session.next_paper_id:
        return
    playback_started_paper_id = _latest_playback_started_paper_id(conn, session_id)
    if playback_started_paper_id != session.current_paper_id:
        return
    _schedule_next_paper_search_prefetch(session_id, session.next_paper_id)


def _schedule_paper_search_update(
    session_id: str,
    paper,
    trail_paper_ids: list[str],
    config: dict[str, object],
    *,
    origin: str,
    from_paper_id: str | None,
) -> None:
    key = (session_id, paper.id)
    with SEARCH_LOCK:
        SEARCH_SESSION_TARGETS[session_id] = paper.id
        if key in SEARCH_INFLIGHT:
            return
        SEARCH_INFLIGHT.add(key)

    def _run() -> None:
        try:
            if not _search_target_is_current(session_id, paper.id):
                raise PrefetchCancelled()
            with connection() as conn:
                client = require_openai_client("paper search update")
                update_payload = _paper_search_payload(
                    conn,
                    client,
                    session_id,
                    paper,
                    trail_paper_ids=trail_paper_ids,
                    config=config,
                )
                if not _search_target_is_current(session_id, paper.id):
                    raise PrefetchCancelled()
                next_paper_id = str(update_payload.get("next_paper_id") or "")
                if next_paper_id:
                    _set_session_next_paper_id(conn, session_id, next_paper_id)
                    _maybe_schedule_next_paper_search_prefetch(conn, session_id)
                event = _append_session_event_message(
                    conn,
                    session_id,
                    "paper_search_updated",
                    {
                        "session_id": session_id,
                        "paper_id": paper.id,
                        "origin": origin,
                        "from_paper_id": from_paper_id,
                        **update_payload,
                    },
                )
                _session_room_broadcast(session_id, event)
        except PrefetchCancelled:
            return
        except Exception:  # noqa: BLE001
            logger.warning("paper search update failed: session_id=%s paper_id=%s", session_id, paper.id, exc_info=True)
        finally:
            with SEARCH_LOCK:
                SEARCH_INFLIGHT.discard(key)

    SEARCH_EXECUTOR.submit(_run)


def _session_cost_payload(conn, session_id: str) -> SessionCostsResponse | None:
    summary = get_session_generation_costs(conn, session_id)
    if summary is None:
        return None
    rows = generation_cost_rows(conn, session_id)
    request_times = session_requested_at_by_paper_id(conn, session_id)
    items = generation_cost_items_from_rows(rows, requested_at_by_paper_id=request_times)
    audio_duration_ms = _session_audio_duration_ms(conn, session_id)
    session = get_playback_session(conn, session_id)
    return SessionCostsResponse(
        session_id=summary.session_id,
        total_elapsed_ms=summary.total_elapsed_ms,
        total_wall_elapsed_ms=summary.total_wall_elapsed_ms,
        total_cost_usd=summary.total_cost_usd,
        is_final=bool(session and session.status != "active"),
        total_elapsed_ms_without_prefetch=generation_cost_wall_elapsed_ms(
            conn,
            session_id,
            requested_at_by_paper_id=request_times,
        ),
        total_cost_usd_without_prefetch=generation_cost_total_cost_usd(
            conn,
            session_id,
            requested_at_by_paper_id=request_times,
        ),
        audio_duration_ms=audio_duration_ms if audio_duration_ms > 0 else None,
        items=items,
    )


def _paper_cost_payload(conn, session_id: str, paper_id: str) -> SessionCostsResponse | None:
    rows = generation_cost_rows(conn, session_id, paper_id=paper_id)
    request_times = session_requested_at_by_paper_id(conn, session_id)
    items = generation_cost_items_from_rows(rows, requested_at_by_paper_id=request_times, missing_as_zero=True)
    audio_duration_ms = _paper_audio_duration_ms(conn, session_id, paper_id)
    total_elapsed_ms = generation_cost_wall_elapsed_ms(conn, session_id, paper_id=paper_id)
    total_cost_usd = generation_cost_total_cost_usd(conn, session_id, paper_id=paper_id)
    is_final = all(item.status == "calculated" for item in items) if items else False
    return SessionCostsResponse(
        session_id=session_id,
        total_elapsed_ms=total_elapsed_ms,
        total_cost_usd=total_cost_usd,
        is_final=is_final,
        total_elapsed_ms_without_prefetch=generation_cost_wall_elapsed_ms(
            conn,
            session_id,
            paper_id=paper_id,
            requested_at_by_paper_id=request_times,
        ),
        total_cost_usd_without_prefetch=generation_cost_total_cost_usd(
            conn,
            session_id,
            paper_id=paper_id,
            requested_at_by_paper_id=request_times,
        ),
        audio_duration_ms=audio_duration_ms if audio_duration_ms > 0 else None,
        items=items,
    )


def _costs_updated_payload(conn, session_id: str, paper_id: str | None) -> dict[str, object] | None:
    session_costs = _session_cost_payload(conn, session_id)
    if session_costs is None:
        return None
    paper_costs = _paper_cost_payload(conn, session_id, paper_id) if paper_id else None
    payload: dict[str, object] = {
        "session_id": session_id,
        "session_costs": session_costs.model_dump(mode="json"),
    }
    if paper_id:
        payload["paper_id"] = paper_id
    if paper_costs is not None:
        payload["paper_costs"] = paper_costs.model_dump(mode="json")
    return payload


def _notify_costs_updated(conn, session_id: str, paper_id: str | None) -> None:
    payload = _costs_updated_payload(conn, session_id, paper_id)
    if payload is None:
        return
    event = _append_session_event_message(conn, session_id, "session_costs_updated", payload)
    _session_room_broadcast(session_id, event)


def _record_generation_cost_and_notify(
    conn,
    kind: str,
    *,
    session_id: str | None,
    paper_id: str | None,
    started_at: datetime,
    finished_at: datetime,
    elapsed_ms: int,
    estimated_cost_usd: float,
    detail: dict[str, object] | None = None,
) -> None:
    record_generation_cost(
        conn,
        kind,
        session_id=session_id,
        paper_id=paper_id,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_ms=elapsed_ms,
        estimated_cost_usd=estimated_cost_usd,
        detail=detail,
    )
    if session_id is not None:
        _notify_costs_updated(conn, session_id, paper_id)


def _record_generation_notice(
    notices: list[str],
    message: str,
    *,
    session_id: str | None = None,
    paper_id: str | None = None,
) -> None:
    notices.append(message)
    logger.warning("generation notice: session_id=%s paper_id=%s message=%s", session_id, paper_id, message)


def _merge_search_payloads(payloads: list[dict[str, object]], limit: int, seed: int | None = None) -> dict[str, object]:
    merged_scores: dict[str, float] = {}
    merged_hits: dict[str, dict[str, object]] = {}
    merged_rejected: dict[str, dict[str, object]] = {}
    hit_source_modes: dict[str, set[str]] = defaultdict(set)
    rejected_source_modes: dict[str, set[str]] = defaultdict(set)
    fallback_used = True

    def _rank_weight(rank: int) -> float:
        return 1.0 / max(rank, 1)

    def _noise(key: str) -> float:
        return random.Random(f"{seed}:{key}").uniform(0.0, 1e-6)

    def _payload_source_modes(payload: dict[str, object]) -> list[str]:
        modes = payload.get("search_modes")
        if isinstance(modes, list):
            return [str(mode) for mode in modes if str(mode)]
        mode = payload.get("search_mode")
        if isinstance(mode, str) and mode:
            return [mode]
        return []

    for payload_index, payload in enumerate(payloads):
        payload_modes = _payload_source_modes(payload)
        fallback_used = fallback_used and bool(payload.get("fallback_used"))
        for rank, hit in enumerate(payload.get("hits") or [], start=1):
            if not isinstance(hit, dict):
                continue
            paper = hit.get("paper")
            if not isinstance(paper, dict):
                continue
            paper_id = str(paper.get("id") or "")
            if not paper_id:
                continue
            score = float(hit.get("score", 0.0))
            contribution = (score * _rank_weight(rank)) + _noise(f"{payload_index}:{paper_id}:{rank}")
            merged_scores[paper_id] = merged_scores.get(paper_id, 0.0) + contribution
            current = merged_hits.get(paper_id)
            if current is None or score > float(current.get("score", 0.0)):
                merged_hits[paper_id] = dict(hit)
            if payload_modes:
                hit_source_modes[paper_id].update(payload_modes)

    hit_ids = set(merged_hits)
    for payload_index, payload in enumerate(payloads):
        payload_modes = _payload_source_modes(payload)
        for rank, candidate in enumerate(payload.get("rejected_candidates") or [], start=1):
            if not isinstance(candidate, dict):
                continue
            paper_id = str(candidate.get("paper_id") or "")
            if not paper_id or paper_id in hit_ids:
                continue
            score = float(candidate.get("score", 0.0))
            contribution = (score * _rank_weight(rank)) + _noise(f"rejected:{payload_index}:{paper_id}:{rank}")
            merged_scores[paper_id] = merged_scores.get(paper_id, 0.0) + contribution
            current = merged_rejected.get(paper_id)
            if current is None or score > float(current.get("score", 0.0)):
                merged_rejected[paper_id] = dict(candidate)
            if payload_modes:
                rejected_source_modes[paper_id].update(payload_modes)

    hits = sorted(
        merged_hits.values(),
        key=lambda item: (-merged_scores.get(str(item.get("paper", {}).get("id", "")), 0.0), str(item.get("paper", {}).get("id", ""))),
    )
    rejected_candidates = sorted(
        merged_rejected.values(),
        key=lambda item: (-merged_scores.get(str(item.get("paper_id", "")), 0.0), str(item.get("paper_id", ""))),
    )
    for hit in hits:
        paper_id = str(hit.get("paper", {}).get("id", ""))
        hit["source_modes"] = sort_search_modes(hit_source_modes.get(paper_id, set()))
    for candidate in rejected_candidates:
        paper_id = str(candidate.get("paper_id", ""))
        candidate["source_modes"] = sort_search_modes(rejected_source_modes.get(paper_id, set()))
    return {
        "hits": hits[:limit],
        "rejected_candidates": rejected_candidates[:limit],
        "fallback_used": fallback_used,
    }


def _session_config_from_message(message: SessionClientMessage) -> dict[str, object]:
    return {
        "source_url": message.source_url,
        "model_name": message.model_name,
        "include_old_vectors": message.include_old_vectors,
        "limit": message.limit,
        "route1_weight": message.route1_weight,
        "route2_weight": message.route2_weight,
        "seed": message.seed,
        "search_modes": message.search_modes,
    }


def _timed_embed(client, model_name: str, text: str) -> tuple:
    """Run embed_text and return (EmbeddingResult, started_at, finished_at, elapsed_ms)."""
    started_at = datetime.now(UTC)
    t0 = perf_counter()
    result = embed_text(client, model_name, text)
    finished_at = datetime.now(UTC)
    return result, started_at, finished_at, int((perf_counter() - t0) * 1000)


def _timing_from_result(result: object, elapsed_ms: int) -> tuple[datetime, datetime]:
    started_at = getattr(result, "started_at", None)
    finished_at = getattr(result, "finished_at", None)
    if hasattr(started_at, "timestamp") and hasattr(finished_at, "timestamp"):
        return started_at, finished_at
    finished_at = datetime.now(UTC)
    started_at = finished_at - timedelta(milliseconds=max(int(elapsed_ms), 0))
    return started_at, finished_at


def _paper_search_payload(
    conn,
    client,
    session_id: str,
    paper,
    *,
    trail_paper_ids: list[str],
    config: dict[str, object],
    cost_recorder: CostRecorder | None = None,
) -> dict[str, object]:
    simple_search_query = build_followup_query(paper.title, paper.abstract)
    enabled_search_modes = set(config.get("search_modes") or [])
    if not enabled_search_modes:
        enabled_search_modes = {"simple", "keyword_list", "fulltext_query"}
    method_limit = 10
    search_requests: dict[str, set[str]] = {}
    notices: list[str] = []
    pending_next_paper_costs: list[tuple[str, datetime, datetime, int, float, dict[str, object]]] = []

    def _add_search_request(mode: str, query_text: str) -> None:
        normalized_query = query_text.strip()
        if not normalized_query:
            return
        search_requests.setdefault(normalized_query, set()).add(mode)

    def _queue_next_paper_cost(
        kind: str,
        started_at: datetime,
        finished_at: datetime,
        elapsed_ms: int,
        estimated_cost_usd: float,
        detail: dict[str, object],
    ) -> None:
        pending_next_paper_costs.append((kind, started_at, finished_at, elapsed_ms, estimated_cost_usd, detail))

    if "simple" in enabled_search_modes:
        _add_search_request("simple", simple_search_query)

    keyword_result = None
    fulltext_result = None
    model_name = str(config["model_name"])
    with ThreadPoolExecutor(max_workers=3) as executor:
        keyword_future = (
            executor.submit(
                generate_search_keyword,
                make_client(settings.openai_api_key),
                settings.explanation_model,
                paper.title,
                paper.abstract,
            )
            if "keyword_list" in enabled_search_modes
            else None
        )
        fulltext_future = (
            executor.submit(
                generate_fulltext_query,
                make_client(settings.openai_api_key),
                settings.explanation_model,
                paper.title,
                paper.abstract,
            )
            if "fulltext_query" in enabled_search_modes
            else None
        )
        simple_embed_future = (
            executor.submit(_timed_embed, make_client(settings.openai_api_key), model_name, simple_search_query)
            if "simple" in enabled_search_modes and simple_search_query.strip()
            else None
        )
        if keyword_future is not None:
            try:
                keyword_result = keyword_future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("search keyword generation failed: session_id=%s paper_id=%s error=%s", session_id, paper.id, exc)
                _record_generation_notice(
                    notices,
                    "検索キーワードの生成に失敗しました。API を利用できませんでした。",
                    session_id=session_id,
                    paper_id=paper.id,
                )
                keyword_result = None
        if fulltext_future is not None:
            try:
                fulltext_result = fulltext_future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("fulltext query generation failed: session_id=%s paper_id=%s error=%s", session_id, paper.id, exc)
                _record_generation_notice(
                    notices,
                    "全文検索クエリの生成に失敗しました。API を利用できませんでした。",
                    session_id=session_id,
                    paper_id=paper.id,
                )
                fulltext_result = None
        simple_embed_result = None
        simple_embed_started_at = None
        simple_embed_finished_at = None
        simple_embed_elapsed_ms = 0
        if simple_embed_future is not None:
            try:
                simple_embed_result, simple_embed_started_at, simple_embed_finished_at, simple_embed_elapsed_ms = simple_embed_future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("simple query embedding failed: session_id=%s paper_id=%s error=%s", session_id, paper.id, exc)
                _record_generation_notice(
                    notices,
                    "通常検索用の埋め込み生成に失敗しました。API を利用できませんでした。",
                    session_id=session_id,
                    paper_id=paper.id,
                )

    if keyword_result is not None:
        keyword_started_at, keyword_finished_at = _timing_from_result(keyword_result, keyword_result.elapsed_ms)
        _queue_next_paper_cost(
            "keyword_generation",
            started_at=keyword_started_at,
            finished_at=keyword_finished_at,
            elapsed_ms=keyword_result.elapsed_ms,
            estimated_cost_usd=float(
                estimate_completion_cost_usd(
                    settings.explanation_model,
                    keyword_result.input_tokens,
                    keyword_result.output_tokens,
                )
            ),
            detail={
                "source_paper_id": paper.id,
                "model_name": settings.explanation_model,
                "input_tokens": keyword_result.input_tokens,
                "output_tokens": keyword_result.output_tokens,
            },
        )
        keyword_search_query = keyword_result.search_keyword.strip()
        if keyword_search_query:
            _add_search_request("keyword_list", keyword_search_query)
    else:
        keyword_search_query = ""

    if fulltext_result is not None:
        fulltext_started_at, fulltext_finished_at = _timing_from_result(fulltext_result, fulltext_result.elapsed_ms)
        _queue_next_paper_cost(
            "query_generation",
            started_at=fulltext_started_at,
            finished_at=fulltext_finished_at,
            elapsed_ms=fulltext_result.elapsed_ms,
            estimated_cost_usd=float(
                estimate_completion_cost_usd(
                    settings.explanation_model,
                    fulltext_result.input_tokens,
                    fulltext_result.output_tokens,
                )
            ),
            detail={
                "source_paper_id": paper.id,
                "model_name": settings.explanation_model,
                "input_tokens": fulltext_result.input_tokens,
                "output_tokens": fulltext_result.output_tokens,
            },
        )
        fulltext_search_query = fulltext_result.search_query.strip()
        if fulltext_search_query:
            _add_search_request("fulltext_query", fulltext_search_query)
    else:
        fulltext_search_query = ""

    precomputed_embeddings: dict[str, object] = {}
    if simple_embed_result is not None and simple_search_query.strip() and simple_embed_started_at is not None and simple_embed_finished_at is not None:
        _queue_next_paper_cost(
            "embedding",
            started_at=simple_embed_started_at,
            finished_at=simple_embed_finished_at,
            elapsed_ms=simple_embed_elapsed_ms,
            estimated_cost_usd=float(
                estimate_embedding_cost_usd(model_name, simple_embed_result.input_tokens)
            ),
            detail={
                "source_paper_id": paper.id,
                "model_name": model_name,
                "input_tokens": simple_embed_result.input_tokens,
                "scope": "query",
                "query_modes": ["simple"],
            },
        )
        precomputed_embeddings[simple_search_query.strip()] = simple_embed_result

    queries_needing_embed = [q for q in search_requests if q not in precomputed_embeddings]
    if queries_needing_embed:
        with ThreadPoolExecutor(max_workers=len(queries_needing_embed)) as executor:
            embed_futures = {
                q: executor.submit(_timed_embed, make_client(settings.openai_api_key), model_name, q)
                for q in queries_needing_embed
            }
        for q, fut in embed_futures.items():
            try:
                embed_result, embed_started_at, embed_finished_at, embed_elapsed_ms = fut.result()
                query_modes = search_requests[q]
                _queue_next_paper_cost(
                    "embedding",
                    started_at=embed_started_at,
                    finished_at=embed_finished_at,
                    elapsed_ms=embed_elapsed_ms,
                    estimated_cost_usd=float(
                        estimate_embedding_cost_usd(model_name, embed_result.input_tokens)
                    ),
                    detail={
                        "source_paper_id": paper.id,
                        "model_name": model_name,
                        "input_tokens": embed_result.input_tokens,
                        "scope": "query",
                        "query_modes": sort_search_modes(query_modes),
                    },
                )
                precomputed_embeddings[q] = embed_result
            except Exception as exc:  # noqa: BLE001
                logger.warning("query embedding failed: session_id=%s paper_id=%s query=%r error=%s", session_id, paper.id, q, exc)
                _record_generation_notice(
                    notices,
                    "検索用の埋め込み生成に失敗しました。API を利用できませんでした。",
                    session_id=session_id,
                    paper_id=paper.id,
                )

    search_payloads: list[dict[str, object]] = []
    for normalized_query, query_modes in search_requests.items():
        embed_result = precomputed_embeddings.get(normalized_query)
        if embed_result is None:
            logger.warning("skipping search for query %r: embedding unavailable", normalized_query)
            continue
        search_request = SearchRequest(
            query=normalized_query,
            model_name=model_name,
            include_old_vectors=bool(config["include_old_vectors"]),
            exclude_paper_ids=trail_paper_ids,
            limit=method_limit,
            route1_weight=float(config["route1_weight"]),
            route2_weight=float(config["route2_weight"]),
            seed=config["seed"] if config["seed"] is None else int(config["seed"]),
        )
        search_response = search_papers(
            conn,
            client,
            search_request,
            embed_result.embedding,
            cost_recorder=lambda kind, started_at, finished_at, elapsed_ms, estimated_cost_usd, detail: _queue_next_paper_cost(
                kind,
                started_at,
                finished_at,
                elapsed_ms,
                estimated_cost_usd,
                {**detail, "source_paper_id": paper.id},
            ),
        )
        search_payload = search_response.model_dump(mode="json")
        search_payload["search_modes"] = sort_search_modes(query_modes)
        search_payloads.append(search_payload)

    if not search_payloads:
        search_payloads.append(
            {
                "hits": [],
                "rejected_candidates": [],
                "fallback_used": True,
            }
        )
    merged_search_response = _merge_search_payloads(
        search_payloads,
        int(config["limit"]),
        seed=config["seed"] if config["seed"] is None else int(config["seed"]),
    )
    next_paper_id, _ = weighted_choice_hit(merged_search_response["hits"], set(trail_paper_ids))
    for kind, started_at, finished_at, elapsed_ms, estimated_cost_usd, detail in pending_next_paper_costs:
        if cost_recorder is not None:
            cost_recorder(
                kind,
                started_at,
                finished_at,
                elapsed_ms,
                estimated_cost_usd,
                {**detail, "paper_id": paper.id},
            )
            continue
        _record_generation_cost_and_notify(
            conn,
            kind,
            session_id=session_id,
            paper_id=paper.id,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_ms=elapsed_ms,
            estimated_cost_usd=estimated_cost_usd,
            detail={**detail, "paper_id": paper.id},
        )
    return {
        "search": merged_search_response,
        "simple_search_query": simple_search_query,
        "followup_query": simple_search_query,
        "keyword_search_query": keyword_search_query or simple_search_query,
        "search_keyword": keyword_search_query or simple_search_query,
        "fulltext_search_query": fulltext_search_query or simple_search_query,
        "search_modes": sorted(enabled_search_modes),
        "next_paper_id": next_paper_id,
        "notices": notices,
    }


def _paper_ready_payload(
    conn,
    client,
    session_id: str,
    paper,
    *,
    origin: str,
    from_paper_id: str | None,
    trail_paper_ids: list[str],
    config: dict[str, object],
    force_explanation: bool = False,
    defer_search: bool = False,
) -> dict[str, object]:
    notices: list[str] = []
    simple_search_query = build_followup_query(paper.title, paper.abstract)
    enabled_search_modes = set(config.get("search_modes") or [])
    if not enabled_search_modes:
        enabled_search_modes = {"simple", "keyword_list", "fulltext_query"}
    prefetched_search_result = _consume_session_search_prefetch(session_id, paper.id) if defer_search else None
    if defer_search:
        if prefetched_search_result is None:
            _schedule_paper_search_update(
                session_id,
                paper,
                trail_paper_ids,
                config,
                origin=origin,
                from_paper_id=from_paper_id,
            )
    explanation_response = generate_explanation(
        paper.id,
        force=force_explanation,
        cost_recorder=lambda kind, started_at, finished_at, elapsed_ms, estimated_cost_usd, detail: _record_generation_cost_and_notify(
            conn,
            kind,
            session_id=session_id,
            paper_id=paper.id,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_ms=elapsed_ms,
            estimated_cost_usd=estimated_cost_usd,
            detail=detail,
        ),
        notice_recorder=lambda message: _record_generation_notice(notices, message, session_id=session_id, paper_id=paper.id),
    )
    search_result: dict[str, object] = {"hits": [], "rejected_candidates": [], "fallback_used": True}
    keyword_search_query = simple_search_query
    fulltext_search_query = simple_search_query
    search_deferred = defer_search
    if prefetched_search_result is not None:
        search_result = prefetched_search_result
        keyword_search_query = str(search_result.get("keyword_search_query") or simple_search_query)
        fulltext_search_query = str(search_result.get("fulltext_search_query") or simple_search_query)
        next_paper_id = str(search_result.get("next_paper_id") or "")
        if next_paper_id:
            _set_session_next_paper_id(conn, session_id, next_paper_id)
        search_notices = search_result.get("notices")
        if isinstance(search_notices, list) and search_notices:
            notices.extend(str(notice) for notice in search_notices)
        search_deferred = False
    if not defer_search:
        search_result = _paper_search_payload(
            conn,
            client,
            session_id,
            paper,
            trail_paper_ids=trail_paper_ids,
            config=config,
        )
        keyword_search_query = str(search_result.get("keyword_search_query") or simple_search_query)
        fulltext_search_query = str(search_result.get("fulltext_search_query") or simple_search_query)
        next_paper_id = str(search_result.get("next_paper_id") or "")
        if next_paper_id:
            _set_session_next_paper_id(conn, session_id, next_paper_id)
        search_notices = search_result.get("notices")
        if isinstance(search_notices, list) and search_notices:
            notices.extend(str(notice) for notice in search_notices)
    session_costs = _session_cost_payload(conn, session_id)
    try:
        paper_costs = _paper_cost_payload(conn, session_id, paper.id)
    except Exception:  # noqa: BLE001
        logger.warning("paper cost payload failed: session_id=%s paper_id=%s", session_id, paper.id, exc_info=True)
        paper_costs = None
    return {
        "session_id": session_id,
        "origin": origin,
        "from_paper_id": from_paper_id,
        "trail_paper_ids": trail_paper_ids,
        "next_paper_id": search_result.get("next_paper_id") if not search_deferred else None,
        "paper": paper.model_dump(mode="json"),
        "search": search_result.get("search") if not search_deferred else {"hits": [], "rejected_candidates": [], "fallback_used": True},
        "search_deferred": search_deferred,
        "simple_search_query": simple_search_query,
        "followup_query": simple_search_query,
        "keyword_search_query": keyword_search_query or simple_search_query,
        "search_keyword": keyword_search_query or simple_search_query,
        "fulltext_search_query": fulltext_search_query or simple_search_query,
        "search_modes": sorted(enabled_search_modes),
        "explanation": explanation_response.explanation,
        "audio_url": explanation_response.audio_url,
        "audio_urls": explanation_response.audio_urls,
        "audio_duration_ms": explanation_response.audio_duration_ms,
        "notices": notices,
        "paper_costs": paper_costs.model_dump(mode="json") if paper_costs is not None else None,
        "session_costs": session_costs.model_dump(mode="json") if session_costs is not None else None,
        "memo": (lambda m: str(m["memo"]) if m else "")(get_paper_memo(conn, paper.id)),
    }


def _append_session_event_message(
    conn,
    session_id: str,
    event_type: str,
    payload: dict[str, object],
) -> dict[str, object]:
    seq = append_session_event(conn, session_id, event_type, payload)
    return _session_event_message(session_id, seq, event_type, payload)


def _session_audio_duration_ms(conn, session_id: str) -> int:
    events = list_session_events(conn, session_id)
    total = 0
    for event in events:
        if event.event_type != "paper_ready":
            continue
        payload = event.payload
        duration = payload.get("audio_duration_ms")
        if isinstance(duration, (int, float)) and duration > 0:
            total += int(duration)
    return total


def _paper_audio_duration_ms(conn, session_id: str, paper_id: str) -> int:
    events = list_session_events(conn, session_id)
    for event in reversed(events):
        if event.event_type != "paper_ready":
            continue
        payload = event.payload
        paper = payload.get("paper")
        if not isinstance(paper, dict):
            continue
        if str(paper.get("id", "")) != paper_id:
            continue
        duration = payload.get("audio_duration_ms")
        if isinstance(duration, (int, float)) and duration > 0:
            return int(duration)
        return 0
    return 0


def _start_session(message: SessionClientMessage) -> list[dict[str, object]]:
    if not message.source_url:
        raise ValueError("source_url is required")
    session_id = str(uuid4())
    config = _session_config_from_message(message)
    with connection() as conn:
        client = require_openai_client("session start")
        paper, origin = resolve_paper_from_source(conn, message.source_url)
        create_playback_session(
            conn,
            session_id=session_id,
            root_source_url=message.source_url,
            root_paper_id=paper.id,
            current_paper_id=paper.id,
            next_paper_id=None,
            config=config,
        )
        append_session_trail_item(conn, session_id, paper.id)
        record_transition(conn, None, paper.id)
        started_event = _append_session_event_message(
            conn,
            session_id,
            "session_started",
            {
                "session_id": session_id,
                "source_url": message.source_url,
                "config": config,
                "root_paper": paper.model_dump(mode="json"),
                "origin": origin,
            },
        )
        trail_ids = list_session_trail_paper_ids(conn, session_id)
        paper_ready = _paper_ready_payload(
            conn,
            client,
            session_id,
            paper,
            origin=origin,
            from_paper_id=None,
            trail_paper_ids=trail_ids,
            config=config,
            defer_search=True,
        )
        paper_event = _append_session_event_message(conn, session_id, "paper_ready", paper_ready)
    return [started_event, paper_event]


def _advance_session(message: SessionClientMessage) -> list[dict[str, object]]:
    if not message.session_id:
        raise ValueError("session_id is required")
    with connection() as conn:
        client = require_openai_client("session advance")
        session = get_playback_session(conn, message.session_id)
        if session is None:
            raise ValueError(f"session not found: {message.session_id}")
        current_paper = get_paper(conn, session.current_paper_id)
        if current_paper is None:
            raise ValueError(f"paper not found: {session.current_paper_id}")
        next_candidate_paper_id = pop_session_next_candidate(conn, message.session_id)
        if next_candidate_paper_id:
            next_paper_id = next_candidate_paper_id
            origin = "next_candidate"
        else:
            next_paper_id = session.next_paper_id
            origin = "search"
            if not next_paper_id:
                history_events = list_session_events(conn, message.session_id)
                last_paper_event = latest_event_payload(history_events, "paper_ready")
                if last_paper_event is None:
                    raise ValueError(f"current paper event not found for session: {message.session_id}")
                origin, next_paper_data = pick_next_paper_from_search_payload(conn, message.session_id, last_paper_event)
                next_paper_id = str(next_paper_data.get("id", ""))
                if not next_paper_id:
                    raise ValueError(f"next paper id missing for session: {message.session_id}")
        next_paper = get_paper(conn, next_paper_id)
        if next_paper is None:
            raise ValueError(f"paper not found: {next_paper_id}")
        next_requested_event = _append_session_event_message(
            conn,
            message.session_id,
            "session_next_requested",
            {
                "session_id": message.session_id,
                "from_paper_id": current_paper.id,
                "to_paper_id": next_paper.id,
            },
        )
        append_session_trail_item(conn, message.session_id, next_paper.id)
        update_playback_session(conn, message.session_id, current_paper_id=next_paper.id)
        record_transition(conn, current_paper.id, next_paper.id)
        advanced_event = _append_session_event_message(
            conn,
            message.session_id,
            "session_advanced",
            {
                "session_id": message.session_id,
                "from_paper_id": current_paper.id,
                "to_paper_id": next_paper.id,
            },
        )
        trail_ids = list_session_trail_paper_ids(conn, message.session_id)
        config = session.config
        paper_ready = _paper_ready_payload(
            conn,
            client,
            message.session_id,
            next_paper,
            origin=origin,
            from_paper_id=current_paper.id,
            trail_paper_ids=trail_ids,
            config=config,
            defer_search=True,
        )
        paper_event = _append_session_event_message(conn, message.session_id, "paper_ready", paper_ready)
    return [next_requested_event, advanced_event, paper_event]


def _set_next_candidate_paper(message: SessionClientMessage) -> list[dict[str, object]]:
    if not message.session_id:
        raise ValueError("session_id is required")
    if not message.paper_id:
        raise ValueError("paper_id is required")
    with connection() as conn:
        session = get_playback_session(conn, message.session_id)
        if session is None:
            raise ValueError(f"session not found: {message.session_id}")
        current_paper = get_paper(conn, session.current_paper_id)
        if current_paper is None:
            raise ValueError(f"paper not found: {session.current_paper_id}")
        next_candidate_paper = get_paper(conn, message.paper_id)
        if next_candidate_paper is None:
            raise ValueError(f"paper not found: {message.paper_id}")
        if next_candidate_paper.id == current_paper.id:
            raise ValueError("current paper cannot be selected as next candidate")
        set_session_next_candidate(conn, message.session_id, next_candidate_paper.id)
        _set_session_next_paper_id(conn, message.session_id, next_candidate_paper.id)
        _maybe_schedule_next_paper_search_prefetch(conn, message.session_id)
        next_candidate_event = _append_session_event_message(
            conn,
            message.session_id,
            "session_next_candidate_updated",
            {
                "session_id": message.session_id,
                "paper_id": next_candidate_paper.id,
                "next_paper_id": next_candidate_paper.id,
            },
        )
    return [next_candidate_event]


def _resume_session(message: SessionClientMessage) -> list[dict[str, object]]:
    if not message.session_id:
        raise ValueError("session_id is required")
    with connection() as conn:
        events = list_session_events(conn, message.session_id, after_seq=message.last_event_seq)
    return [_session_event_message(event.session_id, event.seq, event.event_type, event.payload) for event in events]


def _stop_session(message: SessionClientMessage) -> list[dict[str, object]]:
    if not message.session_id:
        return []
    with connection() as conn:
        session = get_playback_session(conn, message.session_id)
        if session is None:
            raise ValueError(f"session not found: {message.session_id}")
        update_playback_session(conn, message.session_id, status="stopped")
        _clear_session_prefetch(message.session_id)
        _clear_session_search(message.session_id)
        stopped_event = _append_session_event_message(
            conn,
            message.session_id,
            "session_stopped",
            {"session_id": message.session_id, "status": "stopped"},
        )
    return [stopped_event]


def _record_playback_started(message: SessionClientMessage) -> list[dict[str, object]]:
    if not message.session_id:
        raise ValueError("session_id is required")
    with connection() as conn:
        session = get_playback_session(conn, message.session_id)
        if session is None:
            raise ValueError(f"session not found: {message.session_id}")
        paper_id = message.paper_id or session.current_paper_id
        if paper_id != session.current_paper_id:
            raise ValueError(f"current paper mismatch: {paper_id}")
        event = _append_session_event_message(
            conn,
            message.session_id,
            "session_playback_started",
            {
                "session_id": message.session_id,
                "paper_id": paper_id,
            },
        )
        _maybe_schedule_next_paper_search_prefetch(conn, message.session_id)
    return [event]


@app.websocket("/sessions/ws")
async def session_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=128)
    sender_task = asyncio.create_task(_session_sender(websocket, outbox))
    current_session_id: str | None = None

    def _bind_session(session_id: str | None) -> None:
        nonlocal current_session_id
        if not session_id:
            return
        if current_session_id == session_id:
            return
        if current_session_id is not None:
            _session_room_unbind(current_session_id, outbox, loop)
        _session_room_bind(session_id, outbox, loop)
        current_session_id = session_id

    try:
        while True:
            raw_message = await websocket.receive_json()
            message = SessionClientMessage.model_validate(raw_message)
            try:
                if message.type == "start":
                    events = await anyio.to_thread.run_sync(_start_session, message)
                elif message.type == "resume":
                    events = await anyio.to_thread.run_sync(_resume_session, message)
                elif message.type == "next":
                    events = await anyio.to_thread.run_sync(_advance_session, message)
                elif message.type == "set_next_candidate":
                    events = await anyio.to_thread.run_sync(_set_next_candidate_paper, message)
                elif message.type == "stop":
                    events = await anyio.to_thread.run_sync(_stop_session, message)
                elif message.type == "regenerate":
                    events = await anyio.to_thread.run_sync(_regenerate_session, message)
                elif message.type == "playback_started":
                    events = await anyio.to_thread.run_sync(_record_playback_started, message)
                else:
                    raise ValueError(f"unknown session command: {message.type}")
            except ValueError as exc:
                logger.warning("session command rejected: type=%s session_id=%s error=%s", message.type, message.session_id, exc)
                await outbox.put(
                    {
                        "type": "error",
                        "message": str(exc),
                        "session_id": message.session_id,
                    }
                )
                continue
            except Exception as exc:  # noqa: BLE001
                logger.exception("session command failed: type=%s session_id=%s", message.type, message.session_id)
                await outbox.put(
                    {
                        "type": "error",
                        "message": str(exc),
                        "session_id": message.session_id,
                    }
                )
                continue
            event_session_id = message.session_id
            for event in events:
                maybe_session_id = event.get("session_id")
                if isinstance(maybe_session_id, str) and maybe_session_id:
                    event_session_id = maybe_session_id
                    break
            if not event_session_id and current_session_id is not None:
                event_session_id = current_session_id
            _bind_session(event_session_id)
            should_broadcast = _should_broadcast_session_command(message.type)
            for event in events:
                if should_broadcast:
                    _session_room_broadcast(event_session_id, event)
                else:
                    await outbox.put(event)
                if event.get("type") == "session_stopped" and current_session_id is not None:
                    _session_room_unbind(current_session_id, outbox, loop)
                    current_session_id = None
            if current_session_id is not None:
                for pending_event in _session_room_drain_pending(current_session_id):
                    await outbox.put(pending_event)
    except WebSocketDisconnect:
        return
    except RuntimeError as exc:
        if "WebSocket is not connected" not in str(exc):
            raise
        return
    finally:
        if current_session_id is not None:
            _session_room_unbind(current_session_id, outbox, loop)
        sender_task.cancel()
        with suppress(asyncio.CancelledError):
            await sender_task


def explanation_audio_path(paper_id: str) -> Path:
    return settings.cache_dir / "explanations" / paper_id / "explanation.wav"


def explanation_audio_chunk_dir(paper_id: str) -> Path:
    return settings.cache_dir / "explanations" / paper_id / "chunks"


def explanation_audio_chunk_path(paper_id: str, chunk_index: int) -> Path:
    return explanation_audio_chunk_dir(paper_id) / f"{chunk_index:04d}.wav"


def explanation_audio_chunk_url(paper_id: str, chunk_index: int) -> str:
    return f"/audio/{paper_id}/chunks/{chunk_index:04d}"


def explanation_audio_chunk_texts(explanation: str) -> list[str]:
    return chunk_text(explanation, max_length=220)


def explanation_audio_meta_path(paper_id: str) -> Path:
    return settings.cache_dir / "explanations" / paper_id / "speaker.json"


def _read_explanation_audio_speaker_id(paper_id: str) -> str | None:
    meta_path = explanation_audio_meta_path(paper_id)
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    speaker_id = payload.get("speaker_id") if isinstance(payload, dict) else None
    return str(speaker_id) if speaker_id is not None else None


def _write_explanation_audio_speaker_id(paper_id: str, speaker_id: str) -> None:
    meta_path = explanation_audio_meta_path(paper_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"speaker_id": speaker_id}, ensure_ascii=False), encoding="utf-8")


def clear_explanation_audio_cache(paper_id: str) -> None:
    audio_path = explanation_audio_path(paper_id)
    if audio_path.exists():
        audio_path.unlink()
    chunk_dir = explanation_audio_chunk_dir(paper_id)
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    meta_path = explanation_audio_meta_path(paper_id)
    if meta_path.exists():
        meta_path.unlink()


def _wav_duration_ms(path: Path) -> int | None:
    """Return duration of a WAV file in milliseconds, or None if unreadable."""
    import wave
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return int(frames * 1000 / rate)
    except Exception:  # noqa: BLE001
        return None


def ensure_explanation_audio(paper_id: str, explanation: str, force: bool = False) -> None:
    _ensure_explanation_audio(paper_id, explanation, force=force)


def _ensure_explanation_audio(
    paper_id: str,
    explanation: str,
    *,
    force: bool = False,
    cost_recorder: CostRecorder | None = None,
    should_continue: Callable[[], bool] | None = None,
    notice_recorder: Callable[[str], None] | None = None,
) -> None:
    audio_path = explanation_audio_path(paper_id)
    started_at = datetime.now(UTC)
    started_perf = perf_counter()
    last_error: Exception | None = None
    paper_speaker = build_voicevox_speaker(
        url=settings.voicevox_url,
        fallback_speaker_id=settings.voicevox_speaker_id,
        key=paper_id,
        speed=settings.voicevox_speed_scale,
        volume=settings.voicevox_volume_scale,
    )
    cached_speaker_id = _read_explanation_audio_speaker_id(paper_id)
    if audio_path.exists() and not force and cached_speaker_id == paper_speaker.speaker_id:
        if cost_recorder is not None:
            cache_hit_at = datetime.now(UTC)
            cost_recorder(
                "audio",
                cache_hit_at,
                cache_hit_at,
                0,
                0.0,
                {
                    "paper_id": paper_id,
                    "cache_hit": True,
                    "speaker_id": paper_speaker.speaker_id,
                    "force": force,
                },
            )
        return
    if should_continue is not None and not should_continue():
        raise PrefetchCancelled()
    if audio_path.exists() or force:
        clear_explanation_audio_cache(paper_id)
    _write_explanation_audio_speaker_id(paper_id, paper_speaker.speaker_id)
    for attempt in range(5):
        try:
            chunk_dir = explanation_audio_chunk_dir(paper_id)
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_paths: list[Path] = []
            for index, chunk in enumerate(chunk_text(explanation, max_length=220)):
                if should_continue is not None and not should_continue():
                    raise PrefetchCancelled()
                path = chunk_dir / f"{index:04d}.wav"
                if not path.exists():
                    path.write_bytes(paper_speaker.create_audio_bytes(chunk))
                chunk_paths.append(path)
            if should_continue is not None and not should_continue():
                raise PrefetchCancelled()
            if chunk_paths:
                merge_wav_files(chunk_paths, audio_path)
            if cost_recorder is not None:
                finished_at = datetime.now(UTC)
                cost_recorder(
                    "audio",
                    started_at,
                    finished_at,
                    int((perf_counter() - started_perf) * 1000),
                    0.0,
                    {"paper_id": paper_id, "chunk_count": len(chunk_paths), "force": force},
                )
            return
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, PrefetchCancelled):
                raise
            last_error = exc
            logger.warning(
                "audio generation failed: paper_id=%s attempt=%s error=%s",
                paper_id,
                attempt + 1,
                exc,
            )
            time.sleep(2)
    if last_error is not None:
        logger.warning("audio generation gave up: paper_id=%s error=%s", paper_id, last_error)
        if notice_recorder is not None:
            notice_recorder("音声生成に失敗しました。API を利用できませんでした。")


def generate_explanation(
    paper_id: str,
    force: bool = False,
    *,
    cost_recorder: CostRecorder | None = None,
    should_continue: Callable[[], bool] | None = None,
    notice_recorder: Callable[[str], None] | None = None,
) -> ExplanationResponse:
    notices: list[str] = []

    def record_notice(message: str) -> None:
        notices.append(message)
        if notice_recorder is not None:
            notice_recorder(message)

    with connection() as conn:
        paper = get_paper(conn, paper_id)
        explanation = None if force else get_explanation(conn, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")

    if explanation is None:
        client = require_openai_client("explanation generation")
        prompt = (
            "次の arXiv 論文アブストを、研究者でない人にも伝わる日本語で 5 文前後に解説してください。\n"
            "タイトル: {title}\n"
            "アブスト: {abstract}\n"
            "重要点、何が新しいか、どんな用途かを簡潔に含めてください。"
        ).format(title=paper.title, abstract=paper.abstract)
        started_at = datetime.now(UTC)
        started_perf = perf_counter()
        response = client.responses.create(
            model=settings.explanation_model,
            input=prompt,
            reasoning={"effort": "none"},
        )
        explanation = response.output_text.strip()
        finished_at = datetime.now(UTC)
        elapsed_ms = int((perf_counter() - started_perf) * 1000)
        usage = getattr(response, "usage", None)
        input_tokens = None
        output_tokens = None
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        if cost_recorder is not None:
            cost_recorder(
                "explanation",
                started_at,
                finished_at,
                elapsed_ms,
                float(estimate_completion_cost_usd(settings.explanation_model, input_tokens, output_tokens)),
                {
                    "paper_id": paper_id,
                    "model_name": settings.explanation_model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "force": force,
                },
            )
        with connection() as conn:
            upsert_explanation(conn, paper_id, settings.explanation_model, explanation)
    elif cost_recorder is not None:
        cache_hit_at = datetime.now(UTC)
        cost_recorder(
            "explanation",
            cache_hit_at,
            cache_hit_at,
            0,
            0.0,
            {
                "paper_id": paper_id,
                "model_name": settings.explanation_model,
                "cache_hit": True,
                "force": force,
            },
        )
    if should_continue is not None and not should_continue():
        raise PrefetchCancelled()
    _ensure_explanation_audio(
        paper_id,
        explanation,
        force=force,
        cost_recorder=cost_recorder,
        should_continue=should_continue,
        notice_recorder=record_notice,
    )
    audio_urls = [explanation_audio_chunk_url(paper_id, index) for index, _ in enumerate(explanation_audio_chunk_texts(explanation))]
    audio_duration_ms = _wav_duration_ms(explanation_audio_path(paper_id))

    return ExplanationResponse(
        paper_id=paper.id,
        title=paper.title,
        explanation=explanation,
        audio_url=f"/audio/{paper.id}",
        audio_urls=audio_urls,
        audio_duration_ms=audio_duration_ms,
        notices=notices,
    )


@app.post("/explanations/{paper_id:path}", response_model=ExplanationResponse)
def explain(paper_id: str) -> ExplanationResponse:
    return generate_explanation(paper_id)


def _regenerate_session(message: SessionClientMessage) -> list[dict[str, object]]:
    if not message.session_id:
        raise ValueError("session_id is required")
    with connection() as conn:
        session = get_playback_session(conn, message.session_id)
        if session is None:
            raise ValueError(f"session not found: {message.session_id}")
        paper = get_paper(conn, session.current_paper_id)
        if paper is None:
            raise ValueError(f"paper not found: {session.current_paper_id}")
        trail_ids = list_session_trail_paper_ids(conn, message.session_id)
        update_playback_session(conn, message.session_id, current_paper_id=paper.id)
        regenerated_event = _append_session_event_message(
            conn,
            message.session_id,
            "session_regenerated",
            {
                "session_id": message.session_id,
                "paper_id": paper.id,
                "title": paper.title,
            },
        )
        paper_ready = _append_session_event_message(
            conn,
            message.session_id,
            "paper_ready",
            _paper_ready_payload(
                conn,
                require_openai_client("session regenerate"),
                message.session_id,
                paper,
                origin="regenerate",
                from_paper_id=paper.id,
                trail_paper_ids=trail_ids,
                config=session.config,
                force_explanation=True,
                defer_search=True,
            ),
        )
    return [regenerated_event, paper_ready]


@app.get("/audio/{paper_id:path}/chunks/{chunk_index}")
def audio_chunk(paper_id: str, chunk_index: int) -> FileResponse:
    path = explanation_audio_chunk_path(paper_id, chunk_index)
    if not path.exists():
        with connection() as conn:
            explanation = get_explanation(conn, paper_id)
            if explanation is None:
                _ = generate_explanation(paper_id)
            else:
                ensure_explanation_audio(paper_id, explanation)
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio chunk not found")
    return FileResponse(path)


@app.get("/audio/{paper_id:path}")
def audio(paper_id: str) -> FileResponse:
    path = explanation_audio_path(paper_id)
    if not path.exists():
        with connection() as conn:
            explanation = get_explanation(conn, paper_id)
            if explanation is None:
                _ = generate_explanation(paper_id)
            else:
                ensure_explanation_audio(paper_id, explanation)
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(path)
