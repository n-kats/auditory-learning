from __future__ import annotations

import asyncio
import sys
import threading
import time
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from queue import Queue
from typing import Literal

import fastapi
import openai
import psycopg
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, Form, UploadFile
from pdf2image import convert_from_path
from PIL import Image
from pydantic import BaseModel
from starlette.websockets import WebSocket, WebSocketDisconnect

from v2_auditory_learning.db import Repository
from v2_auditory_learning.costs import estimate_completion_cost_usd
from v2_auditory_learning.generation_queue import GenerationTaskScheduler
from v2_auditory_learning.settings import (
    data_dir,
    default_reasoning_effort,
    frontend_origin_regex,
    frontend_url,
    default_model_name,
    postgres_dsn,
    prompt_explain_path,
    prompt_speak_path,
    requested_voicevox_url,
    voicevox_url,
)
from v2_auditory_learning.session_sync import (
    SessionBroadcastHub,
    build_session_snapshot_event,
)
from v2_auditory_learning.utils.gpt_utils import run_gpt, to_image_content
from v2_auditory_learning.utils.pdf_utils import download_pdf
from v2_auditory_learning.utils.voice_utils import VoiceVoxSpeaker, text_to_wav

if requested_voicevox_url and requested_voicevox_url.strip().rstrip("/") == voicevox_url:
    print(f"[INFO] Using VOICEVOX URL: {voicevox_url}", file=sys.stderr)
else:
    print(f"[INFO] Using fallback VOICEVOX URL: {voicevox_url}", file=sys.stderr)
print("[INFO] Using postgres repository", file=sys.stderr)

app = fastapi.FastAPI(title="v2-auditory-learning")
client = openai.Client()
cors_options: dict[str, object] = {
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
if frontend_url:
    cors_options["allow_origins"] = [frontend_url]
elif frontend_origin_regex:
    cors_options["allow_origins"] = []
    cors_options["allow_origin_regex"] = frontend_origin_regex
else:
    cors_options["allow_origins"] = []
app.add_middleware(CORSMiddleware, **cors_options)
repository: Repository | None = None
repository_lock = threading.Lock()
session_broadcast_hub = SessionBroadcastHub()
generation_scheduler: GenerationTaskScheduler["GenerationResult"] = GenerationTaskScheduler()


def get_repository() -> Repository:
    global repository
    if repository is None:
        with repository_lock:
            if repository is None:
                repository = Repository(postgres_dsn)
    return repository


def wait_for_database_ready(timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            get_repository()
            print("[INFO] postgres repository is ready", file=sys.stderr)
            return
        except psycopg.OperationalError as error:
            last_error = error
            print(f"[INFO] waiting for postgres repository: {error}", file=sys.stderr)
            time.sleep(1)
    raise RuntimeError("postgres repository is not ready") from last_error


class InitRequest(BaseModel):
    url: str
    prompt_explain_text: str | None = None
    prompt_speak_text: str | None = None
    model_name: str | None = None
    reasoning_effort: str | None = None


class InitResponse(BaseModel):
    request_id: str
    source_url: str
    page_num: int


class SessionSummary(BaseModel):
    request_id: str
    source_url: str
    page_num: int | None = None
    current_page: int | None = None
    is_favorited: bool = False
    prompt_explain_text: str = ""
    prompt_speak_text: str = ""
    model_name: str = default_model_name
    reasoning_effort: str = ""
    total_generation_count: int = 0
    total_generation_elapsed_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    created_at: str
    updated_at: str


class SessionSnapshot(BaseModel):
    request_id: str
    source_url: str
    page_num: int | None = None
    current_page: int | None = None
    is_favorited: bool = False
    prompt_explain_text: str = ""
    prompt_speak_text: str = ""
    model_name: str = default_model_name
    reasoning_effort: str = ""
    total_generation_count: int = 0
    total_generation_elapsed_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    created_at: str
    updated_at: str


class GenerationStatusEvent(BaseModel):
    type: Literal["generation_started", "generation_finished"]
    request_id: str
    page_num: int


class FavoriteToggleResponse(BaseModel):
    request_id: str
    page_num: int
    favorited: bool


class FavoriteToggleRequest(BaseModel):
    page_num: int | None = None


class FavoriteItem(BaseModel):
    request_id: str
    favorite_page_num: int
    favorited_at: str
    paper_id: str
    source_url: str
    page_num: int
    current_page: int | None = None
    prompt_explain_text: str = ""
    prompt_speak_text: str = ""
    model_name: str = default_model_name
    reasoning_effort: str = ""
    total_generation_count: int = 0
    total_generation_elapsed_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    created_at: str
    updated_at: str
    is_favorited: bool = True


class FavoriteListResponse(BaseModel):
    items: list[FavoriteItem]


class PromptResponse(BaseModel):
    prompt_explain_text: str
    prompt_speak_text: str
    model_name: str
    reasoning_effort: str


class SessionSettingsResponse(BaseModel):
    request_id: str
    source_url: str
    prompt_explain_text: str
    prompt_speak_text: str
    model_name: str
    reasoning_effort: str


class SessionSettingsRequest(BaseModel):
    prompt_explain_text: str | None = None
    prompt_speak_text: str | None = None
    model_name: str | None = None
    reasoning_effort: str | None = None


def load_default_prompt_explain_text() -> str:
    return prompt_explain_path.read_text().strip()


def load_default_prompt_speak_text() -> str:
    return prompt_speak_path.read_text().strip()


def resolve_upload_source_url(request_id: str, filename: str | None) -> str:
    safe_filename = Path(filename).name if filename else "uploaded.pdf"
    return f"upload://{request_id}/{safe_filename}"


async def save_uploaded_pdf(upload_file: UploadFile, destination: Path, *, source_url: str) -> None:
    chunk_size = 1024 * 1024
    chunk_index = 0
    try:
        with destination.open("wb") as output_file:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                chunk_index += 1
                try:
                    output_file.write(chunk)
                except OSError as error:
                    print(
                        f"[ERROR] Failed to write upload chunk {chunk_index} for {source_url} to {destination}: {error}",
                        file=sys.stderr,
                    )
                    raise
    finally:
        await upload_file.close()


def initialize_session_from_pdf(
    request_id: str,
    source_url: str,
    pdf_path: Path,
    *,
    prompt_explain_text: str | None,
    prompt_speak_text: str | None,
    model_name: str | None,
    reasoning_effort: str | None,
) -> InitResponse:
    work_dir = data_dir / request_id
    image_dir = work_dir / "images"
    work_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(pdf_path)
    for i, page in enumerate(pages, start=1):
        if not (image_dir / f"{i:04d}.png").exists():
            page.save(image_dir / f"{i:04d}.png")
    resolved_prompt_explain_text = (
        prompt_explain_text.strip() if prompt_explain_text and prompt_explain_text.strip() else load_default_prompt_explain_text()
    )
    resolved_prompt_speak_text = (
        prompt_speak_text.strip() if prompt_speak_text and prompt_speak_text.strip() else load_default_prompt_speak_text()
    )
    resolved_model_name = model_name.strip() if model_name and model_name.strip() else default_model_name
    resolved_reasoning_effort = reasoning_effort.strip() if reasoning_effort and reasoning_effort.strip() else ""
    get_repository().upsert_document(
        request_id,
        source_url,
        len(pages),
        current_page=1,
        prompt_explain_text=resolved_prompt_explain_text,
        prompt_speak_text=resolved_prompt_speak_text,
        model_name=resolved_model_name,
        reasoning_effort=resolved_reasoning_effort,
    )
    broadcast_session_snapshot(request_id)
    return InitResponse(request_id=request_id, source_url=source_url, page_num=len(pages))


def resolve_document_prompt_explain_text(request_id: str) -> str:
    row = get_repository().get_document(request_id)
    if row is None:
        return load_default_prompt_explain_text()
    prompt_explain_text = str(row.get("prompt_explain_text") or "").strip()
    return prompt_explain_text if prompt_explain_text else load_default_prompt_explain_text()


def resolve_document_prompt_speak_text(request_id: str) -> str:
    row = get_repository().get_document(request_id)
    if row is None:
        return load_default_prompt_speak_text()
    prompt_speak_text = str(row.get("prompt_speak_text") or "").strip()
    return prompt_speak_text if prompt_speak_text else load_default_prompt_speak_text()


def resolve_document_model_name(request_id: str) -> str:
    row = get_repository().get_document(request_id)
    if row is None:
        return default_model_name
    model_name = str(row.get("model_name") or "").strip()
    return model_name if model_name else default_model_name


def resolve_document_reasoning_effort(request_id: str) -> str:
    row = get_repository().get_document(request_id)
    if row is None:
        return default_reasoning_effort
    effort = str(row.get("reasoning_effort") or "").strip()
    return effort if effort else default_reasoning_effort


def record_session_result(
    request_id: str,
    page: int,
    explanation: str,
    speech_text: str,
    *,
    audio_status: Literal["ready", "failed"] = "ready",
    audio_error: str | None = None,
) -> None:
    repository = get_repository()
    if not hasattr(repository, "upsert_result"):
        return
    repository.upsert_result(
        request_id,
        page,
        explanation,
        speech_text=speech_text,
        prompt_explain_text=resolve_document_prompt_explain_text(request_id),
        prompt_speak_text=resolve_document_prompt_speak_text(request_id),
        model_name=resolve_document_model_name(request_id),
        audio_status=audio_status,
        audio_error=audio_error,
    )


def record_session_usage(
    request_id: str,
    page: int,
    *,
    result_id: str | None,
    kind: str,
    elapsed_ms: int,
    input_tokens: int | None,
    output_tokens: int | None,
    model_name: str,
    detail: dict[str, object] | None = None,
) -> None:
    repository = get_repository()
    current = repository.get_document(request_id)
    if current is None:
        return
    cost_usd = estimate_completion_cost_usd(model_name, input_tokens, output_tokens)
    repository.record_session_usage(
        request_id,
        paper_id=str(current["paper_id"]),
        result_id=result_id,
        kind=kind,
        page_num=page,
        model_name=model_name,
        elapsed_ms=elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        detail=detail,
    )


def build_session_summary(row: dict[str, object]) -> SessionSummary:
    request_id = str(row["request_id"])
    return SessionSummary(
        request_id=request_id,
        source_url=str(row["source_url"]),
        page_num=row["page_num"],
        current_page=row["current_page"],
        is_favorited=bool(row.get("is_favorited", False)),
        prompt_explain_text=str(row.get("prompt_explain_text", "")),
        prompt_speak_text=str(row.get("prompt_speak_text", "")),
        model_name=str(row.get("model_name", default_model_name)),
        reasoning_effort=str(row.get("reasoning_effort", "")),
        total_generation_count=int(row.get("total_generation_count", 0) or 0),
        total_generation_elapsed_ms=int(row.get("total_generation_elapsed_ms", 0) or 0),
        total_input_tokens=int(row.get("total_input_tokens", 0) or 0),
        total_output_tokens=int(row.get("total_output_tokens", 0) or 0),
        total_cost_usd=float(row.get("total_cost_usd", 0.0) or 0.0),
        created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
        updated_at=row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else str(row["updated_at"]),
    )


def build_session_snapshot(row: dict[str, object]) -> SessionSnapshot:
    request_id = str(row["request_id"])
    return SessionSnapshot(
        request_id=request_id,
        source_url=str(row["source_url"]),
        page_num=row["page_num"],
        current_page=row["current_page"],
        is_favorited=bool(row.get("is_favorited", False)),
        prompt_explain_text=str(row.get("prompt_explain_text", "")),
        prompt_speak_text=str(row.get("prompt_speak_text", "")),
        model_name=str(row.get("model_name", default_model_name)),
        reasoning_effort=str(row.get("reasoning_effort", "")),
        total_generation_count=int(row.get("total_generation_count", 0) or 0),
        total_generation_elapsed_ms=int(row.get("total_generation_elapsed_ms", 0) or 0),
        total_input_tokens=int(row.get("total_input_tokens", 0) or 0),
        total_output_tokens=int(row.get("total_output_tokens", 0) or 0),
        total_cost_usd=float(row.get("total_cost_usd", 0.0) or 0.0),
        created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
        updated_at=row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else str(row["updated_at"]),
    )


def build_favorite_item(row: dict[str, object]) -> FavoriteItem:
    return FavoriteItem(
        request_id=str(row["request_id"]),
        favorite_page_num=int(row["favorite_page_num"]),
        favorited_at=row["favorited_at"].isoformat() if hasattr(row["favorited_at"], "isoformat") else str(row["favorited_at"]),
        paper_id=str(row["paper_id"]),
        source_url=str(row["source_url"]),
        page_num=int(row["page_num"]),
        current_page=row["current_page"],
        prompt_explain_text=str(row.get("prompt_explain_text", "")),
        prompt_speak_text=str(row.get("prompt_speak_text", "")),
        model_name=str(row.get("model_name", default_model_name)),
        reasoning_effort=str(row.get("reasoning_effort", "")),
        total_generation_count=int(row.get("total_generation_count", 0) or 0),
        total_generation_elapsed_ms=int(row.get("total_generation_elapsed_ms", 0) or 0),
        total_input_tokens=int(row.get("total_input_tokens", 0) or 0),
        total_output_tokens=int(row.get("total_output_tokens", 0) or 0),
        total_cost_usd=float(row.get("total_cost_usd", 0.0) or 0.0),
        created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
        updated_at=row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else str(row["updated_at"]),
        is_favorited=bool(row.get("is_favorited", True)),
    )


def broadcast_session_snapshot(request_id: str) -> None:
    row = get_repository().get_document(request_id)
    if row is None:
        return
    payload = asdict(
        build_session_snapshot_event(
            request_id=request_id,
            current_page=row.get("current_page"),
            is_favorited=bool(getattr(get_repository(), "is_favorited", lambda _request_id: False)(request_id)),
            prompt_explain_text=str(row.get("prompt_explain_text", "")),
            prompt_speak_text=str(row.get("prompt_speak_text", "")),
            total_generation_count=int(row.get("total_generation_count", 0) or 0),
            total_generation_elapsed_ms=int(row.get("total_generation_elapsed_ms", 0) or 0),
            total_input_tokens=int(row.get("total_input_tokens", 0) or 0),
            total_output_tokens=int(row.get("total_output_tokens", 0) or 0),
            total_cost_usd=float(row.get("total_cost_usd", 0.0) or 0.0),
        )
    )
    payload["source_url"] = row.get("source_url", "")
    payload["page_num"] = row.get("page_num")
    payload["prompt_explain_text"] = row.get("prompt_explain_text", "")
    payload["prompt_speak_text"] = row.get("prompt_speak_text", "")
    payload["model_name"] = row.get("model_name", default_model_name)
    payload["reasoning_effort"] = row.get("reasoning_effort", "")
    payload["total_generation_count"] = row.get("total_generation_count", 0)
    payload["total_generation_elapsed_ms"] = row.get("total_generation_elapsed_ms", 0)
    payload["total_input_tokens"] = row.get("total_input_tokens", 0)
    payload["total_output_tokens"] = row.get("total_output_tokens", 0)
    payload["total_cost_usd"] = row.get("total_cost_usd", 0.0)
    session_broadcast_hub.broadcast(request_id, payload)


def broadcast_session_page_updated(request_id: str, current_page: int) -> None:
    session_broadcast_hub.broadcast(
        request_id,
        {
            "type": "page_updated",
            "request_id": request_id,
            "current_page": current_page,
            "is_favorited": get_repository().is_favorited(request_id, current_page),
        },
    )


def broadcast_favorite_changed(request_id: str, page_num: int, favorited: bool) -> None:
    session_broadcast_hub.broadcast(
        request_id,
        {
            "type": "favorite_toggled",
            "request_id": request_id,
            "page_num": page_num,
            "is_favorited": favorited,
        },
    )


def broadcast_generation_started(request_id: str, page_num: int) -> None:
    session_broadcast_hub.broadcast(
        request_id,
        {
            "type": "generation_started",
            "request_id": request_id,
            "page_num": page_num,
        },
    )


def broadcast_generation_finished(request_id: str, page_num: int) -> None:
    session_broadcast_hub.broadcast(
        request_id,
        {
            "type": "generation_finished",
            "request_id": request_id,
            "page_num": page_num,
        },
    )


def broadcast_playback_started(request_id: str, page_num: int) -> None:
    session_broadcast_hub.broadcast(
        request_id,
        {
            "type": "playback_started",
            "request_id": request_id,
            "page_num": page_num,
        },
    )


def broadcast_playback_stopped(request_id: str) -> None:
    session_broadcast_hub.broadcast(
        request_id,
        {
            "type": "playback_stopped",
            "request_id": request_id,
        },
    )


@app.get("/prompt/default")
def prompt_default() -> PromptResponse:
    return PromptResponse(
        prompt_explain_text=load_default_prompt_explain_text(),
        prompt_speak_text=load_default_prompt_speak_text(),
        model_name=default_model_name,
        reasoning_effort=default_reasoning_effort,
    )


@app.post("/init/")
def init(req: InitRequest) -> InitResponse:
    request_id = get_repository().create_session_id()
    work_dir = data_dir / request_id
    pdf_path = work_dir / "pdf.pdf"
    work_dir.mkdir(parents=True, exist_ok=True)
    if not pdf_path.exists():
        print(f"[INFO] Download PDF from {req.url}", file=sys.stderr)
        pdf_path.write_bytes(download_pdf(req.url))
    return initialize_session_from_pdf(
        request_id,
        req.url,
        pdf_path,
        prompt_explain_text=req.prompt_explain_text,
        prompt_speak_text=req.prompt_speak_text,
        model_name=req.model_name,
        reasoning_effort=req.reasoning_effort,
    )


@app.post("/init/upload/")
async def init_upload(
    file: UploadFile = File(...),
    prompt_explain_text: str | None = Form(None),
    prompt_speak_text: str | None = Form(None),
    model_name: str | None = Form(None),
    reasoning_effort: str | None = Form(None),
) -> InitResponse:
    request_id = get_repository().create_session_id()
    work_dir = data_dir / request_id
    pdf_path = work_dir / "pdf.pdf"
    source_url = resolve_upload_source_url(request_id, file.filename)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Upload init started for {source_url} into {pdf_path}", file=sys.stderr)
    try:
        await save_uploaded_pdf(file, pdf_path, source_url=source_url)
    except Exception as error:
        print(f"[ERROR] Upload init failed while saving {source_url}: {error}", file=sys.stderr)
        print(f"[ERROR] Failed to store uploaded PDF for {source_url} at {pdf_path}: {error}", file=sys.stderr)
        raise
    print(f"[INFO] Upload PDF saved for {source_url}; starting PDF conversion", file=sys.stderr)
    try:
        response = initialize_session_from_pdf(
            request_id,
            source_url,
            pdf_path,
            prompt_explain_text=prompt_explain_text,
            prompt_speak_text=prompt_speak_text,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        )
    except Exception as error:
        print(f"[ERROR] Upload init failed after saving {source_url}: {error}", file=sys.stderr)
        raise
    print(f"[INFO] Upload init finished for {source_url}", file=sys.stderr)
    return response


@app.on_event("startup")
def startup() -> None:
    wait_for_database_ready()


@app.get("/sessions/")
def sessions(limit: int = 20) -> list[SessionSummary]:
    repository = get_repository()
    return [
        build_session_summary({**row, "is_favorited": repository.is_favorited(str(row["request_id"]))})
        for row in repository.list_documents(limit=limit)
    ]


@app.get("/sessions/{request_id}")
def session_snapshot(request_id: str) -> SessionSnapshot:
    row = get_repository().get_document(request_id)
    if row is None:
        raise fastapi.HTTPException(status_code=404, detail="session not found")
    return build_session_snapshot({**row, "is_favorited": get_repository().is_favorited(request_id)})


@app.get("/sessions/{request_id}/settings")
def session_settings(request_id: str) -> SessionSettingsResponse:
    row = get_repository().get_document(request_id)
    if row is None:
        raise fastapi.HTTPException(status_code=404, detail="session not found")
    return SessionSettingsResponse(
        request_id=request_id,
        source_url=str(row["source_url"]),
        prompt_explain_text=str(row.get("prompt_explain_text", "")),
        prompt_speak_text=str(row.get("prompt_speak_text", "")),
        model_name=str(row.get("model_name", default_model_name)),
        reasoning_effort=str(row.get("reasoning_effort", "")),
    )


@app.patch("/sessions/{request_id}/settings")
def update_session_settings(request_id: str, req: SessionSettingsRequest) -> SessionSettingsResponse:
    row = get_repository().update_session_settings(
        request_id,
        prompt_explain_text=req.prompt_explain_text,
        prompt_speak_text=req.prompt_speak_text,
        model_name=req.model_name,
        reasoning_effort=req.reasoning_effort,
    )
    if row is None:
        raise fastapi.HTTPException(status_code=404, detail="session not found")
    broadcast_session_snapshot(request_id)
    cache_dir = data_dir / request_id
    for pattern in ("explain_*.txt", "explain_*.mp3", "speak_*.txt"):
        for file_path in cache_dir.glob(pattern):
            try:
                file_path.unlink()
            except FileNotFoundError:
                pass
    return SessionSettingsResponse(
        request_id=request_id,
        source_url=str(row["source_url"]),
        prompt_explain_text=str(row.get("prompt_explain_text", "")),
        prompt_speak_text=str(row.get("prompt_speak_text", "")),
        model_name=str(row.get("model_name", default_model_name)),
        reasoning_effort=str(row.get("reasoning_effort", "")),
    )


@app.get("/favorites/")
def favorites(limit: int = 20) -> FavoriteListResponse:
    repository = get_repository()
    return FavoriteListResponse(
        items=[build_favorite_item(row) for row in repository.list_favorites(limit=limit)]
    )


@app.post("/favorites/{request_id}/toggle")
def favorite_toggle(request_id: str, req: FavoriteToggleRequest | None = None) -> FavoriteToggleResponse:
    try:
        page_num = req.page_num if req is not None else None
        favorited = get_repository().toggle_favorite(request_id, page_num=page_num)
    except KeyError as error:
        raise fastapi.HTTPException(status_code=404, detail="session not found") from error
    current = get_repository().get_document(request_id)
    resolved_page_num = int(page_num if page_num is not None else current["current_page"]) if current is not None else 1
    broadcast_favorite_changed(request_id, resolved_page_num, favorited)
    return FavoriteToggleResponse(request_id=request_id, page_num=resolved_page_num, favorited=favorited)


@app.post("/sessions/{request_id}/favorite")
def favorite_toggle_session(request_id: str, req: FavoriteToggleRequest | None = None) -> FavoriteToggleResponse:
    return favorite_toggle(request_id, req)


@app.post("/sessions/{request_id}/favorite/toggle")
def favorite_toggle_session_alias(request_id: str, req: FavoriteToggleRequest | None = None) -> FavoriteToggleResponse:
    return favorite_toggle(request_id, req)


class PlaybackStartedRequest(BaseModel):
    page_num: int


@app.post("/sessions/{request_id}/playback")
def session_playback_started(request_id: str, req: PlaybackStartedRequest) -> dict[str, object]:
    broadcast_playback_started(request_id, req.page_num)
    return {"request_id": request_id, "page_num": req.page_num}


@app.post("/sessions/{request_id}/playback/stop")
def session_playback_stopped(request_id: str) -> dict[str, object]:
    broadcast_playback_stopped(request_id)
    return {"request_id": request_id}


@app.websocket("/sessions/ws")
async def session_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    request_id = websocket.query_params.get("request_id")
    if not request_id:
        await websocket.close(code=1008)
        return

    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=128)
    loop = asyncio.get_running_loop()
    session_broadcast_hub.subscribe(request_id, queue, loop)
    repository = get_repository()
    snapshot_row = repository.get_document(request_id)
    if snapshot_row is not None:
        await queue.put(
            {
                "type": "session_snapshot",
                "request_id": request_id,
                "source_url": snapshot_row.get("source_url", ""),
                "page_num": snapshot_row.get("page_num"),
                "current_page": snapshot_row.get("current_page"),
                "is_favorited": repository.is_favorited(request_id),
                "prompt_explain_text": snapshot_row.get("prompt_explain_text", ""),
                "prompt_speak_text": snapshot_row.get("prompt_speak_text", ""),
                "model_name": snapshot_row.get("model_name", default_model_name),
                "reasoning_effort": snapshot_row.get("reasoning_effort", ""),
                "total_generation_count": snapshot_row.get("total_generation_count", 0),
                "total_generation_elapsed_ms": snapshot_row.get("total_generation_elapsed_ms", 0),
                "total_input_tokens": snapshot_row.get("total_input_tokens", 0),
                "total_output_tokens": snapshot_row.get("total_output_tokens", 0),
                "total_cost_usd": snapshot_row.get("total_cost_usd", 0.0),
            }
        )

    async def sender() -> None:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)

    sender_task = asyncio.create_task(sender())
    try:
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        session_broadcast_hub.unsubscribe(request_id, queue)
        sender_task.cancel()


class ImageRequest(BaseModel):
    request_id: str
    page: int


@app.post("/image/")
def image(req: ImageRequest) -> fastapi.responses.FileResponse:
    work_dir = data_dir / req.request_id
    image_path = work_dir / "images" / f"{req.page:04d}.png"
    return fastapi.responses.FileResponse(image_path)


class ExplainRequest(BaseModel):
    request_id: str
    page: int


class ExplainResponse(BaseModel):
    explanation: str
    speech_text: str = ""
    audio_status: Literal["ready", "failed"] = "ready"
    audio_error: str | None = None


class GenerationResult(BaseModel):
    explanation: str
    speech_text: str = ""
    audio_status: Literal["ready", "failed"] = "ready"
    audio_error: str | None = None


speaker = VoiceVoxSpeaker(
    speaker_id="1",
    url=voicevox_url,
)


@app.post("/explain/")
def explain(req: ExplainRequest) -> ExplainResponse:
    image_path = data_dir / req.request_id / "images" / f"{req.page:04d}.png"
    cache_path = data_dir / req.request_id / f"explain_{req.page:04d}.txt"
    audio_path = data_dir / req.request_id / f"explain_{req.page:04d}.mp3"

    audio_status: Literal["ready", "failed"] = "ready"
    audio_error: str | None = None
    generation_result = generate_explanation_through_queue(
        f"{req.request_id}:{req.page:04d}",
        (image_path, cache_path, audio_path),
    )
    explanation = generation_result.explanation
    audio_status = generation_result.audio_status
    audio_error = generation_result.audio_error
    record_session_result(
        req.request_id,
        req.page,
        explanation,
        generation_result.speech_text,
        audio_status=audio_status,
        audio_error=audio_error,
    )
    get_repository().update_current_page(req.request_id, req.page)
    broadcast_session_page_updated(req.request_id, req.page)

    next_image_path = data_dir / req.request_id / "images" / f"{req.page + 1:04d}.png"
    if next_image_path.exists():
        next_cache_path = data_dir / req.request_id / f"explain_{req.page + 1:04d}.txt"
        next_audio_path = data_dir / req.request_id / f"explain_{req.page + 1:04d}.mp3"
        if not (next_cache_path.exists() and next_audio_path.exists()):
            _ = reserve_generation(
                f"{req.request_id}:{req.page + 1:04d}",
                (next_image_path, next_cache_path, next_audio_path),
                priority=10,
            )

    return ExplainResponse(
        explanation=explanation,
        speech_text=generation_result.speech_text,
        audio_status=audio_status,
        audio_error=audio_error,
    )


def generate_explanation(image_path: Path, prompt_explain_text: str, model_name: str, reasoning_effort: str):
    image = Image.open(image_path)
    image_type = "png"
    image_content = to_image_content(image, image_type)
    response = run_gpt(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_explain_text,
                    },
                    image_content,
                ],
            }
        ],
        json_mode=False,
        model=model_name,
        reasoning_effort=reasoning_effort or default_reasoning_effort,
    )
    return response


def generate_speech_text(explanation: str, prompt_speak_text: str, model_name: str, reasoning_effort: str):
    response = run_gpt(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_speak_text,
                    },
                    {
                        "type": "text",
                        "text": explanation,
                    },
                ],
            }
        ],
        json_mode=False,
        model=model_name,
        reasoning_effort=reasoning_effort or default_reasoning_effort,
    )
    return response


def worker(
    fn,
    scheduler: GenerationTaskScheduler["GenerationResult"],
) -> None:
    while True:
        job = scheduler.get_next_job()
        if job is None:
            continue
        result = fn(job.task_id, *job.args, force=job.force, priority=job.priority)
        scheduler.complete(job.task_id, result)


def generation_task(
    task_id: str,
    image_path: Path,
    cache_path: Path,
    audio_path: Path,
    *,
    force: bool = False,
    priority: int = 0,
) -> GenerationResult | Exception:
    print(f"[INFO] Generating explanation for {image_path}", file=sys.stderr)
    should_announce_generation = force or priority <= 0
    started_generation = False
    started_at = time.perf_counter()
    if ":" in task_id:
        request_id, page_text = task_id.split(":", 1)
        page = int(page_text)
    else:
        request_id = task_id
        page = 1
    try:
        speech_path = cache_path.with_name(f"speak_{page:04d}.txt")
        prompt_explain_text = resolve_document_prompt_explain_text(request_id)
        prompt_speak_text = resolve_document_prompt_speak_text(request_id)
        model_name = resolve_document_model_name(request_id)
        reasoning_effort = resolve_document_reasoning_effort(request_id)
        cached_result = get_repository().get_result(
            request_id,
            page,
            prompt_explain_text=prompt_explain_text,
            prompt_speak_text=prompt_speak_text,
            model_name=model_name,
        )
        cached_speech_text = ""
        if cached_result is not None:
            cached_speech_text = str(cached_result.get("speech_text") or "")
        if (
            not force
            and cached_result is not None
            and cache_path.exists()
            and speech_path.exists()
            and audio_path.exists()
        ):
            explanation = cache_path.read_text()
            print(f"[INFO] Reusing cached explanation for {image_path}", file=sys.stderr)
            return GenerationResult(
                explanation=explanation,
                speech_text=speech_path.read_text(),
                audio_status="ready",
                audio_error=None,
            )
        if should_announce_generation:
            started_generation = True
            broadcast_generation_started(request_id, page)
        try:
            gpt_result = generate_explanation(
                image_path,
                prompt_explain_text,
                model_name,
                reasoning_effort,
            )
            explanation = gpt_result.content
            cache_path.write_text(explanation)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to generate explanation for {image_path}: {exc}", file=sys.stderr)
            return exc
        try:
            gpt_speech_result = generate_speech_text(
                explanation,
                prompt_speak_text,
                model_name,
                reasoning_effort,
            )
            speech_text = gpt_speech_result.content
            speech_path.write_text(speech_text)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to generate speech text for {image_path}: {exc}", file=sys.stderr)
            print(f"[ERROR] Explanation was: {explanation}", file=sys.stderr)
            result_row = get_repository().upsert_result(
                request_id,
                page,
                explanation,
                speech_text=cached_speech_text,
                prompt_explain_text=prompt_explain_text,
                prompt_speak_text=prompt_speak_text,
                model_name=model_name,
                audio_status="failed",
                audio_error=str(exc),
            )
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            record_session_usage(
                request_id,
                page,
                result_id=str(result_row["result_id"]) if result_row is not None else None,
                kind="explanation",
                elapsed_ms=elapsed_ms,
                input_tokens=gpt_result.input_tokens,
                output_tokens=gpt_result.output_tokens,
                model_name=model_name,
                detail={
                    "kind": "explanation",
                    "audio_status": "failed",
                    "audio_error": str(exc),
                },
            )
            record_session_usage(
                request_id,
                page,
                result_id=str(result_row["result_id"]) if result_row is not None else None,
                kind="speech",
                elapsed_ms=0,
                input_tokens=0,
                output_tokens=0,
                model_name=model_name,
                detail={
                    "kind": "speech",
                    "audio_status": "failed",
                    "audio_error": str(exc),
                },
            )
            broadcast_session_snapshot(request_id)
            return GenerationResult(
                explanation=explanation,
                speech_text=cached_speech_text,
                audio_status="failed",
                audio_error=str(exc),
            )
        try:
            text_to_wav(speech_text, speaker, audio_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to generate audio for {image_path}: {exc}", file=sys.stderr)
            print(f"[ERROR] Speech text was: {speech_text}", file=sys.stderr)
            print(f"[INFO] Continuing without audio for {image_path}", file=sys.stderr)
            result_row = get_repository().upsert_result(
                request_id,
                page,
                explanation,
                speech_text=speech_text,
                prompt_explain_text=prompt_explain_text,
                prompt_speak_text=prompt_speak_text,
                model_name=model_name,
                audio_status="failed",
                audio_error=str(exc),
            )
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            record_session_usage(
                request_id,
                page,
                result_id=str(result_row["result_id"]) if result_row is not None else None,
                kind="explanation",
                elapsed_ms=elapsed_ms,
                input_tokens=gpt_result.input_tokens,
                output_tokens=gpt_result.output_tokens,
                model_name=model_name,
                detail={
                    "kind": "explanation",
                    "audio_status": "failed",
                    "audio_error": str(exc),
                },
            )
            record_session_usage(
                request_id,
                page,
                result_id=str(result_row["result_id"]) if result_row is not None else None,
                kind="speech",
                elapsed_ms=0,
                input_tokens=gpt_speech_result.input_tokens,
                output_tokens=gpt_speech_result.output_tokens,
                model_name=model_name,
                detail={
                    "kind": "speech",
                    "audio_status": "failed",
                    "audio_error": str(exc),
                },
            )
            broadcast_session_snapshot(request_id)
            return GenerationResult(explanation=explanation, speech_text=speech_text, audio_status="failed", audio_error=str(exc))
        print(f"[INFO] Finished generating explanation for {image_path}", file=sys.stderr)
        print(f"[INFO] Explanation saved to {cache_path}", file=sys.stderr)
        print(f"[INFO] Speech text saved to {speech_path}", file=sys.stderr)
        print(f"[INFO] Audio saved to {audio_path}", file=sys.stderr)
        result_row = get_repository().upsert_result(
            request_id,
            page,
            explanation,
            speech_text=speech_text,
            prompt_explain_text=prompt_explain_text,
            prompt_speak_text=prompt_speak_text,
            model_name=model_name,
            audio_status="ready",
            audio_error=None,
        )
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        record_session_usage(
            request_id,
            page,
            result_id=str(result_row["result_id"]) if result_row is not None else None,
            kind="explanation",
            elapsed_ms=elapsed_ms,
            input_tokens=gpt_result.input_tokens,
            output_tokens=gpt_result.output_tokens,
            model_name=model_name,
            detail={
                "kind": "explanation",
                "audio_status": "ready",
                "audio_error": None,
            },
        )
        record_session_usage(
            request_id,
            page,
            result_id=str(result_row["result_id"]) if result_row is not None else None,
            kind="speech",
            elapsed_ms=0,
            input_tokens=gpt_speech_result.input_tokens,
            output_tokens=gpt_speech_result.output_tokens,
            model_name=model_name,
            detail={
                "kind": "speech",
                "audio_status": "ready",
                "audio_error": None,
            },
        )
        broadcast_session_snapshot(request_id)
        return GenerationResult(explanation=explanation, speech_text=speech_text, audio_status="ready", audio_error=None)
    finally:
        if started_generation:
            broadcast_generation_finished(request_id, page)


threading.Thread(target=worker, args=(generation_task, generation_scheduler), daemon=True).start()


def reserve_generation(
    task_id: str,
    args: tuple[Path, Path, Path],
    *,
    priority: int = 10,
    force: bool = False,
) -> Queue[GenerationResult | Exception]:
    queue: Queue[GenerationResult | Exception] = Queue()
    generation_scheduler.reserve(task_id, args, queue, priority=priority, force=force)
    return queue


def generate_explanation_through_queue(
    task_id: str,
    args: tuple[Path, Path, Path],
    *,
    priority: int = 0,
    force: bool = False,
) -> GenerationResult:
    queue = reserve_generation(task_id, args, priority=priority, force=force)
    result = queue.get()
    if isinstance(result, Exception):
        raise result
    return result


@app.post("/audio/")
def audio(req: ExplainRequest) -> fastapi.responses.FileResponse:
    audio_path = data_dir / req.request_id / f"explain_{req.page:04d}.mp3"
    if not audio_path.exists():
        speech_path = data_dir / req.request_id / f"speak_{req.page:04d}.txt"
        if speech_path.exists():
            speech_text = speech_path.read_text()
        else:
            result_row = get_repository().get_result(
                req.request_id,
                req.page,
                prompt_explain_text=resolve_document_prompt_explain_text(req.request_id),
                prompt_speak_text=resolve_document_prompt_speak_text(req.request_id),
                model_name=resolve_document_model_name(req.request_id),
            )
            if result_row is None:
                explanation_path = data_dir / req.request_id / f"explain_{req.page:04d}.txt"
                speech_text = explanation_path.read_text()
            else:
                speech_text = str(result_row.get("speech_text") or result_row.get("explanation") or "")
        text_to_wav(speech_text, speaker, audio_path)
    return fastapi.responses.FileResponse(audio_path)


@app.post("/regenerate/")
def regenerate(req: ExplainRequest) -> ExplainResponse:
    image_path = data_dir / req.request_id / "images" / f"{req.page:04d}.png"
    cache_path = data_dir / req.request_id / f"explain_{req.page:04d}.txt"
    audio_path = data_dir / req.request_id / f"explain_{req.page:04d}.mp3"
    generation_result = generate_explanation_through_queue(
        f"{req.request_id}:{req.page:04d}",
        (image_path, cache_path, audio_path),
        force=True,
    )
    get_repository().update_current_page(req.request_id, req.page)
    broadcast_session_page_updated(req.request_id, req.page)

    return ExplainResponse(
        explanation=generation_result.explanation,
        speech_text=generation_result.speech_text,
        audio_status=generation_result.audio_status,
        audio_error=generation_result.audio_error,
    )
