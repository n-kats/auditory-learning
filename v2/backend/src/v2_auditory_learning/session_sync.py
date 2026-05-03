from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class SessionSyncState:
    request_id: str | None = None
    current_page: int | None = None
    is_favorited: bool = False
    total_generation_count: int = 0
    total_generation_elapsed_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    last_event_type: str | None = None
    revision: int = 0


@dataclass(frozen=True)
class SessionSyncEvent:
    type: Literal["session_snapshot", "session_started", "page_updated", "favorite_toggled", "session_stopped"]
    request_id: str
    current_page: int | None = None
    is_favorited: bool | None = None
    prompt_explain_text: str | None = None
    prompt_speek_text: str | None = None
    total_generation_count: int | None = None
    total_generation_elapsed_ms: int | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_cost_usd: float | None = None


def apply_session_sync_event(state: SessionSyncState, event: SessionSyncEvent) -> SessionSyncState:
    if state.request_id is not None and event.request_id != state.request_id and event.type != "session_started":
        return state

    next_state = state
    changed = False

    if event.type in {"session_snapshot", "session_started"}:
        if next_state.request_id != event.request_id:
            next_state = replace(next_state, request_id=event.request_id)
            changed = True
        if event.current_page is not None and next_state.current_page != event.current_page:
            next_state = replace(next_state, current_page=event.current_page)
            changed = True
        if event.is_favorited is not None and next_state.is_favorited != event.is_favorited:
            next_state = replace(next_state, is_favorited=event.is_favorited)
            changed = True
        if event.total_generation_count is not None and next_state.total_generation_count != event.total_generation_count:
            next_state = replace(next_state, total_generation_count=event.total_generation_count)
            changed = True
        if (
            event.total_generation_elapsed_ms is not None
            and next_state.total_generation_elapsed_ms != event.total_generation_elapsed_ms
        ):
            next_state = replace(next_state, total_generation_elapsed_ms=event.total_generation_elapsed_ms)
            changed = True
        if event.total_input_tokens is not None and next_state.total_input_tokens != event.total_input_tokens:
            next_state = replace(next_state, total_input_tokens=event.total_input_tokens)
            changed = True
        if event.total_output_tokens is not None and next_state.total_output_tokens != event.total_output_tokens:
            next_state = replace(next_state, total_output_tokens=event.total_output_tokens)
            changed = True
        if event.total_cost_usd is not None and next_state.total_cost_usd != event.total_cost_usd:
            next_state = replace(next_state, total_cost_usd=event.total_cost_usd)
            changed = True
    elif event.type == "page_updated":
        if event.current_page is not None and next_state.current_page != event.current_page:
            next_state = replace(next_state, current_page=event.current_page)
            changed = True
    elif event.type == "favorite_toggled":
        next_is_favorited = bool(event.is_favorited)
        if next_state.is_favorited != next_is_favorited:
            next_state = replace(next_state, is_favorited=next_is_favorited)
            changed = True
    elif event.type == "session_stopped":
        changed = True

    if not changed:
        return next_state
    return replace(next_state, last_event_type=event.type, revision=next_state.revision + 1)


def simulate_session_sync(events: list[SessionSyncEvent]) -> SessionSyncState:
    state = SessionSyncState()
    for event in events:
        state = apply_session_sync_event(state, event)
    return state


@dataclass(frozen=True)
class SessionSubscriber:
    queue: asyncio.Queue[dict[str, object]]
    loop: asyncio.AbstractEventLoop


class SessionBroadcastHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[SessionSubscriber]] = {}

    def subscribe(self, request_id: str, queue: asyncio.Queue[dict[str, object]], loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._subscribers.setdefault(request_id, []).append(SessionSubscriber(queue=queue, loop=loop))

    def unsubscribe(self, request_id: str, queue: asyncio.Queue[dict[str, object]]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(request_id)
            if not subscribers:
                return
            self._subscribers[request_id] = [subscriber for subscriber in subscribers if subscriber.queue is not queue]
            if not self._subscribers[request_id]:
                del self._subscribers[request_id]

    def broadcast(self, request_id: str, payload: dict[str, object]) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(request_id, []))
        for subscriber in subscribers:
            subscriber.loop.call_soon_threadsafe(self._enqueue, subscriber.queue, payload)

    @staticmethod
    def _enqueue(queue: asyncio.Queue[dict[str, object]], payload: dict[str, object]) -> None:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass


def build_session_snapshot_event(
    request_id: str,
    current_page: int | None,
    is_favorited: bool,
    *,
    prompt_explain_text: str | None = None,
    prompt_speek_text: str | None = None,
    total_generation_count: int | None = None,
    total_generation_elapsed_ms: int | None = None,
    total_input_tokens: int | None = None,
    total_output_tokens: int | None = None,
    total_cost_usd: float | None = None,
) -> SessionSyncEvent:
    return SessionSyncEvent(
        type="session_snapshot",
        request_id=request_id,
        current_page=current_page,
        is_favorited=is_favorited,
        prompt_explain_text=prompt_explain_text,
        prompt_speek_text=prompt_speek_text,
        total_generation_count=total_generation_count,
        total_generation_elapsed_ms=total_generation_elapsed_ms,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=total_cost_usd,
    )
