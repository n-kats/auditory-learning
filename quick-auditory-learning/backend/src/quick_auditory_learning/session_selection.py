from __future__ import annotations

import random
from collections.abc import Iterable

from psycopg import Connection

from quick_auditory_learning.db import list_session_events

SEARCH_MODE_ORDER = {"simple": 0, "keyword_list": 1, "fulltext_query": 2}


def sort_search_modes(modes: set[str] | list[str]) -> list[str]:
    unique_modes = list(dict.fromkeys(str(mode) for mode in modes if str(mode)))
    return sorted(unique_modes, key=lambda mode: (SEARCH_MODE_ORDER.get(mode, len(SEARCH_MODE_ORDER)), mode))


def weighted_choice_hit(hits: list[dict[str, object]], trail_ids: set[str]) -> tuple[str, dict[str, object]]:
    eligible_hits: list[tuple[dict[str, object], int]] = []
    for rank, hit in enumerate(hits, start=1):
        paper_data = hit.get("paper") if isinstance(hit, dict) else None
        if not isinstance(paper_data, dict):
            continue
        paper_id = str(paper_data.get("id", ""))
        if not paper_id or paper_id in trail_ids:
            continue
        eligible_hits.append((paper_data, rank))
    if not eligible_hits:
        raise ValueError("next paper not found")
    weights = [1.0 / max(rank, 1) for _, rank in eligible_hits]
    chosen_paper = random.choices([paper for paper, _ in eligible_hits], weights=weights, k=1)[0]
    return str(chosen_paper.get("id", "")), chosen_paper


def latest_event_payload(events: Iterable[object], event_type: str) -> dict[str, object] | None:
    for event in reversed(list(events)):
        if getattr(event, "event_type", None) == event_type:
            payload = getattr(event, "payload", None)
            if isinstance(payload, dict):
                return payload
    return None


def pick_next_paper_from_search_payload(conn: Connection, session_id: str, payload: dict[str, object]) -> tuple[str, dict[str, object]]:
    search = payload.get("search") or {}
    hits = search.get("hits") or []
    trail_ids = set(payload.get("trail_paper_ids") or [])
    paper_id, paper_data = weighted_choice_hit([hit for hit in hits if isinstance(hit, dict)], trail_ids)
    if not paper_id:
        raise ValueError(f"next paper not found for session: {session_id}")
    return paper_id, paper_data


def restore_next_paper_id(conn: Connection, session_id: str) -> str:
    history_events = list_session_events(conn, session_id)
    last_paper_event = latest_event_payload(history_events, "paper_ready")
    if last_paper_event is None:
        raise ValueError(f"current paper event not found for session: {session_id}")
    _, next_paper_data = pick_next_paper_from_search_payload(conn, session_id, last_paper_event)
    next_paper_id = str(next_paper_data.get("id", ""))
    if not next_paper_id:
        raise ValueError(f"next paper id missing for session: {session_id}")
    return next_paper_id
