from __future__ import annotations

from v2_auditory_learning.session_sync import SessionSyncEvent, SessionSyncState, apply_session_sync_event, simulate_session_sync


def test_apply_session_sync_event_tracks_page_and_favorite() -> None:
    state = SessionSyncState()

    state = apply_session_sync_event(
        state,
        SessionSyncEvent(type="session_started", request_id="session-1", current_page=1, is_favorited=False),
    )
    state = apply_session_sync_event(
        state,
        SessionSyncEvent(type="page_updated", request_id="session-1", current_page=4, is_favorited=True),
    )
    state = apply_session_sync_event(
        state,
        SessionSyncEvent(type="favorite_toggled", request_id="session-1", is_favorited=False, page_num=4),
    )

    assert state.request_id == "session-1"
    assert state.current_page == 4
    assert state.is_favorited is False
    assert state.last_event_type == "favorite_toggled"
    assert state.revision == 3


def test_apply_session_sync_event_ignores_other_session_events() -> None:
    state = SessionSyncState(request_id="session-1", current_page=2, is_favorited=True, revision=2, last_event_type="page_updated")

    next_state = apply_session_sync_event(
        state,
        SessionSyncEvent(type="page_updated", request_id="session-2", current_page=10, is_favorited=False),
    )

    assert next_state == state


def test_simulate_session_sync_combines_events() -> None:
    state = simulate_session_sync(
        [
            SessionSyncEvent(type="session_started", request_id="session-1", current_page=1, is_favorited=False),
            SessionSyncEvent(type="page_updated", request_id="session-1", current_page=3),
            SessionSyncEvent(type="favorite_toggled", request_id="session-1", is_favorited=True),
            SessionSyncEvent(type="session_stopped", request_id="session-1"),
        ]
    )

    assert state.request_id == "session-1"
    assert state.current_page == 3
    assert state.is_favorited is True
    assert state.last_event_type == "session_stopped"
    assert state.revision == 4
