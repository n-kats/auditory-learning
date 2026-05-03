from __future__ import annotations

import httpx

VOICEVOX_VERSION_PATH = "/version"
VOICEVOX_URL_CHECK_TIMEOUT = 2.0


def is_voicevox_url_available(url: str) -> bool:
    normalized = url.strip().rstrip("/")
    if not normalized:
        return False

    try:
        parsed = httpx.URL(normalized)
    except Exception:  # noqa: BLE001
        return False

    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.host:
        return False
    if parsed.path not in {"", "/"}:
        return False
    if parsed.query or parsed.fragment:
        return False

    try:
        response = httpx.get(f"{normalized}{VOICEVOX_VERSION_PATH}", timeout=VOICEVOX_URL_CHECK_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError:
        return False
    return True


def resolve_voicevox_url(candidate: str | None, fallback: str) -> str:
    if candidate is None:
        return fallback

    normalized = candidate.strip().rstrip("/")
    if is_voicevox_url_available(normalized):
        return normalized
    return fallback
