from __future__ import annotations

import httpx

from v2_auditory_learning.utils import voicevox_url


def test_resolve_voicevox_url_uses_custom_url_when_available(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(voicevox_url.httpx, "get", fake_get)

    resolved = voicevox_url.resolve_voicevox_url("http://voicevox:50021", "http://voicevox:50021")

    assert resolved == "http://voicevox:50021"
    assert captured["url"] == "http://voicevox:50021/version"
    assert captured["kwargs"]["timeout"] == voicevox_url.VOICEVOX_URL_CHECK_TIMEOUT


def test_resolve_voicevox_url_falls_back_when_unavailable(monkeypatch) -> None:
    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("unavailable", request=httpx.Request("GET", "http://example.com/version"))

    monkeypatch.setattr(voicevox_url.httpx, "get", fake_get)

    resolved = voicevox_url.resolve_voicevox_url("http://example.com:50021", "http://voicevox:50021")

    assert resolved == "http://voicevox:50021"
