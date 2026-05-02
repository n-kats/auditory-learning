from __future__ import annotations

import httpx

from v2_auditory_learning.utils import pdf_utils


def test_download_pdf_uses_browser_like_headers(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        content = b"%PDF-1.7"

        def raise_for_status(self) -> None:
            return None

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(pdf_utils.httpx, "get", fake_get)

    content = pdf_utils.download_pdf("https://openreview.net/pdf?id=t2fZ2GOwAT")

    assert content == b"%PDF-1.7"
    assert captured["url"] == "https://openreview.net/pdf?id=t2fZ2GOwAT"
    assert captured["kwargs"]["follow_redirects"] is True
    assert captured["kwargs"]["timeout"] == 30.0
    assert captured["kwargs"]["headers"] == pdf_utils.PDF_DOWNLOAD_HEADERS


def test_download_pdf_raises_for_non_success_status(monkeypatch) -> None:
    request = httpx.Request("GET", "https://openreview.net/pdf?id=t2fZ2GOwAT")

    class FakeResponse:
        content = b"Forbidden"

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "forbidden",
                request=request,
                response=httpx.Response(403, request=request),
            )

    monkeypatch.setattr(pdf_utils.httpx, "get", lambda *args, **kwargs: FakeResponse())

    try:
        pdf_utils.download_pdf("https://openreview.net/pdf?id=t2fZ2GOwAT")
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 403
    else:
        raise AssertionError("expected HTTPStatusError")
