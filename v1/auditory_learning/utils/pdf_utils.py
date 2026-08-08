from __future__ import annotations

import httpx

PDF_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/x-pdf,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}


def download_pdf(url: str) -> bytes:
    response = httpx.get(
        url,
        headers=PDF_DOWNLOAD_HEADERS,
        follow_redirects=True,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.content
