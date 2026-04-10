from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import arxiv
from psycopg import Connection

from quick_auditory_learning.db import upsert_paper
from quick_auditory_learning.models import Paper
from quick_auditory_learning.repository import get_paper

ARXIV_ABS_PDF_RE = re.compile(r"^/(?P<section>abs|pdf)/(?P<identifier>[^/?#]+)")
ARXIV_VERSION_RE = re.compile(r"^(?P<identifier>.+?)(?P<version>v\d+)?$")


def parse_arxiv_identifier(source_url: str) -> tuple[str, str]:
    source = source_url.strip()
    if not source:
        raise ValueError("arxiv source url is empty")
    if source.startswith("arXiv:"):
        source = source.removeprefix("arXiv:")
    parsed = urlparse(source)
    candidate = source
    if parsed.scheme and parsed.netloc:
        match = ARXIV_ABS_PDF_RE.match(parsed.path)
        if match:
            candidate = match.group("identifier")
        else:
            candidate = parsed.path.strip("/")
    candidate = candidate.removesuffix(".pdf")
    candidate = candidate.split("?")[0].split("#")[0]
    if not candidate:
        raise ValueError(f"invalid arxiv source url: {source_url}")
    return candidate, strip_arxiv_version(candidate)


def strip_arxiv_version(identifier: str) -> str:
    match = ARXIV_VERSION_RE.match(identifier.strip())
    if match is None:
        return identifier.strip()
    return match.group("identifier")


def fetch_arxiv_result(identifier: str) -> arxiv.Result | None:
    client = arxiv.Client(page_size=1, delay_seconds=0.5, num_retries=3)
    search = arxiv.Search(id_list=[identifier])
    try:
        return next(client.results(search))
    except StopIteration:
        return None
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to fetch arxiv paper: {identifier}") from exc


def result_to_paper_record(result: arxiv.Result, source_url: str, db_id: str, requested_identifier: str) -> dict[str, object]:
    short_id = requested_identifier or db_id
    authors = ", ".join(author.name for author in result.authors)
    categories = list(result.categories)
    versions = [short_id] if short_id != db_id else []
    raw = {
        "source_url": source_url,
        "entry_id": result.entry_id,
        "published": result.published.isoformat() if result.published is not None else None,
        "updated": result.updated.isoformat() if result.updated is not None else None,
        "primary_category": result.primary_category,
        "authors": [author.name for author in result.authors],
        "categories": categories,
        "links": [link.href for link in result.links],
    }
    return {
        "id": db_id,
        "submitter": None,
        "authors": authors or None,
        "title": result.title.strip(),
        "comments": result.comment,
        "journal_ref": result.journal_ref,
        "doi": result.doi,
        "abstract": result.summary.strip(),
        "report_no": None,
        "categories": json.dumps(categories),
        "versions": json.dumps(versions),
        "raw": json.dumps(raw),
    }


def resolve_paper_from_source(conn: Connection, source_url: str) -> tuple[Paper, str]:
    requested_identifier, db_identifier = parse_arxiv_identifier(source_url)
    paper = get_paper(conn, db_identifier)
    if paper is not None:
        return paper, "db"

    result = fetch_arxiv_result(requested_identifier) or fetch_arxiv_result(db_identifier)
    if result is None:
        raise ValueError(f"arxiv paper not found: {source_url}")

    upsert_paper(conn, result_to_paper_record(result, source_url, db_identifier, requested_identifier))
    paper = get_paper(conn, db_identifier)
    if paper is None:
        raise ValueError(f"failed to store arxiv paper: {source_url}")
    return paper, "arxiv"


def resolve_paper_from_identifier(conn: Connection, identifier: str) -> tuple[Paper, str]:
    requested_identifier = identifier.strip()
    if not requested_identifier:
        raise ValueError("arxiv paper id is empty")
    db_identifier = strip_arxiv_version(requested_identifier)
    paper = get_paper(conn, db_identifier)
    if paper is not None:
        return paper, "db"
    raise ValueError(f"paper not found in database: {identifier}")
