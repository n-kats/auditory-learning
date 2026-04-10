from __future__ import annotations

import json
import logging
from pathlib import Path

from quick_auditory_learning.db import (
    ImportResult,
    JsonlImportState,
    connection,
    ensure_schema,
    jsonl_import_state,
    upsert_jsonl_import_state,
    upsert_paper,
)
from quick_auditory_learning.models import Paper

logger = logging.getLogger(__name__)


def normalize_paper_record(payload: dict[str, object]) -> dict[str, object]:
    paper = Paper.model_validate(payload)
    raw = dict(payload)
    if "journal-ref" not in raw and paper.journal_ref is not None:
        raw["journal-ref"] = paper.journal_ref
    if "report-no" not in raw and paper.report_no is not None:
        raw["report-no"] = paper.report_no
    return {
        "id": paper.id,
        "submitter": paper.submitter,
        "authors": paper.authors,
        "title": paper.title,
        "comments": paper.comments,
        "journal_ref": paper.journal_ref,
        "doi": paper.doi,
        "abstract": paper.abstract,
        "report_no": paper.report_no,
        "categories": json.dumps(paper.categories),
        "versions": json.dumps(paper.versions),
        "raw": json.dumps(raw),
    }


def count_jsonl_records(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total


def import_jsonl(path: Path) -> ImportResult:
    imported = 0
    updated = 0
    total = count_jsonl_records(path)
    next_percent = 1
    logger.info("jsonl import started: path=%s", path)
    logger.info("jsonl import total: path=%s total=%s", path, total)
    with connection() as conn:
        ensure_schema(conn)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                upsert_paper(conn, normalize_paper_record(payload))
                imported += 1
                updated += 1
                if total > 0:
                    percent = int(imported * 100 / total)
                    if percent >= next_percent:
                        logger.info(
                            "jsonl import progress: path=%s percent=%s imported=%s updated=%s total=%s",
                            path,
                            percent,
                            imported,
                            updated,
                            total,
                        )
                        next_percent = percent + 1
                elif imported % 10000 == 0:
                    logger.info("jsonl import progress: path=%s imported=%s updated=%s", path, imported, updated)
        stat = path.stat()
        upsert_jsonl_import_state(conn, str(path), stat.st_mtime_ns, stat.st_size)
    logger.info("jsonl import finished: path=%s imported=%s updated=%s total=%s", path, imported, updated, total)
    return ImportResult(imported=imported, updated=updated)


def jsonl_import_is_stale(
    state: JsonlImportState | None,
    source_mtime_ns: int,
    source_size: int,
) -> bool:
    if state is None:
        return True
    return state.source_mtime_ns != source_mtime_ns or state.source_size != source_size


def jsonl_needs_import(path: Path) -> bool:
    if not path.exists():
        return False
    stat = path.stat()
    with connection() as conn:
        ensure_schema(conn)
        state = jsonl_import_state(conn, str(path))
    return jsonl_import_is_stale(state, stat.st_mtime_ns, stat.st_size)


def sync_jsonl(path: Path) -> ImportResult | None:
    if not path.exists():
        logger.warning("jsonl sync skipped: source not found path=%s", path)
        return None
    if not jsonl_needs_import(path):
        logger.info("jsonl sync skipped: up to date path=%s", path)
        return None
    return import_jsonl(path)
