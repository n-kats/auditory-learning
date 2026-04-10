from __future__ import annotations

import logging
import random
from collections.abc import Callable
from collections.abc import Iterable
from datetime import UTC, datetime
from math import sqrt
from time import perf_counter

from openai import OpenAI
from psycopg import Connection

from quick_auditory_learning.costs import estimate_embedding_cost_usd
from quick_auditory_learning.db import count_papers, list_embedding_models
from quick_auditory_learning.embeddings import ensure_search_model, generate_embeddings_for_paper_ids
from quick_auditory_learning.models import EmbeddingModel, Paper, SearchCandidate, SearchHit, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)
CostRecorder = Callable[[str, datetime, datetime, int, float, dict[str, object]], None]


def normalize_tokens(query: str) -> list[str]:
    return [token for token in query.replace("/", " ").replace("-", " ").split() if token]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values) or not left_values:
        return 0.0
    dot = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = sqrt(sum(value * value for value in left_values))
    right_norm = sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def get_paper_rows(conn: Connection, paper_ids: list[str]) -> dict[str, Paper]:
    if not paper_ids:
        return {}
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, submitter, authors, title, comments, journal_ref, doi, abstract, report_no, categories, versions, raw
            FROM arxiv_papers
            WHERE id = ANY(%s)
            """,
            (paper_ids,),
        )
        rows = cursor.fetchall()
    papers: dict[str, Paper] = {}
    for row in rows:
        papers[row["id"]] = Paper(
            id=row["id"],
            submitter=row["submitter"],
            authors=row["authors"],
            title=row["title"],
            comments=row["comments"],
            journal_ref=row["journal_ref"],
            doi=row["doi"],
            abstract=row["abstract"],
            report_no=row["report_no"],
            categories=list(row["categories"]),
            versions=list(row["versions"]),
            raw=dict(row["raw"]),
        )
    return papers


def search_by_keyword(conn: Connection, query: str, limit: int, exclude_paper_ids: list[str]) -> list[tuple[str, float]]:
    if not normalize_tokens(query):
        return []
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, ts_rank_cd(search_text, websearch_to_tsquery('english', %s)) AS score
            FROM arxiv_papers
            WHERE search_text @@ websearch_to_tsquery('english', %s)
              AND NOT (id = ANY(%s))
            ORDER BY score DESC, id ASC
            LIMIT %s
            """,
            (query, query, exclude_paper_ids, limit),
        )
        rows = cursor.fetchall()
    return [(row["id"], float(row["score"])) for row in rows]


def search_by_vector_for_ids(
    conn: Connection,
    model: EmbeddingModel,
    query_embedding: list[float],
    paper_ids: list[str],
    limit: int,
    exclude_paper_ids: list[str],
) -> list[tuple[str, float]]:
    if not paper_ids:
        return []
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT paper_id, 1 - (embedding <=> %s::vector) AS score
            FROM {model.table_name}
            WHERE paper_id = ANY(%s)
              AND NOT (paper_id = ANY(%s))
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s
            """,
            (query_embedding, paper_ids, exclude_paper_ids, query_embedding, limit),
        )
        rows = cursor.fetchall()
    return [(row["paper_id"], float(row["score"])) for row in rows]


def search_by_vector(
    conn: Connection,
    model: EmbeddingModel,
    query_embedding: list[float],
    limit: int,
    exclude_paper_ids: list[str],
) -> list[tuple[str, float]]:
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT paper_id, 1 - (embedding <=> %s::vector) AS score
            FROM {model.table_name}
            WHERE NOT (paper_id = ANY(%s))
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s
            """,
            (query_embedding, exclude_paper_ids, query_embedding, limit),
        )
        rows = cursor.fetchall()
    return [(row["paper_id"], float(row["score"])) for row in rows]


def random_paper_ids(conn: Connection, limit: int, exclude_paper_ids: list[str]) -> list[str]:
    if limit <= 0:
        return []
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id
            FROM arxiv_papers
            WHERE NOT (id = ANY(%s))
            ORDER BY random()
            LIMIT %s
            """,
            (exclude_paper_ids, limit),
        )
        return [row["id"] for row in cursor.fetchall()]


def merge_scores(
    route1: dict[str, float],
    route2: dict[str, float],
    route1_weight: float,
    route2_weight: float,
    seed: int | None,
) -> list[tuple[str, float, float, float]]:
    rng = random.Random(seed)
    paper_ids = set(route1) | set(route2)
    merged: list[tuple[str, float, float, float]] = []
    for paper_id in paper_ids:
        score1 = route1.get(paper_id, 0.0)
        score2 = route2.get(paper_id, 0.0)
        score = route1_weight * score1 + route2_weight * score2 + rng.uniform(0.0, 1e-6)
        merged.append((paper_id, score, score1, score2))
    merged.sort(key=lambda item: (-item[1], item[0]))
    return merged


def search_papers(
    conn: Connection,
    client: OpenAI,
    request: SearchRequest,
    query_embedding: list[float],
    cost_recorder: CostRecorder | None = None,
) -> SearchResponse:
    search_started_at = datetime.now(UTC)
    search_started_perf = perf_counter()
    if count_papers(conn) == 0:
        raise ValueError("no papers imported")
    exclude_paper_ids = request.exclude_paper_ids
    keyword_rows = search_by_keyword(conn, request.query, max(request.limit * 4, request.limit), exclude_paper_ids)
    search_model = ensure_search_model(conn, request.model_name, len(query_embedding))
    embedding_models = [search_model]
    if request.include_old_vectors:
        embedding_models.extend(
            model for model in list_embedding_models(conn, request.model_name) if model.table_name != search_model.table_name
        )

    keyword_ids = [paper_id for paper_id, _ in keyword_rows]
    embedding_started_at = datetime.now(UTC)
    embedding_started_perf = perf_counter()
    generated_result = generate_embeddings_for_paper_ids(conn, client, search_model, keyword_ids)
    embedding_finished_at = datetime.now(UTC)
    embedding_elapsed_ms = int((perf_counter() - embedding_started_perf) * 1000)
    if cost_recorder is not None:
        cost_recorder(
            "embedding",
            embedding_started_at,
            embedding_finished_at,
            embedding_elapsed_ms,
            float(estimate_embedding_cost_usd(search_model.model_name, generated_result.input_tokens)),
            {
                "model_name": search_model.model_name,
                "generated_count": generated_result.generated_count,
                "input_tokens": generated_result.input_tokens,
                "scope": "paper_embeddings",
            },
        )

    route1: dict[str, float] = {}
    route2: dict[str, float] = {}
    query_dimension = len(query_embedding)
    for embedding_model in embedding_models:
        if embedding_model.dimension != query_dimension:
            continue
        route1_rows = search_by_vector_for_ids(
            conn,
            embedding_model,
            query_embedding,
            keyword_ids,
            max(request.limit * 4, request.limit),
            exclude_paper_ids,
        )
        vector_rows = search_by_vector(
            conn,
            embedding_model,
            query_embedding,
            max(request.limit * 4, request.limit),
            exclude_paper_ids,
        )
        for paper_id, score in route1_rows:
            route1[paper_id] = max(route1.get(paper_id, 0.0), score)
        for paper_id, score in vector_rows:
            route2[paper_id] = max(route2.get(paper_id, 0.0), score)

    fallback_used = False
    if route1 or route2:
        ranked = merge_scores(route1, route2, request.route1_weight, request.route2_weight, request.seed)
    else:
        fallback_used = True
        ranked = [
            (paper_id, 0.0, 0.0, 0.0)
            for paper_id in random_paper_ids(conn, max(request.limit * 2, request.limit), exclude_paper_ids)
        ]

    selected_rows = ranked[: request.limit]
    rejected_rows = ranked[request.limit : request.limit * 2]
    all_candidate_ids = [paper_id for paper_id, *_ in selected_rows + rejected_rows]
    papers = get_paper_rows(conn, all_candidate_ids)

    hits: list[SearchHit] = []
    for paper_id, score, route1_score, route2_score in selected_rows:
        paper = papers.get(paper_id)
        if paper is None:
            continue
        hits.append(
            SearchHit(
                paper=paper,
                score=score,
                route1_score=route1_score,
                route2_score=route2_score,
            )
        )

    rejected_candidates: list[SearchCandidate] = []
    for paper_id, score, _, _ in rejected_rows:
        paper = papers.get(paper_id)
        if paper is None:
            continue
        rejected_candidates.append(
            SearchCandidate(
                paper=paper,
                paper_id=paper.id,
                title=paper.title,
                score=score,
                reason="候補外",
            )
        )

    if cost_recorder is not None:
        search_finished_at = datetime.now(UTC)
        cost_recorder(
            "search",
            search_started_at,
            search_finished_at,
            int((perf_counter() - search_started_perf) * 1000) - embedding_elapsed_ms,
            0.0,
            {
                "query": request.query,
                "keyword_candidates": len(keyword_ids),
                "route1_hits": len(route1),
                "route2_hits": len(route2),
                "selected": len(hits),
                "rejected": len(rejected_candidates),
                "fallback_used": fallback_used,
            },
        )

    logger.info(
        "search summary: query=%r keyword_candidates=%s generated_embeddings=%s route1_hits=%s route2_hits=%s selected=%s rejected=%s fallback=%s",
        request.query,
        len(keyword_ids),
        generated_result.generated_count,
        len(route1),
        len(route2),
        len(hits),
        len(rejected_candidates),
        fallback_used,
    )
    return SearchResponse(hits=hits, rejected_candidates=rejected_candidates, fallback_used=fallback_used)
