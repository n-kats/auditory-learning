from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import dataclass
from typing import Iterable

from openai import OpenAI
from pgvector import Vector
from psycopg import Connection

from quick_auditory_learning.db import ensure_embedding_table, get_embedding_model, store_embedding
from quick_auditory_learning.models import EmbeddingModel


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: list[float]
    input_tokens: int | None = None


@dataclass(frozen=True)
class EmbeddingBatchResult:
    generated_count: int
    input_tokens: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None


def make_client(api_key: str | None = None) -> OpenAI:
    return OpenAI(api_key=api_key)


def _response_input_tokens(response) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    for attr in ("input_tokens", "prompt_tokens"):
        value = getattr(usage, attr, None)
        if isinstance(value, int):
            return value
    return None


def embed_text(client: OpenAI, model: str, text: str) -> EmbeddingResult:
    response = client.embeddings.create(model=model, input=text)
    return EmbeddingResult(embedding=list(response.data[0].embedding), input_tokens=_response_input_tokens(response))


def to_vector(values: Iterable[float]) -> Vector:
    return Vector(list(values))


def make_embedding_model_version() -> str:
    return datetime.now(UTC).strftime("generated-%Y%m%dT%H%M%S%fZ")


def ensure_model(conn: Connection, model_name: str, model_version: str, dimension: int) -> EmbeddingModel:
    table_name = ensure_embedding_table(conn, model_name, model_version, dimension)
    return EmbeddingModel(
        model_name=model_name,
        model_version=model_version,
        dimension=dimension,
        table_name=table_name,
    )


def ensure_search_model(conn: Connection, model_name: str, dimension: int) -> EmbeddingModel:
    existing_model = get_embedding_model(conn, model_name, None)
    if existing_model is not None and existing_model.dimension == dimension:
        return existing_model
    model_version = make_embedding_model_version()
    return ensure_model(conn, model_name, model_version, dimension)


def generate_embeddings_for_paper_ids(
    conn: Connection,
    client: OpenAI,
    model: EmbeddingModel,
    paper_ids: list[str],
) -> EmbeddingBatchResult:
    started_at = datetime.now(UTC)
    if not paper_ids:
        finished_at = datetime.now(UTC)
        return EmbeddingBatchResult(generated_count=0, input_tokens=0, started_at=started_at, finished_at=finished_at)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT paper_id
            FROM {model.table_name}
            WHERE paper_id = ANY(%s)
            """,
            (paper_ids,),
        )
        existing_ids = {row["paper_id"] for row in cursor.fetchall()}
        missing_ids = [paper_id for paper_id in paper_ids if paper_id not in existing_ids]
        if not missing_ids:
            return EmbeddingBatchResult(generated_count=0, input_tokens=0)
        cursor.execute(
            """
            SELECT id, abstract
            FROM arxiv_papers
            WHERE id = ANY(%s)
            ORDER BY id ASC
            """,
            (missing_ids,),
        )
        rows = cursor.fetchall()
        input_tokens = 0
        for row in rows:
            embedding_result = embed_text(client, model.model_name, row["abstract"])
            input_tokens += embedding_result.input_tokens or 0
            store_embedding(conn, model.table_name, row["id"], embedding_result.embedding)
    finished_at = datetime.now(UTC)
    return EmbeddingBatchResult(generated_count=len(rows), input_tokens=input_tokens, started_at=started_at, finished_at=finished_at)
