from __future__ import annotations

from psycopg import Connection

from quick_auditory_learning.models import Paper


def toggle_favorite(conn: Connection, paper_id: str) -> bool:
    with conn.cursor() as cursor:
        cursor.execute("SELECT 1 FROM favorites WHERE paper_id = %s", (paper_id,))
        exists = cursor.fetchone() is not None
        if exists:
            cursor.execute("DELETE FROM favorites WHERE paper_id = %s", (paper_id,))
            return False
        cursor.execute("INSERT INTO favorites (paper_id) VALUES (%s)", (paper_id,))
        return True


def list_favorites(conn: Connection) -> list[str]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT paper_id FROM favorites ORDER BY favorited_at DESC")
        return [row["paper_id"] for row in cursor.fetchall()]


def list_favorite_items(conn: Connection) -> list[dict[str, str]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT favorites.paper_id, arxiv_papers.title
            FROM favorites
            JOIN arxiv_papers ON arxiv_papers.id = favorites.paper_id
            ORDER BY favorites.favorited_at DESC
            """
        )
        return [
            {"paper_id": row["paper_id"], "title": row["title"]}
            for row in cursor.fetchall()
        ]


def record_transition(conn: Connection, from_paper_id: str | None, to_paper_id: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO playback_transitions (from_paper_id, to_paper_id)
            VALUES (%s, %s)
            """,
            (from_paper_id, to_paper_id),
        )


def recent_transitions(conn: Connection, limit: int = 20) -> list[dict[str, str | None]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT from_paper_id, to_paper_id
            FROM playback_transitions
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [
            {"from_paper_id": row["from_paper_id"], "to_paper_id": row["to_paper_id"]}
            for row in cursor.fetchall()
        ]


def get_paper(conn: Connection, paper_id: str) -> Paper | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, submitter, authors, title, comments, journal_ref, doi, abstract, report_no, categories, versions, raw
            FROM arxiv_papers
            WHERE id = %s
            """,
            (paper_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return Paper(
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


def get_paper_memo(conn: Connection, paper_id: str) -> dict[str, object] | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT paper_id, memo, updated_at
            FROM paper_memos
            WHERE paper_id = %s
            """,
            (paper_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return {
        "paper_id": row["paper_id"],
        "memo": row["memo"],
        "updated_at": row["updated_at"],
    }


def upsert_paper_memo(conn: Connection, paper_id: str, memo: str) -> dict[str, object]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO paper_memos (paper_id, memo, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (paper_id) DO UPDATE
            SET memo = EXCLUDED.memo,
                updated_at = NOW()
            RETURNING paper_id, memo, updated_at
            """,
            (paper_id, memo),
        )
        row = cursor.fetchone()
    return {
        "paper_id": row["paper_id"],
        "memo": row["memo"],
        "updated_at": row["updated_at"],
    }
