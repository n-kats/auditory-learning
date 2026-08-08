from __future__ import annotations

from dataclasses import dataclass
import re
from datetime import UTC, datetime
from time import perf_counter
from unicodedata import normalize

from openai import OpenAI

from quick_auditory_learning.settings import settings

FOLLOWUP_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "analysis",
    "approach",
    "based",
    "between",
    "both",
    "data",
    "during",
    "effect",
    "effects",
    "from",
    "into",
    "their",
    "there",
    "these",
    "this",
    "those",
    "using",
    "with",
    "within",
}


@dataclass(frozen=True)
class SearchKeywordResult:
    search_keyword: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class SearchQueryResult:
    search_query: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None


def extract_followup_tokens(text: str) -> list[str]:
    return [
        token
        for token in normalize("NFKC", text.lower()).replace("/", " ").replace("-", " ").split()
        if token and len(token) >= 4 and token not in FOLLOWUP_STOPWORDS
    ]


def build_followup_query(title: str, abstract: str) -> str:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in [*extract_followup_tokens(title), *extract_followup_tokens(abstract)]:
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= 8:
            break
    return " ".join(tokens) or title.strip() or abstract.strip()


def _normalize_keyword_candidates(text: str) -> list[str]:
    raw_tokens = [
        candidate.strip(" \t\r\n\"'`。、，;:()[]{}")
        for candidate in normalize("NFKC", text).replace("\n", ",").split(",")
    ]
    seen: set[str] = set()
    tokens: list[str] = []
    for token in raw_tokens:
        lowered = token.lower()
        if not lowered or len(lowered) < 3:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        tokens.append(lowered)
        if len(tokens) >= 8:
            break
    return tokens


def _normalize_search_query(text: str) -> str:
    normalized = normalize("NFKC", text).strip()
    normalized = normalized.replace("\r", " ").replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def generate_search_keyword(
    client: OpenAI,
    model_name: str,
    title: str,
    abstract: str,
) -> SearchKeywordResult:
    started_at = datetime.now(UTC)
    started_perf = perf_counter()
    prompt = (
        "次の論文から、検索に使う英語キーワードを 8 個まで作ってください。\n"
        "1 行かカンマ区切りで、短い名詞句だけを返してください。\n"
        "一般的すぎる語は避け、技術用語や対象分野の語を優先してください。\n"
        "説明文や箇条書きの番号は不要です。\n"
        "タイトル: {title}\n"
        "アブスト: {abstract}"
    ).format(title=title, abstract=abstract)
    response = client.responses.create(
        model=model_name,
        input=prompt,
        reasoning={"effort": settings.reasoning_effort},
        store=False,
    )
    text = response.output_text.strip()
    keywords = _normalize_keyword_candidates(text)
    if not keywords:
        keywords = _normalize_keyword_candidates(build_followup_query(title, abstract))
    usage = getattr(response, "usage", None)
    input_tokens = None
    output_tokens = None
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
    finished_at = datetime.now(UTC)
    return SearchKeywordResult(
        search_keyword=" ".join(keywords) if keywords else build_followup_query(title, abstract),
        started_at=started_at,
        finished_at=finished_at,
        elapsed_ms=int((perf_counter() - started_perf) * 1000),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def generate_fulltext_query(
    client: OpenAI,
    model_name: str,
    title: str,
    abstract: str,
) -> SearchQueryResult:
    started_at = datetime.now(UTC)
    started_perf = perf_counter()
    prompt = (
        "次の論文から、PostgreSQL の websearch_to_tsquery('english', ...) にそのまま渡せる全文検索クエリを 1 行で作ってください。\n"
        "英語で、重要な語句を 3 から 6 個ほど使ってください。\n"
        "必要ならダブルクオートで短い句を囲んでください。\n"
        "OR は多くても 2 回までにしてください。\n"
        "説明文や箇条書きの番号は不要です。\n"
        "タイトル: {title}\n"
        "アブスト: {abstract}"
    ).format(title=title, abstract=abstract)
    response = client.responses.create(
        model=model_name,
        input=prompt,
        reasoning={"effort": settings.reasoning_effort},
        store=False,
    )
    query = _normalize_search_query(response.output_text)
    if not query:
        query = build_followup_query(title, abstract)
    usage = getattr(response, "usage", None)
    input_tokens = None
    output_tokens = None
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
    finished_at = datetime.now(UTC)
    return SearchQueryResult(
        search_query=query,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_ms=int((perf_counter() - started_perf) * 1000),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
