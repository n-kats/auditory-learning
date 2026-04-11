from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from quick_auditory_learning.settings import settings


class Paper(BaseModel):
    id: str
    submitter: str | None = None
    authors: str | None = None
    title: str
    comments: str | None = None
    journal_ref: str | None = Field(default=None, alias="journal-ref")
    doi: str | None = None
    abstract: str
    report_no: str | None = Field(default=None, alias="report-no")
    categories: list[str] = Field(default_factory=list)
    versions: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class EmbeddingModel(BaseModel):
    model_name: str
    model_version: str
    dimension: int
    table_name: str


class SearchRequest(BaseModel):
    query: str
    model_name: str = Field(default_factory=lambda: settings.embedding_model_name)
    include_old_vectors: bool = False
    exclude_paper_ids: list[str] = Field(default_factory=list)
    limit: int = 20
    route1_weight: float = 0.55
    route2_weight: float = 0.45
    seed: int | None = None


class SearchHit(BaseModel):
    paper: Paper
    score: float
    route1_score: float = 0.0
    route2_score: float = 0.0
    source_modes: list[str] = Field(default_factory=list)


class SearchCandidate(BaseModel):
    paper: Paper
    paper_id: str
    title: str
    score: float = 0.0
    reason: str
    source_modes: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    rejected_candidates: list[SearchCandidate] = Field(default_factory=list)
    fallback_used: bool = False


class FavoriteToggleResponse(BaseModel):
    paper_id: str
    favorited: bool


class FavoritePaperItem(BaseModel):
    paper_id: str
    title: str


class FavoriteListResponse(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)
    items: list[FavoritePaperItem] = Field(default_factory=list)


class HistoryTransition(BaseModel):
    from_paper_id: str | None = None
    to_paper_id: str


class ExplanationResponse(BaseModel):
    paper_id: str
    title: str
    explanation: str
    audio_url: str
    audio_urls: list[str] = Field(default_factory=list)
    audio_duration_ms: int | None = None
    notices: list[str] = Field(default_factory=list)


class PaperResolveRequest(BaseModel):
    source_url: str


class PaperResolveResponse(BaseModel):
    paper: Paper
    source: str


class PaperMemoUpdateRequest(BaseModel):
    memo: str


class PaperMemoResponse(BaseModel):
    paper_id: str
    memo: str
    updated_at: datetime | None = None


class SessionStartRequest(BaseModel):
    source_url: str
    model_name: str = Field(default_factory=lambda: settings.embedding_model_name)
    include_old_vectors: bool = False
    limit: int = 20
    route1_weight: float = 0.55
    route2_weight: float = 0.45
    seed: int | None = None


class SessionResumeRequest(BaseModel):
    session_id: str
    last_event_seq: int = 0


class SessionNextRequest(BaseModel):
    session_id: str


class SessionStopRequest(BaseModel):
    session_id: str


class SessionClientMessage(BaseModel):
    type: Literal["start", "resume", "next", "set_next_candidate", "stop", "regenerate", "playback_started"]
    source_url: str | None = None
    session_id: str | None = None
    paper_id: str | None = None
    last_event_seq: int = 0
    model_name: str = Field(default_factory=lambda: settings.embedding_model_name)
    include_old_vectors: bool = False
    limit: int = 20
    route1_weight: float = 0.55
    route2_weight: float = 0.45
    seed: int | None = None
    search_modes: list[str] = Field(default_factory=list)


class SessionEvent(BaseModel):
    seq: int
    type: str
    payload: dict[str, Any]


class SessionSnapshot(BaseModel):
    session_id: str
    status: str
    root_source_url: str
    root_paper_id: str
    current_paper_id: str
    next_paper_id: str | None = None
    next_event_seq: int
    config: dict[str, Any]


class SessionListItem(BaseModel):
    session_id: str
    status: str
    session_websocket_connections: int = 0
    root_source_url: str
    root_paper_id: str
    root_paper_title: str | None = None
    current_paper_id: str
    current_paper_title: str | None = None
    next_event_seq: int
    config: dict[str, Any]
    started_at: datetime
    updated_at: datetime
    total_generation_elapsed_ms: int = 0
    total_wall_elapsed_ms: int = 0
    total_generation_cost_usd: float = 0.0


class SessionListResponse(BaseModel):
    sessions: list[SessionListItem]


class SessionCostItem(BaseModel):
    kind: str
    elapsed_ms: int | None = None
    elapsed_ms_without_prefetch: int | None = None
    estimated_cost_usd: float | None = None
    status: str = "calculated"


class SessionCostsResponse(BaseModel):
    session_id: str
    total_elapsed_ms: int | None = None
    total_wall_elapsed_ms: int = 0
    total_cost_usd: float | None = None
    is_final: bool = False
    total_elapsed_ms_without_prefetch: int | None = None
    total_cost_usd_without_prefetch: float | None = None
    audio_duration_ms: int | None = None
    items: list[SessionCostItem] = Field(default_factory=list)
