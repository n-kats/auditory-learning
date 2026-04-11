type ApiErrorBody = {
  detail?: string;
  message?: string;
};

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export type Paper = {
  id: string;
  title: string;
  abstract: string;
  authors?: string | null;
  categories: string[];
  journal_ref?: string | null;
  doi?: string | null;
};

export type SearchHit = {
  paper: Paper;
  score: number;
  route1_score: number;
  route2_score: number;
  source_modes?: string[];
};

export type SearchCandidate = {
  paper: Paper;
  paper_id: string;
  title: string;
  score: number;
  reason: string;
  source_modes?: string[];
};

export type SearchResponse = {
  hits: SearchHit[];
  rejected_candidates: SearchCandidate[];
  fallback_used: boolean;
};

export type EmbeddingModel = {
  model_name: string;
  model_version: string;
  dimension: number;
  table_name: string;
};

export type ExplanationResponse = {
  paper_id: string;
  title: string;
  explanation: string;
  audio_url: string;
  audio_urls: string[];
  audio_duration_ms?: number | null;
  notices?: string[];
};

export type PaperResolveResponse = {
  paper: Paper;
  source: string;
};

export type FavoritePaperItem = {
  paper_id: string;
  title: string;
};

export type PaperMemoResponse = {
  paper_id: string;
  memo: string;
  updated_at: string | null;
};

export type SessionCostItem = {
  kind: string;
  elapsed_ms?: number | null;
  elapsed_ms_without_prefetch?: number | null;
  estimated_cost_usd?: number | null;
  status?: string;
};

export type SessionCosts = {
  session_id: string;
  total_elapsed_ms: number;
  total_wall_elapsed_ms: number;
  total_cost_usd: number;
  is_final?: boolean;
  total_elapsed_ms_without_prefetch?: number | null;
  total_cost_usd_without_prefetch?: number | null;
  audio_duration_ms?: number | null;
  items: SessionCostItem[];
};

export type SessionSummary = {
  session_id: string;
  status: string;
  session_websocket_connections: number;
  root_source_url: string;
  root_paper_id: string;
  root_paper_title?: string | null;
  current_paper_id: string;
  current_paper_title?: string | null;
  next_event_seq: number;
  config: Record<string, unknown>;
  started_at: string;
  updated_at: string;
  total_generation_elapsed_ms: number;
  total_wall_elapsed_ms: number;
  total_generation_cost_usd: number;
};

export type SessionSnapshot = {
  session_id: string;
  status: string;
  root_source_url: string;
  root_paper_id: string;
  current_paper_id: string;
  next_paper_id?: string | null;
  next_event_seq: number;
  config: Record<string, unknown>;
};

export type SessionEventMessage = {
  type: string;
  seq?: number;
  session_id?: string;
  message?: string;
  origin?: string;
  source_url?: string;
  paper_id?: string | null;
  from_paper_id?: string | null;
  to_paper_id?: string | null;
  trail_paper_ids?: string[];
  next_paper_id?: string | null;
  simple_search_query?: string;
  followup_query?: string;
  keyword_search_query?: string;
  fulltext_search_query?: string;
  search_keyword?: string;
  search_modes?: string[];
  paper?: Paper;
  root_paper?: Paper;
  explanation?: string;
  audio_url?: string;
  audio_urls?: string[];
  audio_duration_ms?: number | null;
  notices?: string[];
  search_deferred?: boolean;
  paper_costs?: SessionCosts | null;
  session_costs?: SessionCosts | null;
  memo?: string;
  updated_at?: string | null;
  search?: {
    hits?: SearchHit[];
    rejected_candidates?: SearchCandidate[];
    fallback_used?: boolean;
  };
};

export type HealthResponse = {
  status: string;
  database_ready: boolean;
};

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

export const apiBaseUrl = configuredApiBaseUrl && configuredApiBaseUrl.length > 0 ? configuredApiBaseUrl : "http://localhost:8000";

export function toApiUrl(path: string): string {
  return `${apiBaseUrl}${path}`;
}

export function toWebSocketUrl(path: string): string {
  const base = new URL(apiBaseUrl);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  return `${base.origin}${path}`;
}

export function resolveAudioSourceUrl(baseUrl: string, audioUrl: string | undefined): string | undefined {
  if (!audioUrl) {
    return undefined;
  }
  return audioUrl.startsWith("http") ? audioUrl : `${baseUrl}${audioUrl}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const body = (await response.json()) as ApiErrorBody;
      const detail = body.detail ?? body.message;
      if (detail) {
        message = `${message} - ${detail}`;
      }
    } catch {
      // ignore parse errors and fall back to status only
    }
    throw new ApiError(response.status, message);
  }
  return (await response.json()) as T;
}

export async function searchPapers(payload: {
  query: string;
  model_name: string;
  include_old_vectors: boolean;
  exclude_paper_ids: string[];
  limit: number;
  route1_weight: number;
  route2_weight: number;
  seed: number | null;
}): Promise<SearchResponse> {
  return request<SearchResponse>("/search", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getHealth(): Promise<HealthResponse> {
  return request("/health");
}

export async function toggleFavorite(paperId: string): Promise<{ paper_id: string; favorited: boolean }> {
  return request(`/favorites/${encodeURIComponent(paperId)}/toggle`, {
    method: "POST",
  });
}

export async function listFavorites(): Promise<{ items: FavoritePaperItem[] }> {
  return request("/favorites");
}

export async function recentHistory(limit = 20): Promise<{ transitions: Array<{ from_paper_id: string | null; to_paper_id: string }> }> {
  return request(`/history/recent?limit=${limit}`);
}

export async function recentSessions(limit = 10): Promise<{ sessions: SessionSummary[] }> {
  return request(`/sessions/recent?limit=${limit}`);
}

export async function getSession(sessionId: string): Promise<SessionSnapshot> {
  return request(`/sessions/${encodeURIComponent(sessionId)}`);
}

export async function getSessionEvents(sessionId: string, afterSeq = 0): Promise<{ events: SessionEventMessage[] }> {
  return request(`/sessions/${encodeURIComponent(sessionId)}/events?after_seq=${afterSeq}`);
}

export async function recordTransition(payload: {
  from_paper_id: string | null;
  to_paper_id: string;
}): Promise<{ from_paper_id: string | null; to_paper_id: string }> {
  return request("/history/transition", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function listEmbeddingModels(modelName: string): Promise<{ models: EmbeddingModel[] }> {
  return request(`/embedding-models?model_name=${encodeURIComponent(modelName)}`);
}

export async function generateExplanation(paperId: string): Promise<ExplanationResponse> {
  return request(`/explanations/${encodeURIComponent(paperId)}`, {
    method: "POST",
  });
}

export async function resolvePaper(sourceUrl: string): Promise<PaperResolveResponse> {
  return request("/papers/resolve", {
    method: "POST",
    body: JSON.stringify({ source_url: sourceUrl }),
  });
}

export async function getPaperMemo(paperId: string): Promise<PaperMemoResponse> {
  return request(`/papers/${encodeURIComponent(paperId)}/memo`);
}

export async function savePaperMemo(paperId: string, memo: string): Promise<PaperMemoResponse> {
  return request(`/papers/${encodeURIComponent(paperId)}/memo`, {
    method: "PUT",
    body: JSON.stringify({ memo }),
  });
}
