export type InitResponse = {
  request_id: string;
  page_num: number;
};

export type ExplainResponse = {
  explanation: string;
  audio_status?: "ready" | "failed";
  audio_error?: string | null;
};

export type PromptResponse = {
  prompt_explain_text: string;
  prompt_speek_text: string;
};

export type SessionSummary = {
  request_id: string;
  source_url: string;
  page_num: number | null;
  current_page: number | null;
  is_favorited: boolean;
  prompt_text?: string;
  prompt_explain_text: string;
  prompt_speek_text: string;
  model_name: string;
  total_generation_count: number;
  total_generation_elapsed_ms: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number;
  created_at: string;
  updated_at: string;
};

export type SessionSnapshot = SessionSummary;

export type FavoritePaperItem = SessionSummary;

export type FavoriteListResponse = {
  items: FavoritePaperItem[];
};

export type SessionSyncEvent =
  | {
      type: "session_snapshot";
      request_id: string;
      source_url: string;
      page_num: number | null;
      current_page: number | null;
      is_favorited: boolean;
      prompt_text?: string;
      prompt_explain_text?: string;
      prompt_speek_text?: string;
      model_name?: string;
      total_generation_count?: number;
      total_generation_elapsed_ms?: number;
      total_input_tokens?: number;
      total_output_tokens?: number;
      total_cost_usd?: number;
    }
  | {
      type: "page_updated";
      request_id: string;
      current_page: number;
    }
  | {
      type: "favorite_toggled";
      request_id: string;
      is_favorited: boolean;
    }
  | {
      type: "generation_started";
      request_id: string;
      page_num: number;
    }
  | {
      type: "generation_finished";
      request_id: string;
      page_num: number;
    };

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

export function isNetworkFetchError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }
  if (error.name === "AbortError") {
    return false;
  }
  return error.name === "TypeError" && /fetch/i.test(error.message);
}

export function toWebSocketUrl(path: string): string {
  const base = new URL(apiBaseUrl);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  return new URL(path, base).toString();
}

const configuredApiBaseUrl = import.meta.env.VITE_AUDITORY_LEARNING_V2_API_BASE_URL?.trim();
export const apiBaseUrl = configuredApiBaseUrl && configuredApiBaseUrl.length > 0 ? configuredApiBaseUrl : "http://localhost:8000";

async function parseError(response: Response): Promise<never> {
  let message = `Request failed: ${response.status}`;

  try {
    const body = (await response.json()) as ApiErrorBody;
    const detail = body.detail ?? body.message;
    if (detail && detail.length > 0) {
      message = `${message} - ${detail}`;
    }
  } catch {
    // ignore parse failure
  }

  throw new ApiError(response.status, message);
}

async function requestJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    return parseError(response);
  }

  return (await response.json()) as T;
}

async function requestGetJson<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`);
  if (!response.ok) {
    return parseError(response);
  }
  return (await response.json()) as T;
}

async function requestBlob(path: string, payload: unknown): Promise<Blob> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    return parseError(response);
  }

  return await response.blob();
}

export async function initDocument(
  url: string,
  promptExplainText?: string,
  promptSpeekText?: string,
  modelName?: string,
): Promise<InitResponse> {
  return requestJson<InitResponse>("/init/", {
    url,
    prompt_explain_text: promptExplainText,
    prompt_speek_text: promptSpeekText,
    model_name: modelName,
  });
}

export async function fetchExplanation(requestId: string, page: number): Promise<ExplainResponse> {
  return requestJson<ExplainResponse>("/explain/", { request_id: requestId, page });
}

export async function regenerateExplanation(requestId: string, page: number): Promise<ExplainResponse> {
  return requestJson<ExplainResponse>("/regenerate/", { request_id: requestId, page });
}

export async function fetchPageImage(requestId: string, page: number): Promise<Blob> {
  return requestBlob("/image/", { request_id: requestId, page });
}

export async function fetchPageAudio(requestId: string, page: number): Promise<Blob> {
  return requestBlob("/audio/", { request_id: requestId, page });
}

export async function fetchSessions(limit = 20): Promise<SessionSummary[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  return requestGetJson<SessionSummary[]>(`/sessions/?${params.toString()}`);
}

export async function fetchSessionSnapshot(requestId: string): Promise<SessionSnapshot> {
  return requestGetJson<SessionSnapshot>(`/sessions/${requestId}`);
}

export async function fetchDefaultPrompt(): Promise<PromptResponse> {
  return requestGetJson<PromptResponse>("/prompt/default");
}

export async function fetchFavorites(limit = 20): Promise<FavoriteListResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  return requestGetJson<FavoriteListResponse>(`/favorites/?${params.toString()}`);
}

export async function toggleFavorite(requestId: string): Promise<{ request_id: string; favorited: boolean }> {
  return requestJson<{ request_id: string; favorited: boolean }>(`/sessions/${encodeURIComponent(requestId)}/favorite`, {});
}

export type SessionSettingsResponse = {
  request_id: string;
  source_url: string;
  prompt_explain_text: string;
  prompt_speek_text: string;
  model_name: string;
};

export async function fetchSessionSettings(requestId: string): Promise<SessionSettingsResponse> {
  return requestGetJson<SessionSettingsResponse>(`/sessions/${encodeURIComponent(requestId)}/settings`);
}

export async function updateSessionSettings(
  requestId: string,
  payload: { prompt_explain_text?: string | null; prompt_speek_text?: string | null; model_name?: string | null },
): Promise<SessionSettingsResponse> {
  return requestJson<SessionSettingsResponse>(`/sessions/${encodeURIComponent(requestId)}/settings`, payload);
}
