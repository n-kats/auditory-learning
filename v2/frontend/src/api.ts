export type InitResponse = {
  request_id: string;
  source_url: string;
  page_num: number;
};

export type ExplainResponse = {
  explanation: string;
  speech_text: string;
  audio_status?: "ready" | "failed";
  audio_error?: string | null;
};

export type PromptResponse = {
  prompt_explain_text: string;
  prompt_speak_text: string;
  model_name: string;
  reasoning_effort: string;
};

export type SessionSummary = {
  request_id: string;
  source_url: string;
  page_num: number | null;
  current_page: number | null;
  is_favorited: boolean;
  prompt_explain_text: string;
  prompt_speak_text: string;
  model_name: string;
  reasoning_effort: string;
  total_generation_count: number;
  total_generation_elapsed_ms: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number;
  created_at: string;
  updated_at: string;
};

export type SessionSnapshot = SessionSummary;

export type FavoritePaperItem = SessionSummary & {
  favorite_page_num: number;
  favorited_at: string;
};

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
      prompt_explain_text?: string;
      prompt_speak_text?: string;
      model_name?: string;
      reasoning_effort?: string;
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
      is_favorited?: boolean;
    }
  | {
      type: "favorite_toggled";
      request_id: string;
      is_favorited: boolean;
      page_num?: number;
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
    }
  | {
      type: "playback_started";
      request_id: string;
      page_num: number;
    }
  | {
      type: "playback_stopped";
      request_id: string;
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
  const base = new URL(ensureTrailingSlash(requireApiBaseUrl()));
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  return new URL(normalizePath(path), base).toString();
}

const configuredApiBaseUrl = import.meta.env.VITE_AUDITORY_LEARNING_V2_API_BASE_URL?.trim();
export const apiBaseUrl = configuredApiBaseUrl ?? "";

function requireApiBaseUrl(): string {
  if (configuredApiBaseUrl && configuredApiBaseUrl.length > 0) {
    return configuredApiBaseUrl;
  }
  throw new Error("VITE_AUDITORY_LEARNING_V2_API_BASE_URL is required");
}

function ensureTrailingSlash(baseUrl: string): string {
  return baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
}

export function normalizePath(path: string): string {
  return path.startsWith("/") ? path.slice(1) : path;
}

export function joinApiUrl(baseUrl: string, path: string): string {
  return new URL(normalizePath(path), ensureTrailingSlash(baseUrl)).toString();
}

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
  const response = await fetch(joinApiUrl(requireApiBaseUrl(), path), {
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

async function requestFormData<T>(path: string, formData: FormData): Promise<T> {
  const response = await fetch(joinApiUrl(requireApiBaseUrl(), path), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    return parseError(response);
  }

  return (await response.json()) as T;
}

async function requestPatchJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(joinApiUrl(requireApiBaseUrl(), path), {
    method: "PATCH",
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
  const response = await fetch(joinApiUrl(requireApiBaseUrl(), path));
  if (!response.ok) {
    return parseError(response);
  }
  return (await response.json()) as T;
}

async function requestBlob(path: string, payload: unknown): Promise<Blob> {
  const response = await fetch(joinApiUrl(requireApiBaseUrl(), path), {
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
  promptSpeakText?: string,
  modelName?: string,
  reasoningEffort?: string,
): Promise<InitResponse> {
  return requestJson<InitResponse>("/init/", {
    url,
    prompt_explain_text: promptExplainText,
    prompt_speak_text: promptSpeakText,
    model_name: modelName,
    reasoning_effort: reasoningEffort,
  });
}

export async function initDocumentFromUpload(
  file: File,
  promptExplainText?: string,
  promptSpeakText?: string,
  modelName?: string,
  reasoningEffort?: string,
): Promise<InitResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (promptExplainText !== undefined) formData.append("prompt_explain_text", promptExplainText);
  if (promptSpeakText !== undefined) formData.append("prompt_speak_text", promptSpeakText);
  if (modelName !== undefined) formData.append("model_name", modelName);
  if (reasoningEffort !== undefined) formData.append("reasoning_effort", reasoningEffort);
  return requestFormData<InitResponse>("/init/upload/", formData);
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

export async function toggleFavorite(
  requestId: string,
  pageNum?: number,
): Promise<{ request_id: string; page_num: number; favorited: boolean }> {
  return requestJson<{ request_id: string; page_num: number; favorited: boolean }>(`/sessions/${encodeURIComponent(requestId)}/favorite`, {
    page_num: pageNum,
  });
}

export type SessionSettingsResponse = {
  request_id: string;
  source_url: string;
  prompt_explain_text: string;
  prompt_speak_text: string;
  model_name: string;
  reasoning_effort: string;
};

export async function fetchSessionSettings(requestId: string): Promise<SessionSettingsResponse> {
  return requestGetJson<SessionSettingsResponse>(`/sessions/${encodeURIComponent(requestId)}/settings`);
}

export async function notifyPlaybackStarted(requestId: string, pageNum: number): Promise<void> {
  await requestJson<unknown>(`/sessions/${encodeURIComponent(requestId)}/playback`, { page_num: pageNum });
}

export async function notifyPlaybackStopped(requestId: string): Promise<void> {
  await requestJson<unknown>(`/sessions/${encodeURIComponent(requestId)}/playback/stop`, {});
}

export async function updateSessionSettings(
  requestId: string,
  payload: { prompt_explain_text?: string | null; prompt_speak_text?: string | null; model_name?: string | null; reasoning_effort?: string | null },
): Promise<SessionSettingsResponse> {
  return requestPatchJson<SessionSettingsResponse>(`/sessions/${encodeURIComponent(requestId)}/settings`, payload);
}
