export type InitResponse = {
  request_id: string;
  page_num: number;
};

export type ExplainResponse = {
  explanation: string;
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

export async function initDocument(url: string): Promise<InitResponse> {
  return requestJson<InitResponse>("/init/", { url });
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
