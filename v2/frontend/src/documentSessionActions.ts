import { fetchSessionSnapshot, initDocument, isNetworkFetchError, toggleFavorite, type SessionSnapshot } from "./api";
import { buildErrorMessage } from "./utils/appText";
import type { DocumentSessionFlowEvent } from "./documentSessionState";
import type { ObjectUrlStore } from "./objectUrlStore";

type LoadPageInvoker = (options: {
  requestId: string;
  page: number;
  regenerate?: boolean;
}) => Promise<void>;

type CommonDeps = {
  dispatchFlowEvent: (event: DocumentSessionFlowEvent) => void;
  imageStore: ObjectUrlStore;
  audioStore: ObjectUrlStore;
  loadPage: LoadPageInvoker;
};

export async function retryInitDocumentWithBackoff(
  url: string,
  promptText: string,
  modelName: string,
): Promise<{ request_id: string; page_num: number }> {
  const maxAttempts = 10;
  let attempt = 0;
  let lastError: unknown = null;

  while (attempt < maxAttempts) {
    try {
      const response = await initDocument(url, promptText, modelName);
      return response;
    } catch (error) {
      lastError = error;
      if (!isNetworkFetchError(error) || attempt + 1 >= maxAttempts) {
        throw error;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
    }
    attempt += 1;
  }

  throw lastError instanceof Error ? lastError : new Error("initDocument failed");
}

export async function startDocumentSession(params: {
  trimmedUrl: string;
  effectivePromptText: string;
  effectiveModelName: string;
  deps: CommonDeps;
}): Promise<void> {
  const { trimmedUrl, effectivePromptText, effectiveModelName, deps } = params;
  if (!trimmedUrl.startsWith("http")) {
    deps.dispatchFlowEvent({ type: "start_failed", error: "URL は http または https で始めてください。" });
    return;
  }

  deps.imageStore.clear();
  deps.audioStore.clear();
  deps.dispatchFlowEvent({ type: "start_requested", draft_url: trimmedUrl });

  try {
    const response = await retryInitDocumentWithBackoff(trimmedUrl, effectivePromptText, effectiveModelName);
    deps.dispatchFlowEvent({
      type: "start_succeeded",
      request_id: response.request_id,
      source_url: trimmedUrl,
      page_num: response.page_num,
    });
    await deps.loadPage({ requestId: response.request_id, page: 1 });
  } catch (error_) {
    deps.dispatchFlowEvent({ type: "start_failed", error: buildErrorMessage(error_) });
  }
}

export async function resumeDocumentSession(params: {
  snapshot: SessionSnapshot;
  deps: CommonDeps;
}): Promise<void> {
  const { snapshot, deps } = params;
  const nextRequestId = snapshot.request_id;
  const nextPage = snapshot.current_page ?? 1;

  deps.dispatchFlowEvent({ type: "resume_requested", request_id: nextRequestId });
  deps.dispatchFlowEvent({ type: "resume_succeeded", snapshot });

  deps.imageStore.clear();
  deps.audioStore.clear();

  await deps.loadPage({ requestId: nextRequestId, page: nextPage });
}

export async function resumeDocumentSessionByRequestId(params: {
  requestId: string;
  deps: CommonDeps;
}): Promise<void> {
  try {
    const snapshot = await fetchSessionSnapshot(params.requestId);
    await resumeDocumentSession({ snapshot, deps: params.deps });
  } catch (error_) {
    params.deps.dispatchFlowEvent({ type: "resume_failed", error: buildErrorMessage(error_) });
  }
}

export async function moveDocumentPage(params: {
  requestId: string | null;
  page: number;
  currentPage: number;
  maxPage: number;
  loadPage: LoadPageInvoker;
}): Promise<void> {
  if (!params.requestId || params.page < 1 || params.page > params.maxPage || params.page === params.currentPage) {
    return;
  }
  await params.loadPage({ requestId: params.requestId, page: params.page });
}

export async function jumpDocumentPage(params: {
  jumpPageValue: string;
  currentPage: number;
  maxPage: number;
  requestId: string | null;
  loadPage: LoadPageInvoker;
}): Promise<void> {
  const nextPage = Number.parseInt(params.jumpPageValue, 10);
  if (!Number.isFinite(nextPage)) {
    return;
  }
  await moveDocumentPage({
    requestId: params.requestId,
    page: nextPage,
    currentPage: params.currentPage,
    maxPage: params.maxPage,
    loadPage: params.loadPage,
  });
}

export async function regenerateDocumentPage(params: {
  requestId: string | null;
  currentPage: number;
  loadPage: LoadPageInvoker;
}): Promise<void> {
  if (!params.requestId) {
    return;
  }
  await params.loadPage({ requestId: params.requestId, page: params.currentPage, regenerate: true });
}

export async function toggleDocumentFavorite(params: {
  requestId: string | null;
  dispatchFlowEvent: (event: DocumentSessionFlowEvent) => void;
}): Promise<void> {
  if (!params.requestId) {
    return;
  }
  try {
    const response = await toggleFavorite(params.requestId);
    params.dispatchFlowEvent({
      type: "favorite_toggled",
      request_id: params.requestId,
      is_favorited: response.favorited,
    });
  } catch (error_) {
    params.dispatchFlowEvent({ type: "favorite_failed", error: buildErrorMessage(error_) });
  }
}
