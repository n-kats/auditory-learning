import {
  fetchExplanation,
  fetchPageAudio,
  fetchPageImage,
  initDocument,
  isNetworkFetchError,
  regenerateExplanation,
  type ExplainResponse,
  type InitResponse,
} from "./api";
import type { DocumentSessionFlowEvent } from "./documentSessionState";
import type { ObjectUrlStore } from "./objectUrlStore";

type LoadPageParams = {
  requestId: string;
  page: number;
  regenerate?: boolean;
  imageStore: ObjectUrlStore;
  audioStore: ObjectUrlStore;
  dispatchFlowEvent: (event: DocumentSessionFlowEvent) => void;
  sequenceRef: { current: number };
};

export type LoadPageOptions = LoadPageParams;

export async function retryInitDocumentWithBackoff(
  url: string,
  promptExplainText: string,
  promptSpeekText: string,
  modelName: string,
  onRetryReset?: () => void,
): Promise<InitResponse> {
  const maxAttempts = 10;
  let attempt = 0;
  let lastError: unknown = null;

  while (attempt < maxAttempts) {
    try {
      const response = await initDocument(url, promptExplainText, promptSpeekText, modelName);
      onRetryReset?.();
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

export async function loadDocumentPage(params: LoadPageParams): Promise<void> {
  const sequence = ++params.sequenceRef.current;
  params.dispatchFlowEvent({
    type: "page_load_started",
    request_id: params.requestId,
    page: params.page,
    regenerate: Boolean(params.regenerate),
    load_id: sequence,
  });

  try {
    const explanationPromise: Promise<ExplainResponse> = params.regenerate
      ? regenerateExplanation(params.requestId, params.page)
      : fetchExplanation(params.requestId, params.page);

    const cacheKey = `${params.requestId}:${params.page}`;

    const imagePromise = params.imageStore.get(cacheKey)
      ? Promise.resolve(params.imageStore.get(cacheKey) as string)
      : fetchPageImage(params.requestId, params.page).then((imageBlob) => params.imageStore.set(cacheKey, imageBlob));

    void imagePromise
      .then((nextImageUrl) => {
        if (sequence !== params.sequenceRef.current) {
          return;
        }
        params.dispatchFlowEvent({
          type: "page_image_loaded",
          request_id: params.requestId,
          page: params.page,
          load_id: sequence,
          image_url: nextImageUrl,
        });
      })
      .catch(() => undefined);

    void explanationPromise
      .then((explanationResponse) => {
        if (sequence !== params.sequenceRef.current) {
          return;
        }
        params.dispatchFlowEvent({
          type: "page_explanation_loaded",
          request_id: params.requestId,
          page: params.page,
          load_id: sequence,
          explanation: explanationResponse.explanation,
          audio_status: explanationResponse.audio_status === "failed" ? "failed" : "ready",
          audio_error: explanationResponse.audio_error ?? null,
        });
      })
      .catch(() => undefined);

    const nextImageUrl = await imagePromise;
    const explanationResponse = await explanationPromise;

    if (params.regenerate) {
      params.audioStore.delete(cacheKey);
    }

    let nextAudioUrl: string | null = null;
    if (explanationResponse.audio_status !== "failed") {
      nextAudioUrl = params.audioStore.get(cacheKey);
      if (!nextAudioUrl) {
        const audioBlob = await fetchPageAudio(params.requestId, params.page);
        nextAudioUrl = params.audioStore.set(cacheKey, audioBlob);
      }
    }

    if (sequence !== params.sequenceRef.current) {
      return;
    }

    params.dispatchFlowEvent({
      type: "page_load_succeeded",
      request_id: params.requestId,
      page: params.page,
      load_id: sequence,
      explanation: explanationResponse.explanation,
      image_url: nextImageUrl,
      audio_url: nextAudioUrl,
      audio_status: explanationResponse.audio_status === "failed" ? "failed" : "ready",
      audio_error: explanationResponse.audio_error ?? null,
    });
  } catch (error_) {
    if (sequence !== params.sequenceRef.current) {
      return;
    }
    params.dispatchFlowEvent({
      type: "page_load_failed",
      request_id: params.requestId,
      page: params.page,
      load_id: sequence,
      error: error_ instanceof Error ? error_.message : "Request failed",
    });
  }
}
