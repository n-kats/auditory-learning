import type { SessionSnapshot, SessionSyncEvent } from "./api";

export type DocumentSessionFlowState = {
  draftUrl: string;
  sourceUrl: string;
  requestId: string | null;
  maxPage: number;
  currentPage: number;
  totalGenerationCount: number;
  totalGenerationElapsedMs: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCostUsd: number;
  jumpPageValue: string;
  explanation: string;
  speechText: string;
  imageUrl: string | null;
  audioUrl: string | null;
  error: string | null;
  audioStatusText: string;
  audioStatusError: string | null;
  generationStatusText: string;
  isInitializing: boolean;
  isLoadingPage: boolean;
  isRegenerating: boolean;
  isFavorited: boolean;
  activeLoadId: number | null;
};

export function createDocumentSessionFlowState(params?: Partial<DocumentSessionFlowState>): DocumentSessionFlowState {
  return {
    draftUrl: params?.draftUrl ?? "",
    sourceUrl: params?.sourceUrl ?? "",
    requestId: params?.requestId ?? null,
    maxPage: params?.maxPage ?? 1,
    currentPage: params?.currentPage ?? 1,
    totalGenerationCount: params?.totalGenerationCount ?? 0,
    totalGenerationElapsedMs: params?.totalGenerationElapsedMs ?? 0,
    totalInputTokens: params?.totalInputTokens ?? 0,
    totalOutputTokens: params?.totalOutputTokens ?? 0,
    totalCostUsd: params?.totalCostUsd ?? 0,
    jumpPageValue: params?.jumpPageValue ?? "1",
    explanation: params?.explanation ?? "",
    speechText: params?.speechText ?? "",
    imageUrl: params?.imageUrl ?? null,
    audioUrl: params?.audioUrl ?? null,
    error: params?.error ?? null,
    audioStatusText: params?.audioStatusText ?? "",
    audioStatusError: params?.audioStatusError ?? null,
    generationStatusText: params?.generationStatusText ?? "",
    isInitializing: params?.isInitializing ?? false,
    isLoadingPage: params?.isLoadingPage ?? false,
    isRegenerating: params?.isRegenerating ?? false,
    isFavorited: params?.isFavorited ?? false,
    activeLoadId: params?.activeLoadId ?? null,
  };
}

function isUploadSourceUrl(sourceUrl: string): boolean {
  return sourceUrl.startsWith("upload://");
}

export type DocumentSessionFlowEvent =
  | { type: "start_requested"; draft_url: string }
  | { type: "start_failed"; error: string }
  | { type: "start_succeeded"; request_id: string; source_url: string; page_num: number }
  | { type: "resume_requested"; request_id: string }
  | { type: "resume_failed"; error: string }
  | { type: "resume_succeeded"; snapshot: SessionSnapshot }
  | { type: "page_load_started"; request_id: string; page: number; regenerate: boolean; load_id: number }
  | { type: "page_image_loaded"; request_id: string; page: number; load_id: number; image_url: string }
  | {
      type: "page_explanation_loaded";
      request_id: string;
      page: number;
      load_id: number;
      explanation: string;
      speech_text: string;
      audio_status: "ready" | "failed";
      audio_error?: string | null;
    }
  | {
      type: "page_load_succeeded";
      request_id: string;
      page: number;
      load_id: number;
      explanation: string;
      speech_text: string;
      image_url: string;
      audio_url: string | null;
      audio_status: "ready" | "failed";
      audio_error?: string | null;
    }
  | { type: "page_load_failed"; request_id: string; page: number; load_id: number; error: string }
  | { type: "generation_started"; request_id: string; page: number }
  | { type: "generation_finished"; request_id: string; page: number }
  | { type: "favorite_toggled"; request_id: string; is_favorited: boolean; page_num?: number }
  | { type: "favorite_failed"; error: string }
  | { type: "ws_event"; event: SessionSyncEvent };

export function applyDocumentSessionFlowEvent(
  state: DocumentSessionFlowState,
  event: DocumentSessionFlowEvent,
): DocumentSessionFlowState {
  switch (event.type) {
    case "start_requested":
    case "resume_requested":
      return {
        ...state,
        error: null,
        audioStatusText: "",
        audioStatusError: null,
        generationStatusText: "",
        speechText: "",
        explanation: "",
        imageUrl: null,
        audioUrl: null,
        isInitializing: true,
        isLoadingPage: false,
        isRegenerating: false,
        activeLoadId: null,
      };
    case "start_failed":
    case "resume_failed":
    case "favorite_failed":
      return {
        ...state,
        error: event.error,
        audioStatusText: "",
        audioStatusError: null,
        generationStatusText: "",
        speechText: "",
        isInitializing: false,
        isLoadingPage: false,
        isRegenerating: false,
        activeLoadId: null,
      };
    case "start_succeeded":
      return {
        ...state,
        draftUrl: isUploadSourceUrl(event.source_url) ? "" : event.source_url,
        sourceUrl: event.source_url,
        requestId: event.request_id,
        maxPage: event.page_num,
        currentPage: 1,
        totalGenerationCount: 0,
        totalGenerationElapsedMs: 0,
        totalInputTokens: 0,
        totalOutputTokens: 0,
        totalCostUsd: 0,
        jumpPageValue: "1",
        explanation: "",
        speechText: "",
        imageUrl: null,
        audioUrl: null,
        error: null,
        audioStatusText: "",
        audioStatusError: null,
        generationStatusText: "",
        isInitializing: false,
        isLoadingPage: false,
        isRegenerating: false,
        isFavorited: false,
        activeLoadId: null,
      };
    case "resume_succeeded": {
      const nextPage = event.snapshot.current_page ?? 1;
      const nextMaxPage = event.snapshot.page_num ?? 1;
      return {
        ...state,
        draftUrl: isUploadSourceUrl(event.snapshot.source_url) ? "" : event.snapshot.source_url,
        sourceUrl: event.snapshot.source_url,
        requestId: event.snapshot.request_id,
        maxPage: nextMaxPage,
        currentPage: nextPage,
        totalGenerationCount: event.snapshot.total_generation_count ?? state.totalGenerationCount,
        totalGenerationElapsedMs: event.snapshot.total_generation_elapsed_ms ?? state.totalGenerationElapsedMs,
        totalInputTokens: event.snapshot.total_input_tokens ?? state.totalInputTokens,
        totalOutputTokens: event.snapshot.total_output_tokens ?? state.totalOutputTokens,
        totalCostUsd: event.snapshot.total_cost_usd ?? state.totalCostUsd,
        jumpPageValue: String(nextPage),
        explanation: "",
        speechText: "",
        imageUrl: null,
        audioUrl: null,
        error: null,
        audioStatusText: "",
        audioStatusError: null,
        generationStatusText: "",
        isInitializing: false,
        isLoadingPage: false,
        isRegenerating: false,
        isFavorited: event.snapshot.is_favorited,
        activeLoadId: null,
      };
    }
    case "page_load_started":
      return {
        ...state,
        requestId: event.request_id,
        currentPage: event.page,
        jumpPageValue: String(event.page),
        explanation: "",
        speechText: "",
        imageUrl: null,
        error: null,
        audioUrl: null,
        audioStatusText: "音声:確認中",
        audioStatusError: null,
        generationStatusText: "",
        isLoadingPage: !event.regenerate,
        isRegenerating: event.regenerate,
        isFavorited: event.page === state.currentPage ? state.isFavorited : false,
        activeLoadId: event.load_id,
      };
    case "page_image_loaded":
      if (state.requestId !== null && state.requestId !== event.request_id) {
        return state;
      }
      if (state.activeLoadId !== event.load_id) {
        return state;
      }
      return {
        ...state,
        requestId: event.request_id,
        currentPage: event.page,
        jumpPageValue: String(event.page),
        imageUrl: event.image_url,
      };
    case "page_explanation_loaded":
      if (state.requestId !== null && state.requestId !== event.request_id) {
        return state;
      }
      if (state.activeLoadId !== event.load_id) {
        return state;
      }
      return {
        ...state,
        requestId: event.request_id,
        currentPage: event.page,
        jumpPageValue: String(event.page),
        explanation: event.explanation,
        speechText: event.speech_text,
        error: null,
        audioStatusText: event.audio_status === "failed" ? "音声:失敗" : "音声:ok",
        audioStatusError: event.audio_error ?? null,
      };
    case "page_load_succeeded":
      if (state.requestId !== null && state.requestId !== event.request_id) {
        return state;
      }
      if (state.activeLoadId !== event.load_id) {
        return state;
      }
      return {
        ...state,
        requestId: event.request_id,
        currentPage: event.page,
        jumpPageValue: String(event.page),
        explanation: event.explanation,
        speechText: event.speech_text,
        imageUrl: event.image_url,
        audioUrl: event.audio_url,
        error: null,
        audioStatusText: event.audio_status === "failed" ? "音声:失敗" : "音声:ok",
        audioStatusError: event.audio_error ?? null,
        isLoadingPage: false,
        isRegenerating: false,
        activeLoadId: null,
        generationStatusText: "",
      };
    case "page_load_failed":
      if (state.requestId !== null && state.requestId !== event.request_id) {
        return state;
      }
      if (state.activeLoadId !== event.load_id) {
        return state;
      }
      return {
        ...state,
        error: event.error,
        audioStatusText: "",
        audioStatusError: null,
        isLoadingPage: false,
        isRegenerating: false,
        activeLoadId: null,
        generationStatusText: "",
      };
    case "generation_started":
      if (state.requestId !== null && state.requestId !== event.request_id) {
        return state;
      }
      return {
        ...state,
        requestId: event.request_id,
        generationStatusText: `${event.page}ページ目生成中`,
      };
    case "generation_finished":
      if (state.requestId !== null && state.requestId !== event.request_id) {
        return state;
      }
      return {
        ...state,
        requestId: event.request_id,
        generationStatusText: "",
      };
    case "favorite_toggled":
      if (event.page_num !== undefined && event.page_num !== state.currentPage) {
        return state;
      }
      if (state.requestId !== null && state.requestId !== event.request_id) {
        return state;
      }
      return {
        ...state,
        requestId: event.request_id,
        isFavorited: event.is_favorited,
      };
    case "ws_event":
      if (event.event.request_id !== state.requestId && state.requestId !== null) {
        return state;
      }
      if (event.event.type === "session_snapshot") {
        return {
          ...state,
          requestId: event.event.request_id,
          currentPage: event.event.current_page ?? state.currentPage,
          maxPage: event.event.page_num ?? state.maxPage,
          isFavorited: event.event.is_favorited,
          totalGenerationCount: event.event.total_generation_count ?? state.totalGenerationCount,
          totalGenerationElapsedMs: event.event.total_generation_elapsed_ms ?? state.totalGenerationElapsedMs,
          totalInputTokens: event.event.total_input_tokens ?? state.totalInputTokens,
          totalOutputTokens: event.event.total_output_tokens ?? state.totalOutputTokens,
          totalCostUsd: event.event.total_cost_usd ?? state.totalCostUsd,
          generationStatusText: "",
        };
      }
      if (event.event.type === "page_updated") {
        return {
          ...state,
          requestId: event.event.request_id,
          currentPage: event.event.current_page,
          jumpPageValue: String(event.event.current_page),
          isFavorited: event.event.is_favorited ?? state.isFavorited,
        };
      }
      if (event.event.type === "favorite_toggled") {
        if (event.event.page_num !== undefined && event.event.page_num !== state.currentPage) {
          return state;
        }
        return {
          ...state,
          requestId: event.event.request_id,
          isFavorited: event.event.is_favorited,
        };
      }
      if (event.event.type === "generation_started") {
        return {
          ...state,
          requestId: event.event.request_id,
          generationStatusText: `${event.event.page_num}ページ目生成中`,
        };
      }
      if (event.event.type === "generation_finished") {
        return {
          ...state,
          requestId: event.event.request_id,
          generationStatusText: "",
        };
      }
      return state;
    default:
      return state;
  }
}

export function simulateDocumentSessionFlow(
  events: DocumentSessionFlowEvent[],
  initialState?: DocumentSessionFlowState,
): DocumentSessionFlowState {
  let state = initialState ?? createDocumentSessionFlowState();
  for (const event of events) {
    state = applyDocumentSessionFlowEvent(state, event);
  }
  return state;
}
