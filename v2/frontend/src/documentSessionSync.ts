import type { SessionSyncEvent } from "./api";

export type DocumentSessionSyncState = {
  requestId: string | null;
  currentPage: number;
  maxPage: number;
  isFavorited: boolean;
  promptExplainText: string;
  promptSpeakText: string;
  modelName: string;
  reasoningEffort: string;
  totalGenerationCount: number;
  totalGenerationElapsedMs: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCostUsd: number;
};

export function createDocumentSessionSyncState(params?: Partial<DocumentSessionSyncState>): DocumentSessionSyncState {
  return {
    requestId: params?.requestId ?? null,
    currentPage: params?.currentPage ?? 1,
    maxPage: params?.maxPage ?? 1,
    isFavorited: params?.isFavorited ?? false,
    promptExplainText: params?.promptExplainText ?? "",
    promptSpeakText: params?.promptSpeakText ?? "",
    modelName: params?.modelName ?? "",
    reasoningEffort: params?.reasoningEffort ?? "",
    totalGenerationCount: params?.totalGenerationCount ?? 0,
    totalGenerationElapsedMs: params?.totalGenerationElapsedMs ?? 0,
    totalInputTokens: params?.totalInputTokens ?? 0,
    totalOutputTokens: params?.totalOutputTokens ?? 0,
    totalCostUsd: params?.totalCostUsd ?? 0,
  };
}

export function applyDocumentSessionSyncEvent(
  state: DocumentSessionSyncState,
  event: SessionSyncEvent,
): DocumentSessionSyncState {
  if (state.requestId !== null && state.requestId !== event.request_id) {
    return state;
  }

  if (event.type === "session_snapshot") {
    const nextState = {
      requestId: event.request_id,
      currentPage: event.current_page ?? state.currentPage,
      maxPage: event.page_num ?? state.maxPage,
      isFavorited: event.is_favorited,
      promptExplainText: event.prompt_explain_text ?? state.promptExplainText,
      promptSpeakText: event.prompt_speak_text ?? state.promptSpeakText,
      modelName: event.model_name ?? state.modelName,
      reasoningEffort: event.reasoning_effort ?? state.reasoningEffort,
      totalGenerationCount: event.total_generation_count ?? state.totalGenerationCount,
      totalGenerationElapsedMs: event.total_generation_elapsed_ms ?? state.totalGenerationElapsedMs,
      totalInputTokens: event.total_input_tokens ?? state.totalInputTokens,
      totalOutputTokens: event.total_output_tokens ?? state.totalOutputTokens,
      totalCostUsd: event.total_cost_usd ?? state.totalCostUsd,
    };
    return nextState.requestId === state.requestId &&
      nextState.currentPage === state.currentPage &&
      nextState.maxPage === state.maxPage &&
      nextState.isFavorited === state.isFavorited &&
      nextState.promptExplainText === state.promptExplainText &&
      nextState.promptSpeakText === state.promptSpeakText &&
      nextState.modelName === state.modelName &&
      nextState.reasoningEffort === state.reasoningEffort &&
      nextState.totalGenerationCount === state.totalGenerationCount &&
      nextState.totalGenerationElapsedMs === state.totalGenerationElapsedMs &&
      nextState.totalInputTokens === state.totalInputTokens &&
      nextState.totalOutputTokens === state.totalOutputTokens &&
      nextState.totalCostUsd === state.totalCostUsd
      ? state
      : nextState;
  }

  if (event.type === "page_updated") {
    const nextState = {
      ...state,
      requestId: event.request_id,
      currentPage: event.current_page,
      isFavorited: event.is_favorited ?? state.isFavorited,
    };
    return nextState.requestId === state.requestId && nextState.currentPage === state.currentPage && nextState.isFavorited === state.isFavorited ? state : nextState;
  }

  if (event.type === "favorite_toggled") {
    if (event.page_num !== undefined && state.currentPage !== event.page_num) {
      return state;
    }
    const nextState = {
      ...state,
      requestId: event.request_id,
      isFavorited: event.is_favorited,
    };
    return nextState.requestId === state.requestId && nextState.isFavorited === state.isFavorited ? state : nextState;
  }

  return state;
}
