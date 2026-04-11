import type { FavoritePaperItem, Paper, SearchCandidate, SearchHit, SessionCosts } from "./api";
import { emptySessionViewState, type SessionViewState } from "./sessionViewState";
import type { SessionReplayState } from "./sessionReplay";

export type AppSessionState = SessionViewState & {
  explanation: string;
  paperCosts: SessionCosts | null;
  sessionCosts: SessionCosts | null;
  paperTitleMap: Record<string, string>;
  backendNotices: string[];
  audioUrls: string[];
  audioIndex: number;
  audioDurationMs: number | null;
  favorites: FavoritePaperItem[];
};

export function emptyAppSessionState(): AppSessionState {
  const base = emptySessionViewState();
  return {
    ...base,
    explanation: "",
    paperCosts: null,
    sessionCosts: null,
    paperTitleMap: {},
    backendNotices: [],
    audioUrls: [],
    audioIndex: 0,
    audioDurationMs: null,
    favorites: [],
  };
}

function snapshotPreviousSearch(state: AppSessionState): {
  previousSearchPaperId: string | null;
  previousRejectedCandidates: SearchCandidate[];
} {
  const previousSearchPaperId = state.searchPaperId ?? state.currentPaper?.id ?? null;
  return {
    previousSearchPaperId,
    previousRejectedCandidates: previousSearchPaperId ? state.rejectedCandidates : [],
  };
}

export function resetSessionScopedAppSessionState(
  state: AppSessionState,
  currentSessionId: string | null,
  options?: { preserveSearchResults?: boolean },
): AppSessionState {
  const preserveSearchResults = options?.preserveSearchResults ?? false;
  return {
    ...state,
    currentSessionId,
    currentPaper: null,
    currentPaperSource: null,
    simpleSearchQuery: preserveSearchResults ? state.simpleSearchQuery : "",
    keywordSearchQuery: preserveSearchResults ? state.keywordSearchQuery : "",
    fulltextSearchQuery: preserveSearchResults ? state.fulltextSearchQuery : "",
    searchModes: preserveSearchResults ? state.searchModes : [],
    trailPaperIds: preserveSearchResults ? state.trailPaperIds : [],
    nextPaperId: preserveSearchResults ? state.nextPaperId : null,
    searchPaperId: preserveSearchResults ? state.searchPaperId : null,
    hits: preserveSearchResults ? state.hits : [],
    rejectedCandidates: preserveSearchResults ? state.rejectedCandidates : [],
    previousSearchPaperId: preserveSearchResults ? state.previousSearchPaperId : null,
    previousRejectedCandidates: preserveSearchResults ? state.previousRejectedCandidates : [],
    fallbackUsed: preserveSearchResults ? state.fallbackUsed : false,
    explanation: "",
    paperCosts: null,
    sessionCosts: null,
    paperTitleMap: {},
    backendNotices: [],
    audioUrls: [],
    audioIndex: 0,
    audioDurationMs: null,
  };
}

export function applyPaperReadyToAppSessionState(
  state: AppSessionState,
  payload: {
    session_id?: string | null;
    paper: Paper;
    origin?: string | null;
    simple_search_query?: string;
    followup_query?: string;
    keyword_search_query?: string;
    search_keyword?: string;
    fulltext_search_query?: string;
    search_modes?: string[];
    trail_paper_ids?: string[];
    next_paper_id?: string | null;
    search_deferred?: boolean;
    explanation?: string;
    paper_costs?: SessionCosts | null;
    session_costs?: SessionCosts | null;
    audio_url?: string;
    audio_urls?: string[];
    audio_duration_ms?: number | null;
    notices?: string[];
    search?: {
      hits?: SearchHit[];
      rejected_candidates?: SearchCandidate[];
      fallback_used?: boolean;
    };
    memo?: string;
  },
): AppSessionState {
  const nextState: AppSessionState = {
    ...state,
    currentSessionId: payload.session_id ?? state.currentSessionId,
    currentPaper: payload.paper,
    currentPaperSource: payload.origin ?? null,
    simpleSearchQuery: payload.simple_search_query ?? payload.followup_query ?? "",
    keywordSearchQuery: payload.keyword_search_query ?? payload.search_keyword ?? payload.followup_query ?? "",
    fulltextSearchQuery: payload.fulltext_search_query ?? "",
    searchModes: payload.search_modes ?? [],
    explanation: payload.explanation ?? "",
    paperCosts: payload.paper_costs ?? null,
    sessionCosts: payload.session_costs ?? null,
    backendNotices: payload.notices ?? [],
    audioUrls: payload.audio_urls && payload.audio_urls.length > 0 ? payload.audio_urls : payload.audio_url ? [payload.audio_url] : [],
    audioIndex: 0,
    audioDurationMs: null,
    trailPaperIds: payload.trail_paper_ids ?? [],
    nextPaperId:
      payload.next_paper_id ?? (state.searchPaperId === payload.paper.id ? state.nextPaperId : null),
    paperTitleMap: {
      ...state.paperTitleMap,
      [payload.paper.id]: payload.paper.title,
    },
  };
  const shouldPreservePreviousSearch =
    state.currentPaper !== null &&
    state.currentPaper.id !== payload.paper.id &&
    (payload.session_id === null || state.currentSessionId === null || state.currentSessionId === payload.session_id);
  if (shouldPreservePreviousSearch) {
    const snapshot = snapshotPreviousSearch(state);
    nextState.previousSearchPaperId = snapshot.previousSearchPaperId;
    nextState.previousRejectedCandidates = snapshot.previousRejectedCandidates;
  } else {
    nextState.previousSearchPaperId = state.previousSearchPaperId;
    nextState.previousRejectedCandidates = state.previousRejectedCandidates;
  }
  if (!payload.search_deferred) {
    nextState.searchPaperId = payload.paper.id;
    nextState.hits = payload.search?.hits ?? [];
    nextState.rejectedCandidates = payload.search?.rejected_candidates ?? [];
    nextState.fallbackUsed = Boolean(payload.search?.fallback_used);
    return nextState;
  }
  if (state.searchPaperId !== payload.paper.id) {
    nextState.searchPaperId = null;
    nextState.hits = [];
    nextState.rejectedCandidates = [];
    nextState.fallbackUsed = false;
  }
  return nextState;
}

export function applySearchUpdatedToAppSessionState(
  state: AppSessionState,
  payload: {
    session_id?: string | null;
    paper_id?: string | null;
    simple_search_query?: string;
    followup_query?: string;
    keyword_search_query?: string;
    search_keyword?: string;
    fulltext_search_query?: string;
    search_modes?: string[];
    next_paper_id?: string | null;
    notices?: string[];
    search?: {
      hits?: SearchHit[];
      rejected_candidates?: SearchCandidate[];
      fallback_used?: boolean;
    };
  },
): AppSessionState {
  return {
    ...state,
    currentSessionId: payload.session_id ?? state.currentSessionId,
    simpleSearchQuery: payload.simple_search_query ?? payload.followup_query ?? state.simpleSearchQuery,
    keywordSearchQuery: payload.keyword_search_query ?? payload.search_keyword ?? payload.followup_query ?? state.keywordSearchQuery,
    fulltextSearchQuery: payload.fulltext_search_query ?? state.fulltextSearchQuery,
    searchModes: payload.search_modes ?? state.searchModes,
    nextPaperId: payload.next_paper_id ?? state.nextPaperId,
    searchPaperId: payload.paper_id ?? state.searchPaperId,
    hits: payload.search?.hits ?? state.hits,
    rejectedCandidates: payload.search?.rejected_candidates ?? state.rejectedCandidates,
    previousSearchPaperId: state.previousSearchPaperId,
    previousRejectedCandidates: state.previousRejectedCandidates,
    fallbackUsed: Boolean(payload.search?.fallback_used),
    backendNotices: payload.notices && payload.notices.length > 0 ? [...state.backendNotices, ...payload.notices] : state.backendNotices,
  };
}

export function applyNextCandidateUpdatedToAppSessionState(
  state: AppSessionState,
  payload: { next_paper_id?: string | null },
): AppSessionState {
  return {
    ...state,
    nextPaperId: payload.next_paper_id ?? state.nextPaperId,
  };
}

export function applySessionStoppedToAppSessionState(state: AppSessionState): AppSessionState {
  return resetSessionScopedAppSessionState(state, null);
}

export function applySessionStartedToAppSessionState(state: AppSessionState, sessionId: string | null): AppSessionState {
  if (state.currentSessionId !== null && state.currentSessionId === sessionId) {
    return {
      ...state,
      currentSessionId: sessionId,
    };
  }
  return resetSessionScopedAppSessionState(state, sessionId, {
    preserveSearchResults: state.currentSessionId === sessionId && sessionId !== null,
  });
}

export function applyReplayToAppSessionState(state: AppSessionState, replayState: SessionReplayState): AppSessionState {
  return {
    ...state,
    currentSessionId: replayState.currentSessionId,
    currentPaper: replayState.currentPaper,
    currentPaperSource: replayState.currentPaperSource,
    simpleSearchQuery: replayState.simpleSearchQuery,
    keywordSearchQuery: replayState.keywordSearchQuery,
    fulltextSearchQuery: replayState.fulltextSearchQuery,
    searchModes: replayState.searchModes,
    trailPaperIds: replayState.trailPaperIds,
    nextPaperId: replayState.nextPaperId,
    paperTitleMap: replayState.paperTitleMap,
    searchPaperId: replayState.searchPaperId,
    hits: replayState.hits,
    rejectedCandidates: replayState.rejectedCandidates,
    previousSearchPaperId: replayState.previousSearchPaperId,
    previousRejectedCandidates: replayState.previousRejectedCandidates,
    fallbackUsed: replayState.fallbackUsed,
    explanation: replayState.explanation,
    paperCosts: replayState.paperCosts,
    sessionCosts: replayState.sessionCosts,
    backendNotices: replayState.notices,
    audioUrls: replayState.audioUrls,
    audioIndex: 0,
    audioDurationMs: replayState.audioDurationMs,
  };
}
