import type { Paper, SearchCandidate, SearchHit, SessionEventMessage } from "./api";

export type SessionViewState = {
  currentSessionId: string | null;
  currentPaper: Paper | null;
  currentPaperSource: string | null;
  simpleSearchQuery: string;
  keywordSearchQuery: string;
  fulltextSearchQuery: string;
  searchModes: string[];
  trailPaperIds: string[];
  nextPaperId: string | null;
  searchPaperId: string | null;
  hits: SearchHit[];
  rejectedCandidates: SearchCandidate[];
  previousSearchPaperId: string | null;
  previousRejectedCandidates: SearchCandidate[];
  fallbackUsed: boolean;
};

export function emptySessionViewState(): SessionViewState {
  return {
    currentSessionId: null,
    currentPaper: null,
    currentPaperSource: null,
    simpleSearchQuery: "",
    keywordSearchQuery: "",
    fulltextSearchQuery: "",
    searchModes: [],
    trailPaperIds: [],
    nextPaperId: null,
    searchPaperId: null,
    hits: [],
    rejectedCandidates: [],
    previousSearchPaperId: null,
    previousRejectedCandidates: [],
    fallbackUsed: false,
  };
}

function snapshotPreviousSearch(state: SessionViewState): {
  previousSearchPaperId: string | null;
  previousRejectedCandidates: SearchCandidate[];
} {
  const previousSearchPaperId = state.searchPaperId ?? state.currentPaper?.id ?? null;
  const previousRejectedCandidates = previousSearchPaperId ? state.rejectedCandidates : [];
  return { previousSearchPaperId, previousRejectedCandidates };
}

function audioUrlsFromEvent(event: SessionEventMessage): string[] {
  return event.audio_urls && event.audio_urls.length > 0 ? event.audio_urls : event.audio_url ? [event.audio_url] : [];
}

export function shouldShowSearchResults(state: Pick<SessionViewState, "currentPaper" | "searchPaperId">): boolean {
  return state.currentPaper !== null && state.searchPaperId === state.currentPaper.id;
}

export function shouldIgnoreStaleSearch(params: {
  currentSessionId: string | null;
  currentPaperId: string | null;
  messageSessionId?: string | null;
  messagePaperId?: string | null;
}): boolean {
  if (!params.messagePaperId) {
    return false;
  }
  if (params.currentSessionId !== null && params.messageSessionId !== null && params.currentSessionId !== params.messageSessionId) {
    return true;
  }
  if (params.currentPaperId !== null && params.currentPaperId !== params.messagePaperId) {
    return true;
  }
  return false;
}

export function applySessionEvent(state: SessionViewState, event: SessionEventMessage): SessionViewState {
  if (event.type === "session_started") {
    const preserveSearchResults =
      state.currentSessionId === null || state.currentSessionId === (event.session_id ?? state.currentSessionId);
    return {
      ...state,
      currentSessionId: event.session_id ?? state.currentSessionId,
      searchPaperId: preserveSearchResults ? state.searchPaperId : null,
      hits: preserveSearchResults ? state.hits : [],
      rejectedCandidates: preserveSearchResults ? state.rejectedCandidates : [],
      fallbackUsed: preserveSearchResults ? state.fallbackUsed : false,
      simpleSearchQuery: preserveSearchResults ? state.simpleSearchQuery : "",
      keywordSearchQuery: preserveSearchResults ? state.keywordSearchQuery : "",
      fulltextSearchQuery: preserveSearchResults ? state.fulltextSearchQuery : "",
      searchModes: preserveSearchResults ? state.searchModes : [],
      trailPaperIds: preserveSearchResults ? state.trailPaperIds : [],
      nextPaperId: preserveSearchResults ? state.nextPaperId : null,
      previousSearchPaperId: preserveSearchResults ? state.previousSearchPaperId : null,
      previousRejectedCandidates: preserveSearchResults ? state.previousRejectedCandidates : [],
    };
  }

  if (event.type === "paper_ready") {
    if (!event.paper) {
      return state;
    }
    const preservePreviousSearch =
      state.currentPaper !== null &&
      state.currentPaper.id !== event.paper.id &&
      (event.from_paper_id === null || state.currentPaper.id === event.from_paper_id);
    const previousSearch = preservePreviousSearch ? snapshotPreviousSearch(state) : { previousSearchPaperId: null, previousRejectedCandidates: [] };
    const nextState: SessionViewState = {
      ...state,
      currentSessionId: event.session_id ?? state.currentSessionId,
      currentPaper: event.paper,
      currentPaperSource: event.origin ?? null,
      simpleSearchQuery: event.simple_search_query ?? event.followup_query ?? "",
      keywordSearchQuery: event.keyword_search_query ?? event.search_keyword ?? event.followup_query ?? "",
      fulltextSearchQuery: event.fulltext_search_query ?? "",
      searchModes: event.search_modes ?? [],
      trailPaperIds: event.trail_paper_ids ?? [],
      nextPaperId: event.next_paper_id ?? null,
      previousSearchPaperId: previousSearch.previousSearchPaperId,
      previousRejectedCandidates: previousSearch.previousRejectedCandidates,
    };
    if (!event.search_deferred) {
      nextState.searchPaperId = event.paper.id;
      nextState.hits = event.search?.hits ?? [];
      nextState.rejectedCandidates = event.search?.rejected_candidates ?? [];
      nextState.fallbackUsed = Boolean(event.search?.fallback_used);
      return nextState;
    }
    if (state.searchPaperId !== event.paper.id) {
      nextState.searchPaperId = null;
      nextState.hits = [];
      nextState.rejectedCandidates = [];
      nextState.fallbackUsed = false;
    }
    return nextState;
  }

  if (event.type === "paper_search_updated") {
    return {
      ...state,
      currentSessionId: event.session_id ?? state.currentSessionId,
      simpleSearchQuery: event.simple_search_query ?? event.followup_query ?? state.simpleSearchQuery,
      keywordSearchQuery: event.keyword_search_query ?? event.search_keyword ?? event.followup_query ?? state.keywordSearchQuery,
      fulltextSearchQuery: event.fulltext_search_query ?? state.fulltextSearchQuery,
      searchModes: event.search_modes ?? state.searchModes,
      nextPaperId: event.next_paper_id ?? state.nextPaperId,
      searchPaperId: event.paper_id ?? state.searchPaperId,
      hits: event.search?.hits ?? state.hits,
      rejectedCandidates: event.search?.rejected_candidates ?? state.rejectedCandidates,
      fallbackUsed: Boolean(event.search?.fallback_used),
    };
  }

  if (event.type === "session_next_candidate_updated") {
    return {
      ...state,
      nextPaperId: event.next_paper_id ?? state.nextPaperId,
    };
  }

  if (event.type === "session_stopped") {
    return emptySessionViewState();
  }

  if (event.type === "session_regenerated") {
    return {
      ...state,
      currentSessionId: event.session_id ?? state.currentSessionId,
    };
  }

  if (event.type === "session_costs_updated") {
    return {
      ...state,
      currentSessionId: event.session_id ?? state.currentSessionId,
    };
  }

  return state;
}
