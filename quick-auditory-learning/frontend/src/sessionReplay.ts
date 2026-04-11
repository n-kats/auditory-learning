import type { Paper, SearchCandidate, SearchHit, SessionCosts, SessionEventMessage, SessionSnapshot } from "./api";

export type SessionReplayState = {
  currentSessionId: string | null;
  currentPaper: Paper | null;
  currentPaperSource: string | null;
  simpleSearchQuery: string;
  keywordSearchQuery: string;
  fulltextSearchQuery: string;
  searchModes: string[];
  trailPaperIds: string[];
  nextPaperId: string | null;
  paperTitleMap: Record<string, string>;
  searchPaperId: string | null;
  hits: SearchHit[];
  rejectedCandidates: SearchCandidate[];
  previousSearchPaperId: string | null;
  previousRejectedCandidates: SearchCandidate[];
  fallbackUsed: boolean;
  explanation: string;
  memo: string;
  paperCosts: SessionCosts | null;
  sessionCosts: SessionCosts | null;
  audioUrls: string[];
  audioDurationMs: number | null;
  notices: string[];
  isPlaying: boolean;
  activeTab: "start" | "favorites" | "session";
  lastEventSeq: number;
};

function emptySessionReplayState(snapshot: SessionSnapshot): SessionReplayState {
  return {
    currentSessionId: snapshot.session_id,
    currentPaper: null,
    currentPaperSource: null,
    simpleSearchQuery: "",
    keywordSearchQuery: "",
    fulltextSearchQuery: "",
    searchModes: [],
    trailPaperIds: [],
    nextPaperId: snapshot.next_paper_id ?? null,
    paperTitleMap: {},
    searchPaperId: null,
    hits: [],
    rejectedCandidates: [],
    previousSearchPaperId: null,
    previousRejectedCandidates: [],
    fallbackUsed: false,
    explanation: "",
    memo: "",
    paperCosts: null,
    sessionCosts: null,
    audioUrls: [],
    audioDurationMs: null,
    notices: [],
    isPlaying: false,
    activeTab: "session",
    lastEventSeq: 0,
  };
}

function audioUrlsFromEvent(event: SessionEventMessage): string[] {
  return event.audio_urls && event.audio_urls.length > 0 ? event.audio_urls : event.audio_url ? [event.audio_url] : [];
}

export function replaySessionEvents(snapshot: SessionSnapshot, events: SessionEventMessage[]): SessionReplayState {
  const state = emptySessionReplayState(snapshot);

  for (const event of events) {
    if (typeof event.seq === "number" && event.seq > state.lastEventSeq) {
      state.lastEventSeq = event.seq;
    }
    if (event.type === "session_started") {
      const nextSessionId = event.session_id ?? state.currentSessionId;
      const preserveSearchResults = state.currentSessionId === null || state.currentSessionId === nextSessionId;
      state.currentSessionId = nextSessionId;
      if (!preserveSearchResults) {
        state.searchPaperId = null;
        state.hits = [];
        state.rejectedCandidates = [];
        state.fallbackUsed = false;
        state.simpleSearchQuery = "";
        state.keywordSearchQuery = "";
        state.fulltextSearchQuery = "";
        state.searchModes = [];
        state.trailPaperIds = [];
        state.nextPaperId = null;
        state.previousSearchPaperId = null;
        state.previousRejectedCandidates = [];
      }
      continue;
    }
    if (event.type === "paper_ready") {
      if (!event.paper) {
        continue;
      }
      const shouldPreservePreviousSearch =
        state.currentPaper !== null &&
        state.currentPaper.id !== event.paper.id &&
        (event.from_paper_id === null || state.currentPaper.id === event.from_paper_id);
      if (shouldPreservePreviousSearch) {
        state.previousSearchPaperId = state.searchPaperId ?? state.currentPaper?.id ?? null;
        state.previousRejectedCandidates = state.previousSearchPaperId ? state.rejectedCandidates : [];
      } else if (state.currentSessionId !== snapshot.session_id) {
        state.previousSearchPaperId = null;
        state.previousRejectedCandidates = [];
      }
      state.currentSessionId = event.session_id ?? state.currentSessionId;
      state.currentPaper = event.paper;
      state.paperTitleMap = { ...state.paperTitleMap, [event.paper.id]: event.paper.title };
      state.currentPaperSource = event.origin ?? null;
      state.simpleSearchQuery = event.simple_search_query ?? event.followup_query ?? "";
      state.keywordSearchQuery = event.keyword_search_query ?? event.search_keyword ?? event.followup_query ?? "";
      state.fulltextSearchQuery = event.fulltext_search_query ?? "";
      state.searchModes = event.search_modes ?? [];
      state.trailPaperIds = event.trail_paper_ids ?? [];
      state.nextPaperId = event.next_paper_id ?? (state.searchPaperId === event.paper.id ? state.nextPaperId : null);
      if (!event.search_deferred) {
        state.searchPaperId = event.paper.id;
        state.hits = event.search?.hits ?? [];
        state.rejectedCandidates = event.search?.rejected_candidates ?? [];
        state.fallbackUsed = Boolean(event.search?.fallback_used);
      } else if (state.searchPaperId !== event.paper.id) {
        state.searchPaperId = null;
        state.hits = [];
        state.rejectedCandidates = [];
        state.fallbackUsed = false;
      }
      state.explanation = event.explanation ?? "";
      state.memo = event.memo ?? "";
      state.paperCosts = event.paper_costs ?? null;
      state.sessionCosts = event.session_costs ?? null;
      state.audioUrls = audioUrlsFromEvent(event);
      state.audioDurationMs = event.audio_duration_ms ?? null;
      state.notices = event.notices ?? [];
      state.isPlaying = false;
      state.activeTab = "session";
      continue;
    }
    if (event.type === "paper_search_updated") {
      if (event.session_id) {
        state.currentSessionId = event.session_id;
      }
      state.simpleSearchQuery = event.simple_search_query ?? event.followup_query ?? state.simpleSearchQuery;
      state.keywordSearchQuery = event.keyword_search_query ?? event.search_keyword ?? event.followup_query ?? state.keywordSearchQuery;
      state.fulltextSearchQuery = event.fulltext_search_query ?? state.fulltextSearchQuery;
      state.searchModes = event.search_modes ?? state.searchModes;
      state.nextPaperId = event.next_paper_id ?? state.nextPaperId;
      state.searchPaperId = event.paper_id ?? state.searchPaperId;
      state.hits = event.search?.hits ?? state.hits;
      state.rejectedCandidates = event.search?.rejected_candidates ?? state.rejectedCandidates;
      state.fallbackUsed = Boolean(event.search?.fallback_used);
      if (event.notices && event.notices.length > 0) {
        state.notices = [...state.notices, ...event.notices];
      }
      continue;
    }
    if (event.type === "session_next_candidate_updated") {
      state.nextPaperId = event.next_paper_id ?? state.nextPaperId;
      continue;
    }
    if (event.type === "session_costs_updated") {
      state.sessionCosts = event.session_costs ?? state.sessionCosts;
      if (state.currentPaper && event.paper_id === state.currentPaper.id && event.paper_costs) {
        state.paperCosts = event.paper_costs;
      }
      continue;
    }
    if (event.type === "session_stopped") {
      state.currentSessionId = null;
      state.currentPaper = null;
      state.currentPaperSource = null;
      state.simpleSearchQuery = "";
      state.keywordSearchQuery = "";
      state.fulltextSearchQuery = "";
      state.searchModes = [];
      state.trailPaperIds = [];
      state.searchPaperId = null;
      state.hits = [];
      state.rejectedCandidates = [];
      state.previousSearchPaperId = null;
      state.previousRejectedCandidates = [];
      state.fallbackUsed = false;
      state.explanation = "";
      state.memo = "";
      state.sessionCosts = null;
      state.audioUrls = [];
      state.audioDurationMs = null;
      state.notices = [];
      state.isPlaying = false;
      state.activeTab = "start";
    }
  }

  return state;
}
