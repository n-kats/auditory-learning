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
  queuedPaperIds: string[];
  nextPaperId: string | null;
  paperTitleMap: Record<string, string>;
  hits: SearchHit[];
  rejectedCandidates: SearchCandidate[];
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
    queuedPaperIds: [],
    nextPaperId: snapshot.next_paper_id ?? null,
    paperTitleMap: {},
    hits: [],
    rejectedCandidates: [],
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
      state.currentSessionId = event.session_id ?? state.currentSessionId;
      continue;
    }
    if (event.type === "paper_ready") {
      if (!event.paper) {
        continue;
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
      state.queuedPaperIds = event.queued_paper_ids ?? [];
      state.nextPaperId = event.next_paper_id ?? null;
      state.hits = event.search?.hits ?? [];
      state.rejectedCandidates = event.search?.rejected_candidates ?? [];
      state.fallbackUsed = Boolean(event.search?.fallback_used);
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
    if (event.type === "session_queued") {
      state.queuedPaperIds = event.queued_paper_ids ?? state.queuedPaperIds;
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
      state.queuedPaperIds = [];
      state.hits = [];
      state.rejectedCandidates = [];
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
