import { describe, expect, it } from "vitest";

import {
  applyNextCandidateUpdatedToAppSessionState,
  applyPaperReadyToAppSessionState,
  applyReplayToAppSessionState,
  applySearchUpdatedToAppSessionState,
  applySessionStartedToAppSessionState,
  applySessionStoppedToAppSessionState,
  emptyAppSessionState,
} from "./appSessionState";

describe("appSessionState helpers", () => {
  it("resets only session-scoped state on start and stop", () => {
    const state = {
      ...emptyAppSessionState(),
      favorites: [{ paper_id: "fav-1", title: "Favorite" }],
      backendNotices: ["notice"],
      explanation: "exp",
      paperCosts: { session_id: "s-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      sessionCosts: { session_id: "s-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      audioUrls: ["/audio/1"],
      paperTitleMap: { "paper-1": "Paper 1" },
      currentSessionId: "session-old",
      currentPaper: { id: "paper-1", title: "Paper 1", abstract: "", categories: [] },
      searchPaperId: "paper-1",
      hits: [{ paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
    };

    const started = applySessionStartedToAppSessionState(state, "session-new");
    expect(started.currentSessionId).toBe("session-new");
    expect(started.currentPaper).toBeNull();
    expect(started.searchPaperId).toBeNull();
    expect(started.hits).toEqual([]);
    expect(started.backendNotices).toEqual([]);
    expect(started.favorites).toEqual(state.favorites);

    const stopped = applySessionStoppedToAppSessionState(started);
    expect(stopped.currentSessionId).toBeNull();
    expect(stopped.currentPaper).toBeNull();
    expect(stopped.searchPaperId).toBeNull();
    expect(stopped.hits).toEqual([]);
    expect(stopped.favorites).toEqual(state.favorites);
  });

  it("applies paper ready and search updated events to the app state", () => {
    let state = emptyAppSessionState();
    state = applySessionStartedToAppSessionState(state, "session-1");

    state = applyPaperReadyToAppSessionState(state, {
      session_id: "session-1",
      search_deferred: true,
      paper: { id: "paper-1", title: "Paper 1", abstract: "", categories: [] },
      origin: "search",
      trail_paper_ids: ["root"],
      next_paper_id: null,
      simple_search_query: "q1",
      keyword_search_query: "q1",
      fulltext_search_query: "q1",
      search_modes: ["simple"],
      explanation: "exp",
      memo: "memo",
      audio_urls: ["/audio/1"],
      audio_duration_ms: 1200,
      notices: ["api notice"],
      paper_costs: { session_id: "session-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      session_costs: { session_id: "session-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });
    expect(state.currentPaper?.id).toBe("paper-1");
    expect(state.paperTitleMap["paper-1"]).toBe("Paper 1");
    expect(state.audioUrls).toEqual(["/audio/1"]);
    expect(state.backendNotices).toEqual(["api notice"]);
    expect(state.searchPaperId).toBeNull();

    state = applySearchUpdatedToAppSessionState(state, {
      session_id: "session-1",
      paper_id: "paper-1",
      next_paper_id: "paper-next",
      simple_search_query: "q1",
      search_modes: ["simple"],
      search: {
        hits: [{ paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
        rejected_candidates: [],
        fallback_used: false,
      },
    });
    expect(state.searchPaperId).toBe("paper-1");
    expect(state.hits[0].paper.id).toBe("paper-2");
    expect(state.nextPaperId).toBe("paper-next");

    state = applyNextCandidateUpdatedToAppSessionState(state, {
      next_paper_id: "paper-alt",
    });
    expect(state.nextPaperId).toBe("paper-alt");
  });

  it("carries the previous paper's rejected candidates into the next paper view", () => {
    let state = emptyAppSessionState();
    state = applySessionStartedToAppSessionState(state, "session-1");

    state = applyPaperReadyToAppSessionState(state, {
      session_id: "session-1",
      search_deferred: true,
      paper: { id: "paper-1", title: "Paper 1", abstract: "", categories: [] },
      origin: "search",
      trail_paper_ids: [],
      next_paper_id: null,
      simple_search_query: "q1",
      keyword_search_query: "q1",
      fulltext_search_query: "q1",
      search_modes: ["simple"],
      explanation: "exp-1",
      memo: "memo-1",
      audio_urls: ["/audio/1"],
      audio_duration_ms: 1200,
      notices: [],
      paper_costs: { session_id: "session-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      session_costs: { session_id: "session-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });
    state = applySearchUpdatedToAppSessionState(state, {
      session_id: "session-1",
      paper_id: "paper-1",
      next_paper_id: "paper-2",
      simple_search_query: "q1",
      search_modes: ["simple"],
      search: {
        hits: [{ paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
        rejected_candidates: [{ paper: { id: "paper-3", title: "Paper 3", abstract: "", categories: [] }, paper_id: "paper-3", title: "Paper 3", score: 0.4, reason: "3rd" }],
        fallback_used: false,
      },
    });

    state = applyPaperReadyToAppSessionState(state, {
      session_id: "session-1",
      search_deferred: true,
      paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] },
      origin: "search",
      trail_paper_ids: ["paper-1"],
      next_paper_id: "paper-4",
      simple_search_query: "q2",
      keyword_search_query: "q2",
      fulltext_search_query: "q2",
      search_modes: ["simple"],
      explanation: "exp-2",
      memo: "memo-2",
      audio_urls: ["/audio/2"],
      audio_duration_ms: 1200,
      notices: [],
      paper_costs: { session_id: "session-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      session_costs: { session_id: "session-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });

    expect(state.currentPaper?.id).toBe("paper-2");
    expect(state.previousSearchPaperId).toBe("paper-1");
    expect(state.previousRejectedCandidates).toHaveLength(1);
    expect(state.previousRejectedCandidates[0].paper_id).toBe("paper-3");
    expect(state.searchPaperId).toBeNull();
    expect(state.rejectedCandidates).toEqual([]);
  });

  it("keeps the current paper when session_started arrives for the same session after paper_ready", () => {
    let state = emptyAppSessionState();
    state = applyPaperReadyToAppSessionState(state, {
      session_id: "session-1",
      search_deferred: true,
      paper: { id: "paper-1", title: "Paper 1", abstract: "", categories: [] },
      origin: "search",
      trail_paper_ids: [],
      next_paper_id: null,
      simple_search_query: "q1",
      keyword_search_query: "q1",
      fulltext_search_query: "q1",
      search_modes: ["simple"],
      explanation: "exp",
      memo: "memo",
      audio_urls: ["/audio/1"],
      audio_duration_ms: 1200,
      notices: [],
      paper_costs: { session_id: "session-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      session_costs: { session_id: "session-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });

    const next = applySessionStartedToAppSessionState(state, "session-1");
    expect(next.currentSessionId).toBe("session-1");
    expect(next.currentPaper?.id).toBe("paper-1");
    expect(next.explanation).toBe("exp");
    expect(next.audioUrls).toEqual(["/audio/1"]);
  });

  it("keeps search state when session_started arrives after a same-session search update", () => {
    let state = emptyAppSessionState();

    state = applySearchUpdatedToAppSessionState(state, {
      session_id: "session-1",
      paper_id: "paper-1",
      next_paper_id: "paper-next",
      simple_search_query: "q1",
      search_modes: ["simple"],
      search: {
        hits: [{ paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
        rejected_candidates: [],
        fallback_used: false,
      },
    });

    state = applySessionStartedToAppSessionState(state, "session-1");
    expect(state.searchPaperId).toBe("paper-1");
    expect(state.hits[0].paper.id).toBe("paper-2");

    state = applyPaperReadyToAppSessionState(state, {
      session_id: "session-1",
      search_deferred: true,
      paper: { id: "paper-1", title: "Paper 1", abstract: "", categories: [] },
      origin: "search",
      trail_paper_ids: [],
      next_paper_id: "paper-next",
      simple_search_query: "q1",
      keyword_search_query: "q1",
      fulltext_search_query: "q1",
      search_modes: ["simple"],
      explanation: "exp",
      memo: "memo",
      audio_urls: ["/audio/1"],
      audio_duration_ms: 1200,
      notices: [],
      paper_costs: { session_id: "session-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      session_costs: { session_id: "session-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });
    expect(state.currentPaper?.id).toBe("paper-1");
    expect(state.searchPaperId).toBe("paper-1");
    expect(state.hits[0].paper.id).toBe("paper-2");
  });

  it("applies replay state without losing unrelated app state", () => {
    const state = {
      ...emptyAppSessionState(),
      favorites: [{ paper_id: "fav-1", title: "Favorite" }],
      backendNotices: ["old"],
    };
    const replayState = {
      currentSessionId: "session-1",
      currentPaper: { id: "paper-1", title: "Paper 1", abstract: "", categories: [] },
      currentPaperSource: "search",
      simpleSearchQuery: "q1",
      keywordSearchQuery: "q1",
      fulltextSearchQuery: "q1",
      searchModes: ["simple"],
      trailPaperIds: ["root"],
      nextPaperId: "paper-next",
      paperTitleMap: { "paper-1": "Paper 1" },
      searchPaperId: "paper-1",
      hits: [{ paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
      rejectedCandidates: [],
      previousSearchPaperId: "paper-1",
      previousRejectedCandidates: [],
      fallbackUsed: false,
      explanation: "exp",
      memo: "memo",
      paperCosts: { session_id: "session-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      sessionCosts: { session_id: "session-1", total_elapsed_ms: 1, total_wall_elapsed_ms: 1, total_cost_usd: 1, items: [] },
      audioUrls: ["/audio/1"],
      audioDurationMs: 1200,
      notices: ["notice"],
      isPlaying: false,
      activeTab: "session" as const,
      lastEventSeq: 4,
    };

    const nextState = applyReplayToAppSessionState(state, replayState);
    expect(nextState.currentSessionId).toBe("session-1");
    expect(nextState.currentPaper?.id).toBe("paper-1");
    expect(nextState.backendNotices).toEqual(["notice"]);
    expect(nextState.favorites).toEqual(state.favorites);
  });
});
