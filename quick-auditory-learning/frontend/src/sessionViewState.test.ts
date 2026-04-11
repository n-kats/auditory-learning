import { describe, expect, it } from "vitest";

import { applySessionEvent, emptySessionViewState, shouldIgnoreStaleSearch, shouldShowSearchResults } from "./sessionViewState";

describe("sessionViewState helpers", () => {
  it("ignores stale search updates from another session or paper", () => {
    expect(
      shouldIgnoreStaleSearch({
        currentSessionId: "session-1",
        currentPaperId: "paper-1",
        messageSessionId: "session-2",
        messagePaperId: "paper-1",
      }),
    ).toBe(true);
    expect(
      shouldIgnoreStaleSearch({
        currentSessionId: "session-1",
        currentPaperId: "paper-1",
        messageSessionId: "session-1",
        messagePaperId: "paper-2",
      }),
    ).toBe(true);
    expect(
      shouldIgnoreStaleSearch({
        currentSessionId: "session-1",
        currentPaperId: "paper-1",
        messageSessionId: "session-1",
        messagePaperId: "paper-1",
      }),
    ).toBe(false);
  });

  it("drops search results when a new session starts and keeps them scoped to the current paper", () => {
    let state = emptySessionViewState();

    state = applySessionEvent(state, { type: "session_started", session_id: "session-1" });
    state = applySessionEvent(state, {
      type: "paper_ready",
      session_id: "session-1",
      search_deferred: true,
      paper: { id: "paper-1", title: "Paper 1", abstract: "", categories: [] },
      simple_search_query: "q1",
      keyword_search_query: "q1",
      fulltext_search_query: "q1",
      search_modes: ["simple"],
      trail_paper_ids: [],
      next_paper_id: null,
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });
    state = applySessionEvent(state, {
      type: "paper_search_updated",
      session_id: "session-1",
      paper_id: "paper-1",
      search: {
        hits: [{ paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
        rejected_candidates: [],
        fallback_used: false,
      },
    });

    expect(shouldShowSearchResults(state)).toBe(true);
    expect(state.hits[0].paper.id).toBe("paper-2");

    state = applySessionEvent(state, { type: "session_stopped", session_id: "session-1" });
    expect(shouldShowSearchResults(state)).toBe(false);
    expect(state.hits).toEqual([]);

    state = applySessionEvent(state, { type: "session_started", session_id: "session-2" });
    state = applySessionEvent(state, {
      type: "paper_ready",
      session_id: "session-2",
      search_deferred: true,
      paper: { id: "paper-3", title: "Paper 3", abstract: "", categories: [] },
      simple_search_query: "q2",
      keyword_search_query: "q2",
      fulltext_search_query: "q2",
      search_modes: ["simple"],
      trail_paper_ids: [],
      next_paper_id: null,
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });

    expect(shouldShowSearchResults(state)).toBe(false);
    expect(state.searchPaperId).toBeNull();
    expect(state.hits).toEqual([]);
  });

  it("keeps a same-session search update that arrived before session_started", () => {
    let state = emptySessionViewState();

    state = applySessionEvent(state, {
      type: "paper_search_updated",
      session_id: "session-1",
      paper_id: "paper-1",
      search: {
        hits: [{ paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
        rejected_candidates: [],
        fallback_used: false,
      },
    });
    expect(state.searchPaperId).toBe("paper-1");
    expect(state.hits[0].paper.id).toBe("paper-2");

    state = applySessionEvent(state, { type: "session_started", session_id: "session-1" });
    expect(state.searchPaperId).toBe("paper-1");
    expect(state.hits[0].paper.id).toBe("paper-2");

    state = applySessionEvent(state, {
      type: "paper_ready",
      session_id: "session-1",
      search_deferred: true,
      paper: { id: "paper-1", title: "Paper 1", abstract: "", categories: [] },
      simple_search_query: "q1",
      keyword_search_query: "q1",
      fulltext_search_query: "q1",
      search_modes: ["simple"],
      trail_paper_ids: [],
      next_paper_id: null,
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });

    expect(shouldShowSearchResults(state)).toBe(true);
    expect(state.currentPaper?.id).toBe("paper-1");
    expect(state.searchPaperId).toBe("paper-1");
    expect(state.hits[0].paper.id).toBe("paper-2");
  });

  it("keeps search results regardless of whether paper_ready or paper_search_updated arrives first", () => {
    const paperReady = {
      type: "paper_ready" as const,
      session_id: "session-1",
      search_deferred: true,
      paper: { id: "paper-1", title: "Paper 1", abstract: "", categories: [] },
      simple_search_query: "q1",
      keyword_search_query: "q1",
      fulltext_search_query: "q1",
      search_modes: ["simple"],
      trail_paper_ids: [],
      next_paper_id: null,
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    } satisfies Parameters<typeof applySessionEvent>[1];
    const paperSearchUpdated = {
      type: "paper_search_updated" as const,
      session_id: "session-1",
      paper_id: "paper-1",
      search: {
        hits: [{ paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
        rejected_candidates: [],
        fallback_used: false,
      },
    } satisfies Parameters<typeof applySessionEvent>[1];

    let state = emptySessionViewState();
    state = applySessionEvent(state, { type: "session_started", session_id: "session-1" });
    state = applySessionEvent(state, paperReady);
    state = applySessionEvent(state, paperSearchUpdated);

    expect(shouldShowSearchResults(state)).toBe(true);
    expect(state.searchPaperId).toBe("paper-1");
    expect(state.hits[0].paper.id).toBe("paper-2");

    state = emptySessionViewState();
    state = applySessionEvent(state, { type: "session_started", session_id: "session-1" });
    state = applySessionEvent(state, paperSearchUpdated);
    state = applySessionEvent(state, paperReady);

    expect(shouldShowSearchResults(state)).toBe(true);
    expect(state.searchPaperId).toBe("paper-1");
    expect(state.hits[0].paper.id).toBe("paper-2");
  });

  it("keeps the previous paper's rejected candidates when moving to the next paper", () => {
    let state = emptySessionViewState();
    state = applySessionEvent(state, { type: "session_started", session_id: "session-1" });
    state = applySessionEvent(state, {
      type: "paper_ready",
      session_id: "session-1",
      search_deferred: true,
      paper: { id: "paper-1", title: "Paper 1", abstract: "", categories: [] },
      simple_search_query: "q1",
      keyword_search_query: "q1",
      fulltext_search_query: "q1",
      search_modes: ["simple"],
      trail_paper_ids: [],
      next_paper_id: null,
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });
    state = applySessionEvent(state, {
      type: "paper_search_updated",
      session_id: "session-1",
      paper_id: "paper-1",
      search: {
        hits: [{ paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
        rejected_candidates: [{ paper: { id: "paper-3", title: "Paper 3", abstract: "", categories: [] }, paper_id: "paper-3", title: "Paper 3", score: 0.4, reason: "3rd" }],
        fallback_used: false,
      },
    });
    state = applySessionEvent(state, {
      type: "paper_ready",
      session_id: "session-1",
      search_deferred: true,
      from_paper_id: "paper-1",
      paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] },
      simple_search_query: "q2",
      keyword_search_query: "q2",
      fulltext_search_query: "q2",
      search_modes: ["simple"],
      trail_paper_ids: ["paper-1"],
      next_paper_id: null,
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });

    expect(state.previousSearchPaperId).toBe("paper-1");
    expect(state.previousRejectedCandidates).toHaveLength(1);
    expect(state.previousRejectedCandidates[0].paper_id).toBe("paper-3");
  });
});
