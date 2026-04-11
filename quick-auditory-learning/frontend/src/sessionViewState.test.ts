import { describe, expect, it } from "vitest";

import { applySessionEvent, emptySessionViewState, shouldIgnoreStaleSearch, shouldIgnoreStaleSearchMessage, shouldShowSearchResults } from "./sessionViewState";

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
    expect(
      shouldIgnoreStaleSearch({
        currentSessionId: "session-1",
        currentPaperId: "paper-1",
        messageSessionId: "session-1",
        messagePaperId: "paper-2",
        pendingPaperId: "paper-2",
      }),
    ).toBe(false);
    expect(
      shouldIgnoreStaleSearch({
        currentSessionId: "session-1",
        currentPaperId: "paper-1",
        messageSessionId: "session-1",
        messagePaperId: "paper-2",
        allowPendingSessionSearch: true,
      }),
    ).toBe(false);
  });

  it("does not treat next-candidate updates as stale search messages", () => {
    expect(
      shouldIgnoreStaleSearchMessage({
        messageType: "session_next_candidate_updated",
        currentSessionId: "session-1",
        currentPaperId: "paper-1",
        messageSessionId: "session-1",
        messagePaperId: "paper-2",
      }),
    ).toBe(false);
  });

  it("accepts the next paper's search update before paper_ready arrives", () => {
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
      next_paper_id: "paper-2",
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });

    const pendingSearchAllowed = shouldIgnoreStaleSearch({
      currentSessionId: state.currentSessionId,
      currentPaperId: state.currentPaper?.id ?? null,
      messageSessionId: "session-1",
      messagePaperId: "paper-2",
      pendingPaperId: state.nextPaperId,
    });
    expect(pendingSearchAllowed).toBe(false);

    state = applySessionEvent(state, {
      type: "paper_search_updated",
      session_id: "session-1",
      paper_id: "paper-2",
      next_paper_id: "paper-3",
      search: {
        hits: [{ paper: { id: "paper-3", title: "Paper 3", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
        rejected_candidates: [],
        fallback_used: false,
      },
    });

    expect(state.currentPaper?.id).toBe("paper-1");
    expect(state.searchPaperId).toBe("paper-2");
    expect(state.hits[0].paper.id).toBe("paper-3");
    expect(state.nextPaperId).toBe("paper-3");
    expect(shouldShowSearchResults(state)).toBe(false);

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
      next_paper_id: "paper-3",
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });

    expect(state.currentPaper?.id).toBe("paper-2");
    expect(state.searchPaperId).toBe("paper-2");
    expect(state.hits[0].paper.id).toBe("paper-3");
    expect(state.nextPaperId).toBe("paper-3");
    expect(shouldShowSearchResults(state)).toBe(true);
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

  it("keeps two client states aligned when one client advances the shared session", () => {
    const initialEvents: Parameters<typeof applySessionEvent>[1][] = [
      { type: "session_started", session_id: "session-1" },
      {
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
      },
      {
        type: "paper_search_updated",
        session_id: "session-1",
        paper_id: "paper-1",
        next_paper_id: "paper-2",
        search: {
          hits: [{ paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
          rejected_candidates: [{ paper: { id: "paper-3", title: "Paper 3", abstract: "", categories: [] }, paper_id: "paper-3", title: "Paper 3", score: 0.4, reason: "3rd" }],
          fallback_used: false,
        },
      },
    ];
    const advanceEvents: Parameters<typeof applySessionEvent>[1][] = [
      {
        type: "session_next_requested",
        session_id: "session-1",
        from_paper_id: "paper-1",
        to_paper_id: "paper-2",
      },
      {
        type: "session_advanced",
        session_id: "session-1",
        from_paper_id: "paper-1",
        to_paper_id: "paper-2",
      },
      {
        type: "paper_ready",
        session_id: "session-1",
        from_paper_id: "paper-1",
        search_deferred: true,
        paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] },
        simple_search_query: "q2",
        keyword_search_query: "q2",
        fulltext_search_query: "q2",
        search_modes: ["simple"],
        trail_paper_ids: ["paper-1"],
        next_paper_id: null,
        search: { hits: [], rejected_candidates: [], fallback_used: true },
      },
      {
        type: "paper_search_updated",
        session_id: "session-1",
        paper_id: "paper-2",
        next_paper_id: "paper-3",
        search: {
          hits: [{ paper: { id: "paper-3", title: "Paper 3", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
          rejected_candidates: [],
          fallback_used: false,
        },
      },
    ];

    let stateA = emptySessionViewState();
    let stateB = emptySessionViewState();
    for (const event of initialEvents) {
      stateA = applySessionEvent(stateA, event);
      stateB = applySessionEvent(stateB, event);
    }
    for (const event of advanceEvents) {
      stateA = applySessionEvent(stateA, event);
      stateB = applySessionEvent(stateB, event);
    }

    expect(stateA.currentSessionId).toBe("session-1");
    expect(stateB.currentSessionId).toBe("session-1");
    expect(stateA.currentPaper?.id).toBe("paper-2");
    expect(stateB.currentPaper?.id).toBe("paper-2");
    expect(stateA.searchPaperId).toBe("paper-2");
    expect(stateB.searchPaperId).toBe("paper-2");
    expect(stateA.nextPaperId).toBe("paper-3");
    expect(stateB.nextPaperId).toBe("paper-3");
    expect(stateA.previousSearchPaperId).toBe("paper-1");
    expect(stateB.previousSearchPaperId).toBe("paper-1");
    expect(stateA.previousRejectedCandidates).toHaveLength(1);
    expect(stateB.previousRejectedCandidates).toHaveLength(1);
  });

  it("reproduces the next-advance regression where the next paper search result arrives before paper_ready", () => {
    let stateA = emptySessionViewState();
    let stateB = emptySessionViewState();

    const sharedEvents: Parameters<typeof applySessionEvent>[1][] = [
      { type: "session_started", session_id: "session-1" },
      {
        type: "paper_ready",
        session_id: "session-1",
        search_deferred: true,
        paper: { id: "paper-1", title: "Paper 1", abstract: "", categories: [] },
        simple_search_query: "q1",
        keyword_search_query: "q1",
        fulltext_search_query: "q1",
        search_modes: ["simple"],
        trail_paper_ids: [],
        next_paper_id: "paper-2",
        search: { hits: [], rejected_candidates: [], fallback_used: true },
      },
      {
        type: "paper_search_updated",
        session_id: "session-1",
        paper_id: "paper-1",
        next_paper_id: "paper-2",
        search: {
          hits: [{ paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
          rejected_candidates: [],
          fallback_used: false,
        },
      },
      {
        type: "session_next_requested",
        session_id: "session-1",
        from_paper_id: "paper-1",
        to_paper_id: "paper-2",
      },
      {
        type: "session_advanced",
        session_id: "session-1",
        from_paper_id: "paper-1",
        to_paper_id: "paper-2",
      },
      {
        type: "paper_search_updated",
        session_id: "session-1",
        paper_id: "paper-2",
        next_paper_id: "paper-3",
        search: {
          hits: [{ paper: { id: "paper-3", title: "Paper 3", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
          rejected_candidates: [],
          fallback_used: false,
        },
      },
      {
        type: "paper_ready",
        session_id: "session-1",
        from_paper_id: "paper-1",
        search_deferred: true,
        paper: { id: "paper-2", title: "Paper 2", abstract: "", categories: [] },
        simple_search_query: "q2",
        keyword_search_query: "q2",
        fulltext_search_query: "q2",
        search_modes: ["simple"],
        trail_paper_ids: ["paper-1"],
        next_paper_id: null,
        search: { hits: [], rejected_candidates: [], fallback_used: true },
      },
    ];

    for (const event of sharedEvents) {
      stateA = applySessionEvent(stateA, event);
      stateB = applySessionEvent(stateB, event);
    }

    expect(stateA.currentPaper?.id).toBe("paper-2");
    expect(stateB.currentPaper?.id).toBe("paper-2");
    expect(stateA.searchPaperId).toBe("paper-2");
    expect(stateB.searchPaperId).toBe("paper-2");
    expect(stateA.hits[0].paper.id).toBe("paper-3");
    expect(stateB.hits[0].paper.id).toBe("paper-3");
    expect(stateA.nextPaperId).toBe("paper-3");
    expect(stateB.nextPaperId).toBe("paper-3");
    expect(shouldShowSearchResults(stateA)).toBe(true);
    expect(shouldShowSearchResults(stateB)).toBe(true);
  });
});
