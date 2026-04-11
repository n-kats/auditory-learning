import { describe, expect, it } from "vitest";

import { replaySessionEvents } from "./sessionReplay";
import type { SessionEventMessage, SessionSnapshot } from "./api";

describe("sessionReplay helpers", () => {
  it("reduces events into the latest replay state", () => {
    const snapshot: SessionSnapshot = {
      session_id: "session-1",
      status: "running",
      root_source_url: "https://arxiv.org/abs/1",
      root_paper_id: "p-root",
      current_paper_id: "p-current",
      next_paper_id: "p-next",
      next_event_seq: 10,
      config: {},
    };
    const events: SessionEventMessage[] = [
      { type: "session_started", seq: 1, session_id: "session-1" },
      {
        type: "paper_ready",
        seq: 2,
        session_id: "session-1",
        paper: { id: "p-current", title: "Current", abstract: "", categories: [] },
        origin: "search",
        simple_search_query: "q1",
        search_modes: ["simple"],
        trail_paper_ids: ["p-root"],
        next_paper_id: null,
        search: { hits: [], rejected_candidates: [], fallback_used: true },
        explanation: "exp",
        memo: "memo",
        audio_urls: ["/audio/1"],
        audio_duration_ms: 900,
        notices: ["API を利用できませんでした。"],
      },
      {
        type: "paper_search_updated",
        seq: 3,
        session_id: "session-1",
        paper_id: "p-current",
        simple_search_query: "q1",
        search_modes: ["simple"],
        next_paper_id: "p-next",
        notices: ["検索の更新が完了しました。"],
        search: { hits: [{ paper: { id: "p-next", title: "Next", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }], rejected_candidates: [], fallback_used: false },
      },
      {
        type: "session_next_candidate_updated",
        seq: 4,
        next_paper_id: "p-alt",
      },
      {
        type: "session_costs_updated",
        seq: 5,
        session_id: "session-1",
        session_costs: { session_id: "session-1", total_elapsed_ms: 100, total_wall_elapsed_ms: 120, total_cost_usd: 0.5, items: [] },
        paper_id: "p-current",
        paper_costs: { session_id: "session-1", total_elapsed_ms: 80, total_wall_elapsed_ms: 90, total_cost_usd: 0.2, items: [] },
      },
    ];

    const state = replaySessionEvents(snapshot, events);

    expect(state.currentSessionId).toBe("session-1");
    expect(state.currentPaper?.id).toBe("p-current");
    expect(state.currentPaperSource).toBe("search");
    expect(state.simpleSearchQuery).toBe("q1");
    expect(state.trailPaperIds).toEqual(["p-root"]);
    expect(state.nextPaperId).toBe("p-alt");
    expect(state.hits[0].paper.id).toBe("p-next");
    expect(state.memo).toBe("memo");
    expect(state.notices).toEqual(["API を利用できませんでした。", "検索の更新が完了しました。"]);
    expect(state.paperCosts?.total_cost_usd).toBe(0.2);
    expect(state.sessionCosts?.total_cost_usd).toBe(0.5);
    expect(state.audioUrls).toEqual(["/audio/1"]);
    expect(state.audioDurationMs).toBe(900);
    expect(state.lastEventSeq).toBe(5);
  });

  it("clears replay state after a stop event", () => {
    const snapshot: SessionSnapshot = {
      session_id: "session-1",
      status: "running",
      root_source_url: "https://arxiv.org/abs/1",
      root_paper_id: "p-root",
      current_paper_id: "p-current",
      next_paper_id: null,
      next_event_seq: 10,
      config: {},
    };
    const events: SessionEventMessage[] = [
      {
        type: "paper_ready",
        seq: 1,
        session_id: "session-1",
        paper: { id: "p-current", title: "Current", abstract: "", categories: [] },
        search: { hits: [], rejected_candidates: [], fallback_used: false },
        notices: ["API を利用できませんでした。"],
      },
      { type: "session_stopped", seq: 2, session_id: "session-1" },
    ];

    const state = replaySessionEvents(snapshot, events);

    expect(state.currentSessionId).toBeNull();
    expect(state.currentPaper).toBeNull();
    expect(state.audioUrls).toEqual([]);
    expect(state.notices).toEqual([]);
    expect(state.activeTab).toBe("start");
  });

  it("keeps a same-session search result even when session_started arrives after it", () => {
    const snapshot: SessionSnapshot = {
      session_id: "session-1",
      status: "running",
      root_source_url: "https://arxiv.org/abs/1",
      root_paper_id: "p-root",
      current_paper_id: "p-current",
      next_paper_id: null,
      next_event_seq: 10,
      config: {},
    };
    const events: SessionEventMessage[] = [
      {
        type: "paper_search_updated",
        seq: 1,
        session_id: "session-1",
        paper_id: "p-current",
        search: {
          hits: [{ paper: { id: "p-next", title: "Next", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
          rejected_candidates: [],
          fallback_used: false,
        },
      },
      { type: "session_started", seq: 2, session_id: "session-1" },
      {
        type: "paper_ready",
        seq: 3,
        session_id: "session-1",
        search_deferred: true,
        paper: { id: "p-current", title: "Current", abstract: "", categories: [] },
        search: { hits: [], rejected_candidates: [], fallback_used: true },
      },
    ];

    const state = replaySessionEvents(snapshot, events);

    expect(state.currentSessionId).toBe("session-1");
    expect(state.currentPaper?.id).toBe("p-current");
    expect(state.searchPaperId).toBe("p-current");
    expect(state.hits[0].paper.id).toBe("p-next");
  });

  it("carries previous rejected candidates into the next paper replay state", () => {
    const snapshot: SessionSnapshot = {
      session_id: "session-1",
      status: "running",
      root_source_url: "https://arxiv.org/abs/1",
      root_paper_id: "p-root",
      current_paper_id: "p-current",
      next_paper_id: null,
      next_event_seq: 10,
      config: {},
    };
    const events: SessionEventMessage[] = [
      {
        type: "paper_ready",
        seq: 1,
        session_id: "session-1",
        paper: { id: "p-current", title: "Current", abstract: "", categories: [] },
        origin: "search",
        search_deferred: true,
        search: { hits: [], rejected_candidates: [], fallback_used: true },
      },
      {
        type: "paper_search_updated",
        seq: 2,
        session_id: "session-1",
        paper_id: "p-current",
        search: {
          hits: [{ paper: { id: "p-next", title: "Next", abstract: "", categories: [] }, score: 1, route1_score: 1, route2_score: 1 }],
          rejected_candidates: [{ paper: { id: "p-rejected", title: "Rejected", abstract: "", categories: [] }, paper_id: "p-rejected", title: "Rejected", score: 0.4, reason: "3rd" }],
          fallback_used: false,
        },
      },
      {
        type: "paper_ready",
        seq: 3,
        session_id: "session-1",
        from_paper_id: "p-current",
        paper: { id: "p-next", title: "Next", abstract: "", categories: [] },
        origin: "search",
        search_deferred: true,
        search: { hits: [], rejected_candidates: [], fallback_used: true },
      },
    ];

    const state = replaySessionEvents(snapshot, events);

    expect(state.currentPaper?.id).toBe("p-next");
    expect(state.previousSearchPaperId).toBe("p-current");
    expect(state.previousRejectedCandidates).toHaveLength(1);
    expect(state.previousRejectedCandidates[0].paper_id).toBe("p-rejected");
  });
});
