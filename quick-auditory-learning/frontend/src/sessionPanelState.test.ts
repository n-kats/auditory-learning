import { describe, expect, it } from "vitest";

import { emptyAppSessionState } from "./appSessionState";
import { buildSessionMessageStatePatch } from "./sessionMessageState";
import { buildSessionOperationStartPatch } from "./sessionOperationState";
import { getSessionPanelMode, shouldShowSearchResultSections } from "./sessionPanelState";

describe("sessionPanelState", () => {
  it("shows loading while a session is starting and no paper is ready yet", () => {
    expect(getSessionPanelMode({ currentSessionId: null, currentPaper: null, loading: true })).toBe("loading");
  });

  it("shows the paper panel when current paper exists", () => {
    expect(
      getSessionPanelMode({
        currentSessionId: "session-1",
        currentPaper: { id: "paper-1", title: "Paper 1", abstract: "", categories: [] },
        loading: true,
      }),
    ).toBe("paper");
  });

  it("shows the start prompt only when there is no current paper and not loading", () => {
    expect(getSessionPanelMode({ currentSessionId: null, currentPaper: null, loading: false })).toBe("start");
  });

  it("moves from loading to paper when start flow receives session_started then paper_ready", () => {
    const startPatch = buildSessionOperationStartPatch("start", { shouldAutoPlay: false });
    let state = emptyAppSessionState();

    expect(getSessionPanelMode({ currentSessionId: state.currentSessionId, currentPaper: state.currentPaper, loading: startPatch.loading })).toBe("loading");

    const startedPatch = buildSessionMessageStatePatch(state, {
      type: "session_started",
      session_id: "session-1",
    });
    expect(startedPatch?.shouldClearOperationState).toBe(false);
    state = startedPatch?.nextState ?? state;
    expect(getSessionPanelMode({ currentSessionId: state.currentSessionId, currentPaper: state.currentPaper, loading: true })).toBe("loading");

    const readyPatch = buildSessionMessageStatePatch(state, {
      type: "paper_ready",
      session_id: "session-1",
      paper: { id: "paper-1", title: "Paper 1", abstract: "Abstract", categories: [] },
      search_deferred: true,
      search: { hits: [], rejected_candidates: [], fallback_used: true },
    });
    state = readyPatch?.nextState ?? state;
    expect(getSessionPanelMode({ currentSessionId: state.currentSessionId, currentPaper: state.currentPaper, loading: false })).toBe("paper");
  });

  it("keeps loading when a session exists but paper is not ready yet", () => {
    expect(
      getSessionPanelMode({
        currentSessionId: "session-1",
        currentPaper: null,
        loading: false,
      }),
    ).toBe("loading");
  });

  it("shows search result sections only when the paper panel is visible", () => {
    expect(shouldShowSearchResultSections("paper")).toBe(true);
    expect(shouldShowSearchResultSections("loading")).toBe(false);
    expect(shouldShowSearchResultSections("start")).toBe(false);
  });
});
