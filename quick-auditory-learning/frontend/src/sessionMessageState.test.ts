import { describe, expect, it } from "vitest";
import { emptyAppSessionState } from "./appSessionState";
import { buildSessionMessageStatePatch } from "./sessionMessageState";
import type { SessionEventMessage } from "./api";

function makePaperReadyMessage(overrides: Partial<SessionEventMessage> = {}): SessionEventMessage {
  return {
    type: "paper_ready",
    session_id: "session-a",
    paper: {
      id: "paper-a",
      title: "Paper A",
      abstract: "Abstract",
      categories: ["cs.AI"],
    },
    memo: "memo",
    audio_urls: ["/audio/a"],
    search_deferred: false,
    search: {
      hits: [],
      rejected_candidates: [],
      fallback_used: false,
    },
    ...overrides,
  };
}

describe("sessionMessageState", () => {
  it("returns a patch for paper_ready", () => {
    const patch = buildSessionMessageStatePatch(emptyAppSessionState(), makePaperReadyMessage());
    expect(patch?.nextState?.currentPaper?.id).toBe("paper-a");
    expect(patch?.shouldRefreshHistory).toBe(true);
    expect(patch?.shouldRefreshSessions).toBe(true);
    expect(patch?.shouldActivateSessionTab).toBe(true);
    expect(patch?.shouldClearOperationState).toBe(true);
    expect(patch?.shouldUpdatePlayingState).toBe(true);
  });

  it("returns a patch for paper_search_updated", () => {
    const patch = buildSessionMessageStatePatch(emptyAppSessionState(), {
      type: "paper_search_updated",
      session_id: "session-a",
      paper_id: "paper-a",
      search: {
        hits: [],
        rejected_candidates: [],
        fallback_used: false,
      },
    });
    expect(patch?.nextState?.searchPaperId).toBe("paper-a");
    expect(patch?.shouldRefreshHistory).toBe(true);
    expect(patch?.shouldRefreshSessions).toBe(true);
    expect(patch?.shouldActivateSessionTab).toBe(false);
  });

  it("returns a patch for session_next_candidate_updated", () => {
    const patch = buildSessionMessageStatePatch(emptyAppSessionState(), {
      type: "session_next_candidate_updated",
      session_id: "session-a",
      paper_id: "paper-a",
      next_paper_id: "paper-b",
    });
    expect(patch?.nextState?.nextPaperId).toBe("paper-b");
    expect(patch?.shouldRefreshHistory).toBe(false);
    expect(patch?.shouldRefreshSessions).toBe(true);
  });

  it("returns a patch for session_started", () => {
    const patch = buildSessionMessageStatePatch(emptyAppSessionState(), {
      type: "session_started",
      session_id: "session-a",
    });
    expect(patch?.nextState?.currentSessionId).toBe("session-a");
    expect(patch?.shouldRefreshHistory).toBe(true);
    expect(patch?.shouldRefreshSessions).toBe(true);
    expect(patch?.shouldActivateSessionTab).toBe(true);
    expect(patch?.shouldClearOperationState).toBe(false);
  });

  it("keeps the current paper when the same-session session_started arrives after paper_ready", () => {
    const paperReady = makePaperReadyMessage({
      search_deferred: true,
    });
    const startedPatch = buildSessionMessageStatePatch(emptyAppSessionState(), paperReady);
    const nextState = startedPatch?.nextState;
    expect(nextState?.currentPaper?.id).toBe("paper-a");

    const patch = buildSessionMessageStatePatch(nextState ?? emptyAppSessionState(), {
      type: "session_started",
      session_id: "session-a",
    });
    expect(patch?.nextState?.currentSessionId).toBe("session-a");
    expect(patch?.nextState?.currentPaper?.id).toBe("paper-a");
    expect(patch?.nextState?.searchPaperId).toBeNull();
    expect(patch?.shouldClearOperationState).toBe(false);
  });

  it("returns a patch for session_stopped", () => {
    const patch = buildSessionMessageStatePatch(emptyAppSessionState(), {
      type: "session_stopped",
      session_id: "session-a",
    });
    expect(patch?.nextState?.currentSessionId).toBeNull();
    expect(patch?.shouldStopAudio).toBe(true);
    expect(patch?.shouldActivateStartTab).toBe(true);
    expect(patch?.shouldClearOperationState).toBe(true);
  });
});
