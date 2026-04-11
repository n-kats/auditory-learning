import type { AppSessionState } from "./appSessionState";
import {
  applyNextCandidateUpdatedToAppSessionState,
  applyPaperReadyToAppSessionState,
  applySearchUpdatedToAppSessionState,
  applySessionStartedToAppSessionState,
  applySessionStoppedToAppSessionState,
} from "./appSessionState";
import type { SessionEventMessage } from "./api";

export type SessionMessageStatePatch = {
  nextState: AppSessionState | null;
  memo?: string | null;
  shouldRefreshHistory: boolean;
  shouldRefreshSessions: boolean;
  shouldActivateSessionTab: boolean;
  shouldActivateStartTab: boolean;
  shouldClearOperationState: boolean;
  shouldStopAudio: boolean;
  shouldUpdatePlayingState: boolean;
  shouldSetPlayingFalse: boolean;
  shouldUpdateSearchPaperIdRef: boolean;
};

export function buildSessionMessageStatePatch(
  state: AppSessionState,
  message: SessionEventMessage,
): SessionMessageStatePatch | null {
  if (message.type === "error" || message.type === "session_advanced" || message.type === "session_costs_updated" || message.type === "session_regenerated") {
    return null;
  }

  if (message.type === "session_started") {
    return {
      nextState: applySessionStartedToAppSessionState(state, message.session_id ?? null),
      memo: null,
      shouldRefreshHistory: true,
      shouldRefreshSessions: true,
      shouldActivateSessionTab: true,
      shouldActivateStartTab: false,
      shouldClearOperationState: false,
      shouldStopAudio: false,
      shouldUpdatePlayingState: false,
      shouldSetPlayingFalse: false,
      shouldUpdateSearchPaperIdRef: true,
    };
  }

  if (message.type === "paper_ready") {
    if (!message.paper) {
      return null;
    }
    return {
      nextState: applyPaperReadyToAppSessionState(state, {
        ...message,
        paper: message.paper,
      }),
      memo: message.memo ?? "",
      shouldRefreshHistory: true,
      shouldRefreshSessions: true,
      shouldActivateSessionTab: true,
      shouldActivateStartTab: false,
      shouldClearOperationState: true,
      shouldStopAudio: false,
      shouldUpdatePlayingState: true,
      shouldSetPlayingFalse: false,
      shouldUpdateSearchPaperIdRef: true,
    };
  }

  if (message.type === "paper_search_updated") {
    return {
      nextState: applySearchUpdatedToAppSessionState(state, message),
      memo: null,
      shouldRefreshHistory: true,
      shouldRefreshSessions: true,
      shouldActivateSessionTab: false,
      shouldActivateStartTab: false,
      shouldClearOperationState: false,
      shouldStopAudio: false,
      shouldUpdatePlayingState: false,
      shouldSetPlayingFalse: false,
      shouldUpdateSearchPaperIdRef: true,
    };
  }

  if (message.type === "session_next_candidate_updated") {
    return {
      nextState: applyNextCandidateUpdatedToAppSessionState(state, message),
      memo: null,
      shouldRefreshHistory: false,
      shouldRefreshSessions: true,
      shouldActivateSessionTab: false,
      shouldActivateStartTab: false,
      shouldClearOperationState: false,
      shouldStopAudio: false,
      shouldUpdatePlayingState: false,
      shouldSetPlayingFalse: false,
      shouldUpdateSearchPaperIdRef: false,
    };
  }

  if (message.type === "session_stopped") {
    return {
      nextState: applySessionStoppedToAppSessionState(state),
      memo: null,
      shouldRefreshHistory: true,
      shouldRefreshSessions: true,
      shouldActivateSessionTab: false,
      shouldActivateStartTab: true,
      shouldClearOperationState: true,
      shouldStopAudio: true,
      shouldUpdatePlayingState: false,
      shouldSetPlayingFalse: false,
      shouldUpdateSearchPaperIdRef: true,
    };
  }

  return null;
}
