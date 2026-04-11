import type { AppSessionState } from "./appSessionState";
import { buildSessionMessageStatePatch, type SessionMessageStatePatch } from "./sessionMessageState";
import type { SessionCosts, SessionEventMessage } from "./api";
import type { SessionOperationKind } from "./sessionOperationState";

export type SessionMessageHandlerResult = {
  patch: SessionMessageStatePatch | null;
  errorMessage: string | null;
  updateLastEventSeq: boolean;
  sessionCosts: SessionCosts | null;
  paperCosts: SessionCosts | null;
  refreshHistory: boolean;
  refreshSessions: boolean;
  operationToMarkLoading: SessionOperationKind | null;
};

export function buildSessionMessageHandlerResult(
  state: AppSessionState,
  message: SessionEventMessage,
  currentPaperId: string | null,
): SessionMessageHandlerResult {
  if (message.type === "session_costs_updated") {
    return {
      patch: null,
      errorMessage: null,
      updateLastEventSeq: true,
      sessionCosts: message.session_costs ?? null,
      paperCosts: currentPaperId && message.paper_id === currentPaperId ? message.paper_costs ?? null : null,
      refreshHistory: false,
      refreshSessions: false,
      operationToMarkLoading: null,
    };
  }

  if (message.type === "session_next_requested" || message.type === "session_advanced" || message.type === "session_regenerated") {
    return {
      patch: null,
      errorMessage: null,
      updateLastEventSeq: true,
      sessionCosts: null,
      paperCosts: null,
      refreshHistory: true,
      refreshSessions: true,
      operationToMarkLoading: message.type === "session_regenerated" ? "regenerate" : "next",
    };
  }

  if (message.type === "error") {
    return {
      patch: null,
      errorMessage: message.message ?? "session error",
      updateLastEventSeq: false,
      sessionCosts: null,
      paperCosts: null,
      refreshHistory: false,
      refreshSessions: false,
      operationToMarkLoading: null,
    };
  }

  const patch = buildSessionMessageStatePatch(state, message);
  return {
    patch,
    errorMessage: null,
    updateLastEventSeq: true,
    sessionCosts: null,
    paperCosts: null,
    refreshHistory: false,
    refreshSessions: false,
    operationToMarkLoading: null,
  };
}
