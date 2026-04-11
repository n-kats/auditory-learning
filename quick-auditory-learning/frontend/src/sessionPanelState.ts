import type { Paper } from "./api";

export type SessionPanelMode = "paper" | "loading" | "start";

export function getSessionPanelMode(params: { currentSessionId: string | null; currentPaper: Paper | null; loading: boolean }): SessionPanelMode {
  if (params.currentSessionId !== null && params.currentPaper === null) {
    return "loading";
  }
  if (params.currentPaper !== null) {
    return "paper";
  }
  if (params.loading) {
    return "loading";
  }
  return "start";
}

export function shouldShowSearchResultSections(sessionPanelMode: SessionPanelMode): boolean {
  return sessionPanelMode === "paper";
}
