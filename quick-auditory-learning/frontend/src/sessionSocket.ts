export type SessionSocketStatus = "idle" | "connecting" | "connected" | "reconnecting" | "closed";

export function buildStopPayload(sessionId: string): { type: "stop"; session_id: string } {
  return { type: "stop", session_id: sessionId };
}

export function buildResumePayload(sessionId: string, lastEventSeq: number): {
  type: "resume";
  session_id: string;
  last_event_seq: number;
} {
  return {
    type: "resume",
    session_id: sessionId,
    last_event_seq: lastEventSeq,
  };
}

export function shouldReconnectAfterClose(manualStop: boolean, sessionId: string | null): boolean {
  return !manualStop && sessionId !== null;
}
