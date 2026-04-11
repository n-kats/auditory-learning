import type { SessionSocketStatus } from "./sessionSocket";

export function formatHeaderStatus(params: {
  databaseReady: boolean;
  wsStatus: SessionSocketStatus;
}): string {
  return [`db: ${params.databaseReady ? "ok" : "init"}`, `ws: ${params.wsStatus === "connected" ? "ok" : params.wsStatus}`].join(" / ");
}

export function formatSessionConnectionCount(sessionWebSocketConnections: number): string | null {
  if (sessionWebSocketConnections <= 0) {
    return null;
  }
  return `接続数: ${sessionWebSocketConnections}`;
}
