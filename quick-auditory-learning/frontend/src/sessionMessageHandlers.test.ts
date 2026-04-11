import { describe, expect, it } from "vitest";
import { emptyAppSessionState } from "./appSessionState";
import { buildSessionMessageHandlerResult } from "./sessionMessageHandlers";

describe("sessionMessageHandlers", () => {
  it("classifies session_costs_updated as a costs-only update", () => {
    const result = buildSessionMessageHandlerResult(
      emptyAppSessionState(),
      {
        type: "session_costs_updated",
        session_id: "session-a",
        session_costs: {
          session_id: "session-a",
          total_elapsed_ms: 100,
          total_wall_elapsed_ms: 100,
          total_cost_usd: 1,
          items: [],
        },
      },
      "paper-a",
    );
    expect(result.sessionCosts?.session_id).toBe("session-a");
    expect(result.paperCosts).toBeNull();
    expect(result.refreshHistory).toBe(false);
    expect(result.refreshSessions).toBe(false);
    expect(result.patch).toBeNull();
  });

  it("classifies session_regenerated as a refresh-triggering update", () => {
    const result = buildSessionMessageHandlerResult(emptyAppSessionState(), { type: "session_regenerated", session_id: "session-a" }, null);
    expect(result.refreshHistory).toBe(true);
    expect(result.refreshSessions).toBe(true);
    expect(result.operationToMarkLoading).toBe("regenerate");
    expect(result.patch).toBeNull();
  });

  it("classifies session_advanced as a loading-triggering update", () => {
    const result = buildSessionMessageHandlerResult(emptyAppSessionState(), { type: "session_advanced", session_id: "session-a" }, null);
    expect(result.refreshHistory).toBe(true);
    expect(result.refreshSessions).toBe(true);
    expect(result.operationToMarkLoading).toBe("next");
    expect(result.patch).toBeNull();
  });

  it("classifies session_next_requested as a loading-triggering update", () => {
    const result = buildSessionMessageHandlerResult(emptyAppSessionState(), { type: "session_next_requested", session_id: "session-a" }, null);
    expect(result.refreshHistory).toBe(true);
    expect(result.refreshSessions).toBe(true);
    expect(result.operationToMarkLoading).toBe("next");
    expect(result.patch).toBeNull();
  });

  it("turns error messages into error patches", () => {
    const result = buildSessionMessageHandlerResult(emptyAppSessionState(), { type: "error", message: "boom" }, null);
    expect(result.errorMessage).toBe("boom");
  });
});
