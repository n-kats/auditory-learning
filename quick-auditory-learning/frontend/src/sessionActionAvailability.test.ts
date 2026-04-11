import { describe, expect, it } from "vitest";
import { canSendSessionAction } from "./sessionActionAvailability";

describe("sessionActionAvailability", () => {
  it("enables session actions only when a session is open and websocket is connected", () => {
    expect(canSendSessionAction({ currentSessionId: null, wsConnected: true })).toBe(false);
    expect(canSendSessionAction({ currentSessionId: "session-1", wsConnected: false })).toBe(false);
    expect(canSendSessionAction({ currentSessionId: "session-1", wsConnected: true })).toBe(true);
  });
});
