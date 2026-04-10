import { describe, expect, it } from "vitest";

import { buildResumePayload, buildStopPayload, shouldReconnectAfterClose } from "./sessionSocket";

describe("sessionSocket helpers", () => {
  it("builds a stop payload", () => {
    expect(buildStopPayload("session-1")).toEqual({
      type: "stop",
      session_id: "session-1",
    });
  });

  it("builds a resume payload", () => {
    expect(buildResumePayload("session-1", 42)).toEqual({
      type: "resume",
      session_id: "session-1",
      last_event_seq: 42,
    });
  });

  it("reconnects only when the session is active and the stop is not manual", () => {
    expect(shouldReconnectAfterClose(false, "session-1")).toBe(true);
    expect(shouldReconnectAfterClose(true, "session-1")).toBe(false);
    expect(shouldReconnectAfterClose(false, null)).toBe(false);
  });
});
