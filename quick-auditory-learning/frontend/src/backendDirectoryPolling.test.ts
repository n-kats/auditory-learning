import { describe, expect, it } from "vitest";
import { HEALTH_POLL_INTERVAL_MS, SESSION_REFRESH_INTERVAL_MS } from "./backendDirectoryPolling";

describe("backendDirectoryPolling", () => {
  it("uses relaxed polling intervals", () => {
    expect(HEALTH_POLL_INTERVAL_MS).toBe(10000);
    expect(SESSION_REFRESH_INTERVAL_MS).toBe(30000);
  });
});
