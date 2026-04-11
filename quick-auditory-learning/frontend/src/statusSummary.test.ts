import { describe, expect, it } from "vitest";

import { formatHeaderStatus, formatSessionConnectionCount } from "./statusSummary";

describe("statusSummary", () => {
  it("includes the session websocket connection count", () => {
    expect(
      formatHeaderStatus({
        databaseReady: true,
        wsStatus: "connected",
      }),
    ).toBe("db: ok / ws: ok");
  });

  it("formats the session connection count label", () => {
    expect(formatSessionConnectionCount(3)).toBe("接続数: 3");
  });

  it("hides zero session connection count", () => {
    expect(formatSessionConnectionCount(0)).toBeNull();
  });
});
