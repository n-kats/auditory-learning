import { describe, expect, it } from "vitest";

import { joinApiUrl } from "./api";

describe("joinApiUrl", () => {
  it("joins a base URL and path", () => {
    expect(joinApiUrl("http://192.168.11.2:8000", "/init/upload/")).toBe(
      "http://192.168.11.2:8000/init/upload/",
    );
  });

  it("preserves a trailing slash on the base URL", () => {
    expect(joinApiUrl("http://192.168.11.2:8000/", "/explain/")).toBe("http://192.168.11.2:8000/explain/");
  });
});
