import { afterEach, describe, expect, it, vi } from "vitest";

import { generateExplanation, toggleFavorite } from "./api";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("api path encoding", () => {
  it("encodes slash-containing paper ids for favorite toggles", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ paper_id: "cond-mat/0104435", favorited: true }),
    }));
    vi.stubGlobal("fetch", fetchMock);

    await toggleFavorite("cond-mat/0104435");

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/favorites/cond-mat%2F0104435/toggle"),
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("encodes slash-containing paper ids for explanation generation", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        paper_id: "cond-mat/0104435",
        title: "Title",
        explanation: "",
        audio_url: "",
        audio_urls: [],
      }),
    }));
    vi.stubGlobal("fetch", fetchMock);

    await generateExplanation("cond-mat/0104435");

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/explanations/cond-mat%2F0104435"),
      expect.objectContaining({ method: "POST" }),
    );
  });
});
