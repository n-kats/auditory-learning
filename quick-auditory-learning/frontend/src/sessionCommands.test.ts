import { describe, expect, it } from "vitest";

import {
  buildNextSessionCommand,
  buildPlaybackStartedSessionCommand,
  buildSetNextCandidateCommand,
  buildRegenerateSessionCommand,
  buildStartSessionCommand,
  buildStopSessionCommand,
} from "./sessionCommands";

describe("sessionCommands helpers", () => {
  it("builds start command payloads", () => {
    expect(
      buildStartSessionCommand({
        sourceUrl: "https://example.com",
        modelName: "model",
        includeOldVectors: true,
        limit: 10,
        route1Weight: 0.55,
        route2Weight: 0.45,
        seed: null,
        searchModes: ["simple", "keyword_list"],
      }),
    ).toEqual({
      type: "start",
      source_url: "https://example.com",
      model_name: "model",
      include_old_vectors: true,
      limit: 10,
      route1_weight: 0.55,
      route2_weight: 0.45,
      seed: null,
      search_modes: ["simple", "keyword_list"],
    });
  });

  it("builds session and next-candidate commands", () => {
    expect(buildStopSessionCommand("session-1")).toEqual({ type: "stop", session_id: "session-1" });
    expect(buildNextSessionCommand("session-1")).toEqual({ type: "next", session_id: "session-1" });
    expect(buildRegenerateSessionCommand("session-1")).toEqual({ type: "regenerate", session_id: "session-1" });
    expect(buildPlaybackStartedSessionCommand("session-1", "paper-1")).toEqual({
      type: "playback_started",
      session_id: "session-1",
      paper_id: "paper-1",
    });
    expect(buildSetNextCandidateCommand("session-1", "paper-2")).toEqual({
      type: "set_next_candidate",
      session_id: "session-1",
      paper_id: "paper-2",
    });
  });
});
