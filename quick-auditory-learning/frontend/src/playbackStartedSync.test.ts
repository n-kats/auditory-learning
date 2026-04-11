import { describe, expect, it } from "vitest";

import { shouldSendPlaybackStarted } from "./playbackStartedSync";

describe("playbackStartedSync", () => {
  it("only sends playback_started after the current paper is stable", () => {
    const common = {
      currentSessionId: "session-1",
      socketOpen: true,
      audioPaused: false,
    };

    expect(
      shouldSendPlaybackStarted({
        ...common,
        currentPaperId: "paper-a",
        isPlaying: true,
        loading: false,
        reportedPaperId: null,
      }),
    ).toBe(true);

    expect(
      shouldSendPlaybackStarted({
        ...common,
        currentPaperId: "paper-a",
        isPlaying: true,
        loading: false,
        reportedPaperId: "paper-a",
      }),
    ).toBe(false);

    expect(
      shouldSendPlaybackStarted({
        ...common,
        currentPaperId: "paper-b",
        isPlaying: true,
        loading: true,
        reportedPaperId: "paper-a",
      }),
    ).toBe(false);

    expect(
      shouldSendPlaybackStarted({
        ...common,
        currentPaperId: "paper-b",
        isPlaying: true,
        loading: false,
        reportedPaperId: "paper-a",
      }),
    ).toBe(true);
  });
});
