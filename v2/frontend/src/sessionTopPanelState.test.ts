import { describe, expect, it } from "vitest";

import { getSpeakerToggleButtonClassName } from "./sessionTopPanelState";

describe("getSpeakerToggleButtonClassName", () => {
  it("keeps the normal class when playback is inactive", () => {
    expect(getSpeakerToggleButtonClassName(false)).toBe("current-session-action-button");
  });

  it("adds the active class when playback is active", () => {
    expect(getSpeakerToggleButtonClassName(true)).toBe("current-session-action-button is-active");
  });
});
