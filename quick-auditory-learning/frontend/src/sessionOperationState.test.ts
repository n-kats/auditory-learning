import { describe, expect, it } from "vitest";
import {
  buildSessionOperationFailurePatch,
  buildSessionOperationIdlePatch,
  buildSessionOperationStartPatch,
  resolveShouldAutoPlayOnAdvance,
} from "./sessionOperationState";

describe("sessionOperationState", () => {
  it("buildSessionOperationStartPatch initializes operation state", () => {
    expect(buildSessionOperationStartPatch("next")).toEqual({
      error: null,
      loading: true,
      pendingAction: "next",
      backendNotices: [],
      shouldAutoPlay: true,
    });
  });

  it("buildSessionOperationStartPatch can disable autoplay for start", () => {
    expect(buildSessionOperationStartPatch("start", { shouldAutoPlay: false })).toEqual({
      error: null,
      loading: true,
      pendingAction: "start",
      backendNotices: [],
      shouldAutoPlay: false,
    });
  });

  it("buildSessionOperationStartPatch can disable autoplay", () => {
    expect(buildSessionOperationStartPatch("resume", { shouldAutoPlay: false })).toEqual({
      error: null,
      loading: true,
      pendingAction: "resume",
      backendNotices: [],
      shouldAutoPlay: false,
    });
  });

  it("buildSessionOperationIdlePatch clears only the operation flags", () => {
    expect(buildSessionOperationIdlePatch()).toEqual({
      error: null,
      loading: false,
      pendingAction: "idle",
      backendNotices: [],
    });
  });

  it("buildSessionOperationFailurePatch records an error and clears loading", () => {
    expect(buildSessionOperationFailurePatch("failed")).toEqual({
      error: "failed",
      loading: false,
      pendingAction: "idle",
      backendNotices: [],
    });
  });

  it("resolveShouldAutoPlayOnAdvance keeps autoplay when playback is active", () => {
    expect(resolveShouldAutoPlayOnAdvance(true, false)).toBe(true);
    expect(resolveShouldAutoPlayOnAdvance(false, true)).toBe(true);
    expect(resolveShouldAutoPlayOnAdvance(false, false)).toBe(false);
  });
});
