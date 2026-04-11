import { describe, expect, it } from "vitest";

import {
  clampAudioRate,
  clampAudioVolume,
  loadAudioRate,
  loadAudioVolume,
  resolveShouldAutoPlayAfterReset,
} from "./audioPlayback";
import { resolveAudioSourceUrl, toWebSocketUrl } from "./api";

describe("audioPlayback helpers", () => {
  it("clamps audio volume to the supported range", () => {
    expect(clampAudioVolume(-1)).toBe(0);
    expect(clampAudioVolume(0.4)).toBe(0.4);
    expect(clampAudioVolume(2)).toBe(1);
  });

  it("clamps audio rate to the supported range", () => {
    expect(clampAudioRate(0.1)).toBe(0.25);
    expect(clampAudioRate(1.5)).toBe(1.5);
    expect(clampAudioRate(10)).toBe(4);
  });

  it("loads saved values with the same clamps", () => {
    expect(loadAudioVolume("0.5")).toBe(0.5);
    expect(loadAudioVolume("9")).toBe(3);
    expect(loadAudioRate("0.1")).toBe(0.25);
    expect(loadAudioRate("2.5")).toBe(2.5);
  });

  it("resolves relative and absolute audio urls", () => {
    expect(resolveAudioSourceUrl("http://localhost:8000", "/audio/abc/chunks/0000")).toBe(
      "http://localhost:8000/audio/abc/chunks/0000",
    );
    expect(resolveAudioSourceUrl("http://localhost:8000", "https://example.com/audio.mp3")).toBe("https://example.com/audio.mp3");
    expect(resolveAudioSourceUrl("http://localhost:8000", undefined)).toBeUndefined();
  });

  it("builds websocket urls from the api base url", () => {
    expect(toWebSocketUrl("/sessions/ws")).toBe("ws://localhost:8000/sessions/ws");
  });

  it("defaults reset autoplay to true unless explicitly disabled", () => {
    expect(resolveShouldAutoPlayAfterReset()).toBe(true);
    expect(resolveShouldAutoPlayAfterReset({ shouldAutoPlay: true })).toBe(true);
    expect(resolveShouldAutoPlayAfterReset({ shouldAutoPlay: false })).toBe(false);
  });
});
