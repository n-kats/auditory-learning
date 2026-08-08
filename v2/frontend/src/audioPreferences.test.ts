import { clampPlaybackRate, clampVolume, loadPlaybackRate, loadVolume } from "./audioPreferences";

describe("audioPreferences", () => {
  it("clamps volume to a safe range", () => {
    expect(clampVolume(-1)).toBe(0);
    expect(clampVolume(0.4)).toBe(0.4);
    expect(clampVolume(3)).toBe(1);
  });

  it("clamps playback rate to a safe range", () => {
    expect(clampPlaybackRate(0.2)).toBe(0.25);
    expect(clampPlaybackRate(1.25)).toBe(1.25);
    expect(clampPlaybackRate(4)).toBe(4);
  });

  it("loads defaults for invalid storage values", () => {
    expect(loadVolume(null)).toBe(1);
    expect(loadVolume("not-a-number")).toBe(1);
    expect(loadPlaybackRate(null)).toBe(1);
    expect(loadPlaybackRate("bad")).toBe(1);
  });

  it("loads saved values with the quick scale", () => {
    expect(loadVolume("2.5")).toBe(2.5);
    expect(loadVolume("9")).toBe(3);
    expect(loadPlaybackRate("0.1")).toBe(0.25);
    expect(loadPlaybackRate("2.5")).toBe(2.5);
    expect(loadPlaybackRate("9")).toBe(4);
  });
});
