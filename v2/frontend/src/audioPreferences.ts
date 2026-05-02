const DEFAULT_VOLUME = 0.72;
const DEFAULT_RATE = 1;

export function clampVolume(value: number): number {
  if (!Number.isFinite(value)) {
    return DEFAULT_VOLUME;
  }
  return Math.min(Math.max(value, 0), 1);
}

export function clampPlaybackRate(value: number): number {
  if (!Number.isFinite(value)) {
    return DEFAULT_RATE;
  }
  return Math.min(Math.max(value, 0.75), 2);
}

export function loadVolume(rawValue: string | null): number {
  if (rawValue === null) {
    return DEFAULT_VOLUME;
  }
  return clampVolume(Number(rawValue));
}

export function loadPlaybackRate(rawValue: string | null): number {
  if (rawValue === null) {
    return DEFAULT_RATE;
  }
  return clampPlaybackRate(Number(rawValue));
}
