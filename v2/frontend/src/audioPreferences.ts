const DEFAULT_VOLUME = 1;
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
  return Math.min(Math.max(value, 0.25), 4);
}

export function loadVolume(rawValue: string | null): number {
  if (rawValue === null) {
    return DEFAULT_VOLUME;
  }
  const value = Number(rawValue);
  if (!Number.isFinite(value)) {
    return DEFAULT_VOLUME;
  }
  return Math.min(Math.max(value, 0), 3);
}

export function loadPlaybackRate(rawValue: string | null): number {
  if (rawValue === null) {
    return DEFAULT_RATE;
  }
  const value = Number(rawValue);
  if (!Number.isFinite(value)) {
    return DEFAULT_RATE;
  }
  return Math.min(Math.max(value, 0.25), 4);
}
