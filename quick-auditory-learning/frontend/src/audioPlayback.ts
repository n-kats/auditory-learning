export function clampAudioVolume(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function clampAudioRate(value: number): number {
  return Math.min(4, Math.max(0.25, value));
}

export function loadAudioVolume(savedValue: string | null): number {
  return savedValue !== null ? Math.min(3, Math.max(0, parseFloat(savedValue))) : 1;
}

export function loadAudioRate(savedValue: string | null): number {
  return savedValue !== null ? Math.min(4, Math.max(0.25, parseFloat(savedValue))) : 1;
}

export function resolveAudioSourceUrl(apiBaseUrl: string, audioUrl: string | undefined): string | undefined {
  if (!audioUrl) {
    return undefined;
  }
  return audioUrl.startsWith("http") ? audioUrl : `${apiBaseUrl}${audioUrl}`;
}

export function resolveShouldAutoPlayAfterReset(options?: { shouldAutoPlay?: boolean }): boolean {
  return options?.shouldAutoPlay ?? true;
}
