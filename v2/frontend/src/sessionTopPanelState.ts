export function getSpeakerToggleButtonClassName(isPlaying: boolean): string {
  return `current-session-action-button${isPlaying ? " is-active" : ""}`;
}
