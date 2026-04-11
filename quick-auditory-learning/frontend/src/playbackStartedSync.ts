export type PlaybackStartedSyncState = {
  currentSessionId: string | null;
  currentPaperId: string | null;
  isPlaying: boolean;
  loading: boolean;
  socketOpen: boolean;
  audioPaused: boolean;
  reportedPaperId: string | null;
};

export function shouldSendPlaybackStarted(state: PlaybackStartedSyncState): boolean {
  return Boolean(
    state.currentSessionId &&
      state.currentPaperId &&
      state.isPlaying &&
      !state.loading &&
      state.socketOpen &&
      !state.audioPaused &&
      state.reportedPaperId !== state.currentPaperId,
  );
}
