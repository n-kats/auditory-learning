import { useEffect } from "react";

import type { Paper } from "./api";

type UseMediaSessionOptions = {
  currentPaper: Paper | null;
  audioUrlsLength: number;
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
};

export function useMediaSession({
  currentPaper,
  audioUrlsLength,
  isPlaying,
  onPlay,
  onPause,
  onStop,
}: UseMediaSessionOptions) {
  useEffect(() => {
    if (!("mediaSession" in navigator)) {
      return;
    }
    const mediaSession = navigator.mediaSession;
    const handlePlayAction = () => {
      if (audioUrlsLength > 0) {
        onPlay();
      }
    };
    const handlePauseAction = () => {
      onPause();
    };
    const handleStopAction = () => {
      onStop();
    };
    mediaSession.setActionHandler("play", handlePlayAction);
    mediaSession.setActionHandler("pause", handlePauseAction);
    mediaSession.setActionHandler("stop", handleStopAction);
    mediaSession.setActionHandler("previoustrack", null);
    mediaSession.setActionHandler("nexttrack", null);
    return () => {
      mediaSession.setActionHandler("play", null);
      mediaSession.setActionHandler("pause", null);
      mediaSession.setActionHandler("stop", null);
      mediaSession.setActionHandler("previoustrack", null);
      mediaSession.setActionHandler("nexttrack", null);
    };
  }, [audioUrlsLength, onPause, onPlay, onStop]);

  useEffect(() => {
    if (!("mediaSession" in navigator)) {
      return;
    }
    const mediaSession = navigator.mediaSession;
    if (typeof MediaMetadata !== "undefined" && currentPaper !== null) {
      mediaSession.metadata = new MediaMetadata({
        title: currentPaper.title,
        artist: currentPaper.authors ?? "quick-auditory-learning",
        album: currentPaper.id,
      });
    } else {
      mediaSession.metadata = null;
    }
    mediaSession.playbackState = audioUrlsLength > 0 && isPlaying ? "playing" : "paused";
  }, [audioUrlsLength, currentPaper, isPlaying]);
}
