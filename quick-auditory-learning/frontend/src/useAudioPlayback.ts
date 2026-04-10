import { useEffect, useRef, useState } from "react";

import { clampAudioRate, clampAudioVolume, loadAudioRate, loadAudioVolume } from "./audioPlayback";

export function useAudioPlayback() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const shouldAutoPlayRef = useRef(true);
  const [audioUrls, setAudioUrls] = useState<string[]>([]);
  const [audioIndex, setAudioIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioDurationMs, setAudioDurationMs] = useState<number | null>(null);
  const [audioVolume, setAudioVolume] = useState<number>(() => loadAudioVolume(localStorage.getItem("audio_volume")));
  const [audioRate, setAudioRate] = useState<number>(() => loadAudioRate(localStorage.getItem("audio_rate")));

  useEffect(() => {
    if (!isPlaying || audioUrls.length === 0) {
      return;
    }
    const current = audioRef.current;
    if (!current) {
      return;
    }
    current.muted = false;
    current.volume = clampAudioVolume(audioVolume);
    current.playbackRate = audioRate;
    const tryPlay = () => {
      void current.play().catch(() => {
        // 読み込み待ちや一時的な失敗は、canplay 到達時に再試行する
      });
    };
    if (current.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      tryPlay();
      return;
    }
    const retryPlay = () => {
      tryPlay();
    };
    current.addEventListener("canplay", retryPlay, { once: true });
    current.addEventListener("loadeddata", retryPlay, { once: true });
    current.addEventListener("loadedmetadata", retryPlay, { once: true });
    tryPlay();
    return () => {
      current.removeEventListener("canplay", retryPlay);
      current.removeEventListener("loadeddata", retryPlay);
      current.removeEventListener("loadedmetadata", retryPlay);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioIndex, audioUrls, isPlaying]);

  useEffect(() => {
    localStorage.setItem("audio_volume", String(audioVolume));
    const current = audioRef.current;
    if (current) {
      current.muted = false;
      current.volume = clampAudioVolume(audioVolume);
    }
  }, [audioVolume]);

  useEffect(() => {
    localStorage.setItem("audio_rate", String(audioRate));
    const current = audioRef.current;
    if (current) current.playbackRate = audioRate;
  }, [audioRate]);

  const stopAudio = () => {
    setIsPlaying(false);
    const current = audioRef.current;
    if (current) {
      current.pause();
      current.currentTime = 0;
    }
  };

  const pauseAudio = () => {
    setIsPlaying(false);
    const current = audioRef.current;
    if (current) {
      current.pause();
    }
  };

  const resetAudio = () => {
    setAudioUrls([]);
    setAudioIndex(0);
    setAudioDurationMs(null);
    setIsPlaying(false);
    shouldAutoPlayRef.current = true;
  };

  return {
    audioRef,
    shouldAutoPlayRef,
    audioUrls,
    setAudioUrls,
    audioIndex,
    setAudioIndex,
    isPlaying,
    setIsPlaying,
    audioDurationMs,
    setAudioDurationMs,
    audioVolume,
    setAudioVolume,
    audioRate,
    setAudioRate,
    stopAudio,
    pauseAudio,
    resetAudio,
  };
}
