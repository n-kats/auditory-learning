import { useEffect, useEffectEvent, useRef, useState } from "react";

import { clampPlaybackRate, clampVolume, loadPlaybackRate, loadVolume } from "./audioPreferences";

type UseAudioPlayerOptions = {
  src: string | null;
  onEnded: () => void;
};

export function useAudioPlayer(options: UseAudioPlayerOptions) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [speakerEnabled, setSpeakerEnabled] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(() => loadVolume(window.localStorage.getItem("v2_audio_volume")));
  const [playbackRate, setPlaybackRate] = useState(() => loadPlaybackRate(window.localStorage.getItem("v2_audio_rate")));

  const handleEnded = useEffectEvent(() => {
    setIsPlaying(false);
    options.onEnded();
  });

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }

    const handlePlay = () => {
      setIsPlaying(true);
    };
    const handlePause = () => {
      if (!audio.ended) {
        setIsPlaying(false);
      }
    };
    const handleEndedDom = () => {
      handleEnded();
    };

    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEndedDom);

    return () => {
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEndedDom);
    };
  }, [handleEnded]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    audio.volume = clampVolume(volume);
    window.localStorage.setItem("v2_audio_volume", String(volume));
  }, [volume]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    audio.playbackRate = clampPlaybackRate(playbackRate);
    window.localStorage.setItem("v2_audio_rate", String(playbackRate));
  }, [playbackRate]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    audio.volume = clampVolume(volume);
    audio.playbackRate = clampPlaybackRate(playbackRate);

    if (!options.src) {
      audio.pause();
      setIsPlaying(false);
      return;
    }

    if (!speakerEnabled) {
      audio.pause();
      setIsPlaying(false);
      return;
    }

    void audio.play().catch(() => {
      setIsPlaying(false);
    });
  }, [options.src, playbackRate, speakerEnabled, volume]);

  return {
    audioRef,
    speakerEnabled,
    setSpeakerEnabled,
    isPlaying,
    volume,
    setVolume,
    playbackRate,
    setPlaybackRate,
  };
}
