from __future__ import annotations

import io
import json
import random
import wave
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import httpx


def split_text(text: str, max_length: int, separators: list[str]) -> list[str]:
    if len(text) <= max_length:
        return [text]

    sub = text[:max_length]
    candidates = [sub.rsplit(separator, 1)[0] + separator for separator in separators if separator in sub]
    if candidates:
        pos = max(len(candidate) for candidate in candidates)
    else:
        pos = max_length
    return [text[:pos]] + split_text(text[pos:], max_length, separators)


def chunk_text(text: str, max_length: int = 300) -> list[str]:
    return split_text(text, max_length, separators=["。", "、", ". "])


def _read_wav_bytes(data: bytes) -> tuple[wave._wave_params, bytes]:
    with wave.open(io.BytesIO(data), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    return params, frames


def _merge_wavs(chunks: list[bytes]) -> bytes:
    if not chunks:
        return b""

    params, frames = _read_wav_bytes(chunks[0])
    collected = [frames]
    for chunk in chunks[1:]:
        next_params, next_frames = _read_wav_bytes(chunk)
        if (
            next_params.nchannels != params.nchannels
            or next_params.sampwidth != params.sampwidth
            or next_params.framerate != params.framerate
            or next_params.comptype != params.comptype
            or next_params.compname != params.compname
        ):
            raise RuntimeError("voicevox audio parameters changed between chunks")
        collected.append(next_frames)

    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams(params)
        for chunk_frames in collected:
            writer.writeframes(chunk_frames)
    return output.getvalue()


def merge_wav_files(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks = [path.read_bytes() for path in inputs if path.exists()]
    output.write_bytes(_merge_wavs(chunks))


@dataclass
class VoiceVoxSpeaker:
    speaker_id: str
    url: str
    speed: float = 1.0
    volume: float = 1.0

    def create_audio_bytes(self, text: str) -> bytes:
        try:
            query_response = httpx.post(
                f"{self.url}/audio_query",
                params={"speaker": self.speaker_id, "text": text},
                timeout=120.0,
            )
            if not (200 <= query_response.status_code < 300):
                raise RuntimeError(f"voicevox api returns {query_response.status_code}")

            synthesis_payload = json.loads(query_response.content.decode("utf-8"))
            synthesis_payload["speedScale"] = self.speed
            synthesis_payload["volumeScale"] = self.volume

            synthesis_response = httpx.post(
                f"{self.url}/synthesis?speaker={self.speaker_id}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(synthesis_payload, ensure_ascii=False).encode("utf-8"),
                timeout=120.0,
            )
            if not (200 <= synthesis_response.status_code < 300):
                raise RuntimeError(f"voicevox api returns {synthesis_response.status_code}")
            return synthesis_response.content
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"voicevox request failed url={self.url} speaker={self.speaker_id} text_len={len(text)} error={exc}"
            ) from exc


@lru_cache(maxsize=8)
def _resolve_random_speaker_ids(url: str) -> tuple[str, ...]:
    response = httpx.get(f"{url}/speakers", timeout=30.0)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return ()
    target_names = ("ずんだもん", "四国めたん", "春日部つむぎ")
    speaker_ids: list[str] = []
    for speaker in payload:
        if not isinstance(speaker, dict):
            continue
        speaker_name = str(speaker.get("speaker_name") or speaker.get("name") or "")
        if not speaker_name:
            continue
        if not any(target in speaker_name for target in target_names):
            continue
        styles = speaker.get("styles")
        if not isinstance(styles, list):
            continue
        for style in styles:
            if not isinstance(style, dict):
                continue
            style_id = style.get("id")
            if style_id is None:
                continue
            speaker_ids.append(str(style_id))
            break
    return tuple(dict.fromkeys(speaker_ids))


@dataclass
class RandomVoiceVoxSpeaker:
    url: str
    fallback_speaker_id: str
    speed: float = 1.0
    volume: float = 1.0

    def _choose_speaker_id(self, key: str | None = None) -> str:
        try:
            speaker_ids = _resolve_random_speaker_ids(self.url)
        except Exception:
            speaker_ids = ()
        if speaker_ids:
            if key is None:
                return random.choice(speaker_ids)
            digest = sha256(f"{self.url}:{key}".encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % len(speaker_ids)
            return speaker_ids[index]
        return self.fallback_speaker_id

    def create_voicevox_speaker(self, key: str | None = None) -> VoiceVoxSpeaker:
        chosen_speaker_id = self._choose_speaker_id(key=key)
        return VoiceVoxSpeaker(
            speaker_id=chosen_speaker_id,
            url=self.url,
            speed=self.speed,
            volume=self.volume,
        )

    def create_audio_bytes(self, text: str) -> bytes:
        chosen_speaker = self.create_voicevox_speaker()
        return chosen_speaker.create_audio_bytes(text)


def build_voicevox_speaker(url: str, fallback_speaker_id: str, key: str | None = None, speed: float = 1.0, volume: float = 1.0) -> VoiceVoxSpeaker:
    return RandomVoiceVoxSpeaker(
        url=url,
        fallback_speaker_id=fallback_speaker_id,
        speed=speed,
        volume=volume,
    ).create_voicevox_speaker(key=key)


def text_to_wav(text: str, speaker: VoiceVoxSpeaker, output: Path, max_length: int = 300) -> None:
    texts = chunk_text(text, max_length=max_length)
    chunks = [speaker.create_audio_bytes(chunk) for chunk in texts]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_merge_wavs(chunks))


def text_chunks_to_wavs(text: str, speaker: VoiceVoxSpeaker, output_dir: Path, max_length: int = 300) -> list[Path]:
    texts = chunk_text(text, max_length=max_length)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, chunk in enumerate(texts):
        path = output_dir / f"{index:04d}.wav"
        if not path.exists():
            path.write_bytes(speaker.create_audio_bytes(chunk))
        paths.append(path)
    return paths
