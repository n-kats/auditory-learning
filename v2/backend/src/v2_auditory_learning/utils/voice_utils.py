import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydub import AudioSegment

from v2_auditory_learning.utils.json_utils import Bson

VOICEVOX_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
VOICEVOX_RETRY_ATTEMPTS = 4
VOICEVOX_RETRY_SLEEP_SECONDS = 0.5
VOICEVOX_CHUNK_MAX_LENGTH = 120
VOICEVOX_CHUNK_MIN_LENGTH = 40
VOICEVOX_CHUNK_SEPARATORS = ["\n", "。", "！", "？", "、", ". "]


def split_text(text: str, max_length: int, separetors: list[str]):
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(normalized_text) <= max_length:
        return [normalized_text]

    sub = normalized_text[:max_length]
    candidates = []
    for separetor in separetors:
        pos = sub.rfind(separetor)
        if pos != -1:
            candidates.append(pos + len(separetor))

    if candidates:
        split_pos = max(candidates)
    else:
        split_pos = max_length

    head = normalized_text[:split_pos].strip()
    tail = normalized_text[split_pos:].lstrip()
    if not tail:
        return [head] if head else []
    if not head:
        return split_text(tail, max_length, separetors)
    return [head] + split_text(tail, max_length, separetors)


def _chunk_lengths(chunks: list[str]) -> list[int]:
    return [len(chunk) for chunk in chunks]


def _create_audio_segments_with_fallback(
    text: str,
    speaker: "VoiceVoxSpeaker",
    max_length: int,
    min_length: int,
) -> list[AudioSegment]:
    chunks = split_text(text, max_length, separetors=VOICEVOX_CHUNK_SEPARATORS)
    print(
        f"[INFO] VOICEVOX split max_length={max_length} min_length={min_length} "
        f"separators={VOICEVOX_CHUNK_SEPARATORS} chunk_lengths={_chunk_lengths(chunks)}",
        file=sys.stderr,
    )
    segments: list[AudioSegment] = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            segments.append(speaker.create_audio_segment(chunk))
            continue
        except Exception as exc:  # noqa: BLE001
            if len(chunk) <= min_length:
                raise
            next_max_length = max(min_length, max_length // 2)
            if next_max_length >= len(chunk):
                next_max_length = max(min_length, len(chunk) // 2)
            if next_max_length >= len(chunk):
                raise
            print(
                f"[WARN] VOICEVOX chunk failed len={len(chunk)}; retrying with max_length={next_max_length} "
                f"and split_lengths={_chunk_lengths(split_text(chunk, next_max_length, separetors=VOICEVOX_CHUNK_SEPARATORS))}: {exc}",
                file=sys.stderr,
            )
            segments.extend(
                _create_audio_segments_with_fallback(
                    chunk,
                    speaker,
                    next_max_length,
                    min_length,
                )
            )
    return segments


def text_to_segment(text: str, speaker: "VoiceVoxSpeaker", max_length=300):
    segments = _create_audio_segments_with_fallback(text, speaker, max_length, VOICEVOX_CHUNK_MIN_LENGTH)
    return sum(segments, AudioSegment.empty())


def text_to_wav(text: str, speaker: "VoiceVoxSpeaker", output: Path, max_length=VOICEVOX_CHUNK_MAX_LENGTH):
    print(
        f"[INFO] VOICEVOX text_to_wav max_length={max_length} min_length={VOICEVOX_CHUNK_MIN_LENGTH}",
        file=sys.stderr,
    )
    segments = _create_audio_segments_with_fallback(text, speaker, max_length, VOICEVOX_CHUNK_MIN_LENGTH)
    sound = sum(segments, AudioSegment.empty())
    sound.export(output, format=os.path.splitext(output.name)[-1][1:])

    print(f"done: {output}", file=sys.stderr)


@dataclass
class VoiceVoxSpeaker:
    speaker_id: str
    url: str
    speed: float = 1.0
    volume: float = 1.0

    def _post_with_retry(self, url: str, *, params: dict[str, object] | None = None, headers: dict[str, str] | None = None, data: bytes | None = None) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(VOICEVOX_RETRY_ATTEMPTS):
            try:
                response = httpx.post(
                    url,
                    params=params,
                    headers=headers,
                    data=data,
                    timeout=VOICEVOX_TIMEOUT,
                )
            except httpx.RequestError as exc:
                last_error = exc
            else:
                if 200 <= response.status_code < 300:
                    return response
                if response.status_code < 500:
                    raise RuntimeError(f"voicevox api returns {response.status_code}")
                last_error = RuntimeError(f"voicevox api returns {response.status_code}")

            if attempt + 1 < VOICEVOX_RETRY_ATTEMPTS:
                time.sleep(VOICEVOX_RETRY_SLEEP_SECONDS)

        if last_error is None:
            raise RuntimeError("voicevox request failed")
        raise last_error

    def create_audio_segment(self, text: str) -> AudioSegment:
        response = self._post_with_retry(
            f"{self.url}/audio_query",
            params={"speaker": self.speaker_id, "text": text},
        )

        synthesis_config = Bson(response.content)
        synthesis_config["speedScale"] = self.speed
        synthesis_config["volumeScale"] = self.volume

        synthesis_response = self._post_with_retry(
            f"{self.url}/synthesis?speaker={self.speaker_id}",
            headers={"Content-Type": "application/json"},
            data=synthesis_config.as_bytes(),
        )
        return AudioSegment.from_file(io.BytesIO(synthesis_response.content))
