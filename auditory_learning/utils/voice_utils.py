import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydub import AudioSegment

from auditory_learning.utils.json_utils import Bson


def split_text(text: str, max_length: int, separetors: list[str]):
    if len(text) < max_length:
        return [text]

    sub = text[:max_length]
    candidates = [sub.rsplit(separetor, 1)[0] + separetor for separetor in separetors if separetor in sub]
    if candidates:
        pos = max([len(x) for x in candidates])
    else:
        pos = max_length

    return [text[:pos]] + split_text(text[pos:], max_length, separetors)


def text_to_segment(text: str, speaker: "VoiceVoxSpeaker", max_length=300):
    texts = split_text(text, max_length, separetors=["。", "、", ". "])
    return sum([speaker.create_audio_segment(text) for text in texts], AudioSegment.empty())


def text_to_wav(text: str, speaker: "VoiceVoxSpeaker", output: Path, max_length=300):
    texts = split_text(text, max_length, separetors=["。", "、", ". "])
    segments = [speaker.create_audio_segment(text) for text in texts]
    sound = sum(segments, AudioSegment.empty())
    sound.export(output, format=os.path.splitext(output.name)[-1][1:])

    print(f"done: {output}", file=sys.stderr)


@dataclass
class VoiceVoxSpeaker:
    speaker_id: str
    url: str
    speed: float = 1.0
    volume: float = 1.0

    def create_audio_segment(self, text: str) -> AudioSegment:
        response = httpx.post(f"{self.url}/audio_query", params={"speaker": self.speaker_id, "text": text})

        if not (200 <= response.status_code < 300):
            raise RuntimeError(f"voicevox api returns {response.status_code}")

        synthesis_config = Bson(response.content)
        synthesis_config["speedScale"] = self.speed
        synthesis_config["volumeScale"] = self.volume

        synthesis_response = httpx.post(
            f"{self.url}/synthesis?speaker={self.speaker_id}",
            headers={"Context-Type": "application/json"},
            data=synthesis_config.as_bytes(),
        )
        if not (200 <= synthesis_response.status_code < 300):
            raise RuntimeError(f"voicevox api returns {synthesis_response.status_code}")
        return AudioSegment.from_file(io.BytesIO(synthesis_response.content))
