from __future__ import annotations

import json
import importlib
import sys
import types

import httpx


def install_pydub_stub(monkeypatch):
    class FakeAudioSegment:
        @classmethod
        def empty(cls):
            return cls()

        @classmethod
        def silent(cls, duration=0):
            return cls()

        @classmethod
        def from_file(cls, *args, **kwargs):
            return cls()

        def __add__(self, other):
            return self

    pydub_module = types.ModuleType("pydub")
    pydub_module.AudioSegment = FakeAudioSegment
    monkeypatch.setitem(sys.modules, "pydub", pydub_module)
    return FakeAudioSegment


def install_main_import_stubs(monkeypatch) -> None:
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)

    pdf2image_module = types.ModuleType("pdf2image")
    pdf2image_module.convert_from_path = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "pdf2image", pdf2image_module)

    pil_module = types.ModuleType("PIL")
    pil_module.Image = types.SimpleNamespace(open=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "PIL", pil_module)


def test_create_audio_segment_uses_timeout(monkeypatch) -> None:
    FakeAudioSegment = install_pydub_stub(monkeypatch)
    voice_utils = importlib.import_module("v2_auditory_learning.utils.voice_utils")
    captured: list[tuple[str, dict[str, object]]] = []

    class FakeResponse:
        def __init__(self, content: bytes):
            self.status_code = 200
            self.content = content

    def fake_post(url: str, **kwargs):
        captured.append((url, kwargs))
        if url.endswith("/audio_query"):
            return FakeResponse(json.dumps({"speedScale": 1.0, "volumeScale": 1.0}).encode("utf-8"))
        return FakeResponse(b"not-real-audio")

    monkeypatch.setattr(voice_utils.httpx, "post", fake_post)
    monkeypatch.setattr(voice_utils.AudioSegment, "from_file", lambda *args, **kwargs: FakeAudioSegment.silent(duration=1))

    speaker = voice_utils.VoiceVoxSpeaker(speaker_id="1", url="http://voicevox:50021", speed=1.5, volume=4.0)
    segment = speaker.create_audio_segment("こんにちは")

    assert isinstance(segment, FakeAudioSegment)
    assert len(captured) == 2
    assert captured[0][1]["timeout"] is voice_utils.VOICEVOX_TIMEOUT
    assert captured[1][1]["timeout"] is voice_utils.VOICEVOX_TIMEOUT


def test_create_audio_segment_retries_voicevox_5xx(monkeypatch) -> None:
    FakeAudioSegment = install_pydub_stub(monkeypatch)
    voice_utils = importlib.import_module("v2_auditory_learning.utils.voice_utils")
    call_counts = {"audio_query": 0, "synthesis": 0}

    class FakeResponse:
        def __init__(self, status_code: int, content: bytes):
            self.status_code = status_code
            self.content = content

    def fake_post(url: str, **kwargs):
        if url.endswith("/audio_query"):
            call_counts["audio_query"] += 1
            return FakeResponse(200, json.dumps({"speedScale": 1.0, "volumeScale": 1.0}).encode("utf-8"))
        call_counts["synthesis"] += 1
        if call_counts["synthesis"] < 4:
            return FakeResponse(500, b"")
        return FakeResponse(200, b"not-real-audio")

    monkeypatch.setattr(voice_utils.httpx, "post", fake_post)
    monkeypatch.setattr(voice_utils.AudioSegment, "from_file", lambda *args, **kwargs: FakeAudioSegment.silent(duration=1))
    monkeypatch.setattr(voice_utils.time, "sleep", lambda *args, **kwargs: None)

    speaker = voice_utils.VoiceVoxSpeaker(speaker_id="1", url="http://voicevox:50021", speed=1.5, volume=4.0)
    segment = speaker.create_audio_segment("こんにちは")

    assert isinstance(segment, FakeAudioSegment)
    assert call_counts["audio_query"] == 1
    assert call_counts["synthesis"] == 4


def test_split_text_breaks_long_text_on_newlines_and_sentences(monkeypatch) -> None:
    install_pydub_stub(monkeypatch)
    voice_utils = importlib.import_module("v2_auditory_learning.utils.voice_utils")

    text = "あ" * 80 + "\n" + "い" * 80 + "。" + "う" * 80
    chunks = voice_utils.split_text(text, 120, separetors=["\n", "。", "、", ". "])

    assert len(chunks) >= 2
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert "\n" not in chunks[0] or chunks[0].endswith("\n")


def test_text_to_wav_retries_with_shorter_chunks_when_voicevox_fails(monkeypatch, tmp_path) -> None:
    FakeAudioSegment = install_pydub_stub(monkeypatch)
    voice_utils = importlib.import_module("v2_auditory_learning.utils.voice_utils")
    attempts: list[str] = []
    exports: list[tuple[str, str]] = []

    def fake_export(self, output, format):
        exports.append((str(output), format))

    monkeypatch.setattr(voice_utils.AudioSegment, "export", fake_export, raising=False)

    class FakeSpeaker:
        def create_audio_segment(self, text: str):
            attempts.append(text)
            if len(text) > 60:
                raise RuntimeError("voicevox api returns 500")
            return FakeAudioSegment.silent(duration=1)

    output = tmp_path / "audio.mp3"
    text = "あ" * 80 + "\n" + "い" * 80

    voice_utils.text_to_wav(text, FakeSpeaker(), output, max_length=120)

    assert output.name == "audio.mp3"
    assert len(exports) == 1
    assert any(len(text_chunk) <= 60 for text_chunk in attempts)
    assert attempts[0] != attempts[-1]


def test_generation_task_keeps_explanation_when_audio_fails(monkeypatch, tmp_path) -> None:
    install_pydub_stub(monkeypatch)
    install_main_import_stubs(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUDITORY_LEARNING_V2_DATA_DIR", str(tmp_path / "data"))
    prompt_explain_path = tmp_path / "prompt_explain.txt"
    prompt_speak_path = tmp_path / "prompt_speak.txt"
    prompt_explain_path.write_text("説明プロンプト")
    prompt_speak_path.write_text("読み上げプロンプト")
    monkeypatch.setenv("AUDITORY_LEARNING_V2_PROMPT_EXPLAIN_PATH", str(prompt_explain_path))
    monkeypatch.setenv("AUDITORY_LEARNING_V2_PROMPT_SPEAK_PATH", str(prompt_speak_path))

    main = importlib.import_module("v2_auditory_learning.main")

    class FakeRepository:
        def get_document(self, request_id: str):
            return {
                "prompt_explain_text": "説明プロンプト",
                "prompt_speak_text": "読み上げプロンプト",
                "model_name": "gpt-5.4-mini",
                "paper_id": "paper-1",
            }

        def get_result(
            self,
            request_id: str,
            page_num: int,
            *,
            prompt_explain_text: str = "",
            prompt_speak_text: str = "",
            model_name: str = "",
        ):
            return None

        def upsert_result(self, *args, **kwargs):
            return {"result_id": "result-1"}

        def record_session_usage(self, *args, **kwargs):
            return {"request_id": "session-1"}

    monkeypatch.setattr(main, "get_repository", lambda: FakeRepository())

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")
    cache_path = tmp_path / "explain.txt"
    audio_path = tmp_path / "explain.mp3"

    class FakeGptResult:
        content = "説明文"
        input_tokens = 100
        output_tokens = 200

    monkeypatch.setattr(
        main,
        "generate_explanation",
        lambda *args, **kwargs: FakeGptResult(),
    )
    monkeypatch.setattr(main, "generate_speech_text", lambda *args, **kwargs: FakeGptResult())

    def fail_text_to_wav(*args, **kwargs):
        raise RuntimeError("voicevox timeout")

    monkeypatch.setattr(main, "text_to_wav", fail_text_to_wav)

    result = main.generation_task("task", image_path, cache_path, audio_path)

    assert result.explanation == "説明文"
    assert result.audio_status == "failed"
    assert "voicevox timeout" in (result.audio_error or "")
    assert cache_path.read_text() == "説明文"
    assert not audio_path.exists()
