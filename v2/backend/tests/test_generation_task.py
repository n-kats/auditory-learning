from __future__ import annotations

import importlib
import sys
import types


def install_import_stubs(monkeypatch) -> None:
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

    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)

    pdf2image_module = types.ModuleType("pdf2image")
    pdf2image_module.convert_from_path = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "pdf2image", pdf2image_module)

    pil_module = types.ModuleType("PIL")
    pil_module.Image = types.SimpleNamespace(open=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "PIL", pil_module)


def load_main(monkeypatch, tmp_path):
    install_import_stubs(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUDITORY_LEARNING_V2_DATA_DIR", str(tmp_path / "data"))
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("説明プロンプト")
    monkeypatch.setenv("AUDITORY_LEARNING_V2_PROMPT_PATH", str(prompt_path))
    return importlib.import_module("v2_auditory_learning.main")


def test_generation_task_reuses_cached_result_without_regenerating(monkeypatch, tmp_path) -> None:
    main = load_main(monkeypatch, tmp_path)
    broadcasts: list[tuple[str, int]] = []

    class FakeRepository:
        def get_document(self, request_id: str):
            return {"prompt_text": "説明プロンプト", "model_name": "gpt-5.4-mini", "paper_id": "paper-1"}

        def get_result(self, request_id: str, page_num: int, *, prompt_text: str = "", model_name: str = ""):
            if prompt_text == "説明プロンプト" and model_name == "gpt-5.4-mini":
                return {"result_id": "result-1", "explanation": "cached explanation"}
            return None

        def upsert_result(self, *args, **kwargs):
            raise AssertionError("cached result should not upsert a new record")

        def record_session_usage(self, *args, **kwargs):
            raise AssertionError("cached result should not record usage")

    monkeypatch.setattr(main, "get_repository", lambda: FakeRepository())
    monkeypatch.setattr(main, "broadcast_generation_started", lambda request_id, page: broadcasts.append(("start", page)))
    monkeypatch.setattr(main, "broadcast_generation_finished", lambda request_id, page: broadcasts.append(("finish", page)))

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")
    cache_path = tmp_path / "explain.txt"
    cache_path.write_text("cached explanation")
    audio_path = tmp_path / "explain.mp3"
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(main, "generate_explanation", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not regenerate")))
    monkeypatch.setattr(main, "text_to_wav", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not synthesize audio")))

    result = main.generation_task("task:0001", image_path, cache_path, audio_path, force=False)

    assert result.explanation == "cached explanation"
    assert result.audio_status == "ready"
    assert result.audio_error is None
    assert cache_path.read_text() == "cached explanation"
    assert audio_path.read_bytes() == b"audio"
    assert broadcasts == []


def test_generation_task_regenerates_when_settings_changed(monkeypatch, tmp_path) -> None:
    main = load_main(monkeypatch, tmp_path)
    recorded_usage: list[dict[str, object]] = []
    upserted_results: list[dict[str, object]] = []
    broadcasts: list[tuple[str, int]] = []

    class FakeRepository:
        def get_document(self, request_id: str):
            return {"prompt_text": "新しいプロンプト", "model_name": "gpt-5.4-mini", "paper_id": "paper-1"}

        def get_result(self, request_id: str, page_num: int, *, prompt_text: str = "", model_name: str = ""):
            return None

        def upsert_result(self, *args, **kwargs):
            upserted_results.append({"args": args, "kwargs": kwargs})
            return {"result_id": "result-1"}

        def record_session_usage(self, *args, **kwargs):
            recorded_usage.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(main, "get_repository", lambda: FakeRepository())
    monkeypatch.setattr(main, "broadcast_generation_started", lambda request_id, page: broadcasts.append(("start", page)))
    monkeypatch.setattr(main, "broadcast_generation_finished", lambda request_id, page: broadcasts.append(("finish", page)))

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")
    cache_path = tmp_path / "explain.txt"
    cache_path.write_text("old cached explanation")
    audio_path = tmp_path / "explain.mp3"
    audio_path.write_bytes(b"old-audio")

    class FakeGptResult:
        content = "fresh explanation"
        input_tokens = 11
        output_tokens = 22

    monkeypatch.setattr(main, "generate_explanation", lambda *args, **kwargs: FakeGptResult())

    def fake_text_to_wav(explanation, speaker, audio_path_arg, max_length=250):
        audio_path_arg.write_bytes(b"fresh-audio")

    monkeypatch.setattr(main, "text_to_wav", fake_text_to_wav)

    result = main.generation_task("task:0001", image_path, cache_path, audio_path, force=False)

    assert result.explanation == "fresh explanation"
    assert result.audio_status == "ready"
    assert result.audio_error is None
    assert cache_path.read_text() == "fresh explanation"
    assert audio_path.read_bytes() == b"fresh-audio"
    assert len(upserted_results) == 1
    assert len(recorded_usage) == 1
    assert broadcasts == [("start", 1), ("finish", 1)]


def test_generation_task_force_regenerates_even_if_cached(monkeypatch, tmp_path) -> None:
    main = load_main(monkeypatch, tmp_path)
    recorded_usage: list[dict[str, object]] = []
    upserted_results: list[dict[str, object]] = []
    broadcasts: list[tuple[str, int]] = []

    class FakeRepository:
        def get_document(self, request_id: str):
            return {"prompt_text": "説明プロンプト", "model_name": "gpt-5.4-mini", "paper_id": "paper-1"}

        def get_result(self, request_id: str, page_num: int, *, prompt_text: str = "", model_name: str = ""):
            if prompt_text == "説明プロンプト" and model_name == "gpt-5.4-mini":
                return {"result_id": "result-1", "explanation": "cached explanation"}
            return None

        def upsert_result(self, *args, **kwargs):
            upserted_results.append({"args": args, "kwargs": kwargs})
            return {"result_id": "result-1"}

        def record_session_usage(self, *args, **kwargs):
            recorded_usage.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(main, "get_repository", lambda: FakeRepository())
    monkeypatch.setattr(main, "broadcast_generation_started", lambda request_id, page: broadcasts.append(("start", page)))
    monkeypatch.setattr(main, "broadcast_generation_finished", lambda request_id, page: broadcasts.append(("finish", page)))

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")
    cache_path = tmp_path / "explain.txt"
    cache_path.write_text("cached explanation")
    audio_path = tmp_path / "explain.mp3"
    audio_path.write_bytes(b"audio")

    class FakeGptResult:
        content = "fresh explanation"
        input_tokens = 11
        output_tokens = 22

    monkeypatch.setattr(main, "generate_explanation", lambda *args, **kwargs: FakeGptResult())

    def fake_text_to_wav(explanation, speaker, audio_path_arg, max_length=250):
        audio_path_arg.write_bytes(b"fresh-audio")

    monkeypatch.setattr(main, "text_to_wav", fake_text_to_wav)

    result = main.generation_task("task:0001", image_path, cache_path, audio_path, force=True)

    assert result.explanation == "fresh explanation"
    assert result.audio_status == "ready"
    assert result.audio_error is None
    assert cache_path.read_text() == "fresh explanation"
    assert audio_path.read_bytes() == b"fresh-audio"
    assert len(upserted_results) == 1
    assert len(recorded_usage) == 1
    assert broadcasts == [("start", 1), ("finish", 1)]
