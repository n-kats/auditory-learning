from __future__ import annotations

import asyncio
import importlib
import sys
import types

from fastapi.testclient import TestClient


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


def test_init_upload_saves_pdf_and_records_source_url(monkeypatch, tmp_path) -> None:
    install_import_stubs(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUDITORY_LEARNING_V2_DATA_DIR", str(tmp_path / "data"))
    prompt_explain_path = tmp_path / "prompt_explain.txt"
    prompt_speak_path = tmp_path / "prompt_speak.txt"
    prompt_explain_path.write_text("説明プロンプト")
    prompt_speak_path.write_text("読み上げプロンプト")
    monkeypatch.setenv("AUDITORY_LEARNING_V2_PROMPT_EXPLAIN_PATH", str(prompt_explain_path))
    monkeypatch.setenv("AUDITORY_LEARNING_V2_PROMPT_SPEAK_PATH", str(prompt_speak_path))

    main = importlib.reload(importlib.import_module("v2_auditory_learning.main"))
    monkeypatch.setattr(main, "data_dir", tmp_path / "data")
    pages = []

    class FakePage:
        def save(self, path):
            pages.append(str(path))
            path.write_bytes(b"png")

    class FakeRepository:
        def __init__(self):
            self.upserted = None

        def create_session_id(self):
            return "session-1"

        def upsert_document(self, *args, **kwargs):
            self.upserted = {"args": args, "kwargs": kwargs}

        def is_favorited(self, request_id: str, page_num: int | None = None):
            return False

    repository = FakeRepository()
    monkeypatch.setattr(main, "get_repository", lambda: repository)
    monkeypatch.setattr(main, "wait_for_database_ready", lambda: None)
    monkeypatch.setattr(main, "convert_from_path", lambda pdf_path: [FakePage(), FakePage()])
    monkeypatch.setattr(main, "broadcast_session_snapshot", lambda request_id: None)

    client = TestClient(main.app)
    response = client.post(
        "/init/upload/",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={
            "prompt_explain_text": "  ",
            "prompt_speak_text": "読み上げ",
            "model_name": "gpt-5.4-mini",
            "reasoning_effort": "medium",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "session-1"
    assert body["source_url"] == "upload://session-1/sample.pdf"
    assert body["page_num"] == 2
    assert repository.upserted is not None
    assert repository.upserted["args"][1] == "upload://session-1/sample.pdf"
    assert pages


def test_save_uploaded_pdf_streams_in_chunks(monkeypatch, tmp_path) -> None:
    install_import_stubs(monkeypatch)
    main = importlib.import_module("v2_auditory_learning.main")

    class FakeUploadFile:
        def __init__(self):
            self.chunks = [b"abc", b"def", b""]
            self.closed = False

        async def read(self, size):
            return self.chunks.pop(0)

        async def close(self):
            self.closed = True

    upload_file = FakeUploadFile()
    destination = tmp_path / "upload.pdf"

    asyncio.run(main.save_uploaded_pdf(upload_file, destination, source_url="upload://session-1/sample.pdf"))

    assert destination.read_bytes() == b"abcdef"
    assert upload_file.closed is True


def test_save_uploaded_pdf_logs_and_raises_on_write_error(monkeypatch, tmp_path, capsys) -> None:
    install_import_stubs(monkeypatch)
    main = importlib.import_module("v2_auditory_learning.main")

    class FakeUploadFile:
        def __init__(self):
            self.chunks = [b"abc", b""]
            self.closed = False

        async def read(self, size):
            return self.chunks.pop(0)

        async def close(self):
            self.closed = True

    class BrokenOutput:
        def write(self, chunk):
            raise OSError("disk full")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeDestination:
        def open(self, mode):
            return BrokenOutput()

    upload_file = FakeUploadFile()
    destination = FakeDestination()

    try:
        asyncio.run(main.save_uploaded_pdf(upload_file, destination, source_url="upload://session-1/sample.pdf"))
    except OSError as error:
        assert str(error) == "disk full"
    else:
        raise AssertionError("OSError was not raised")

    captured = capsys.readouterr()
    assert "Failed to write upload chunk 1" in captured.err
    assert upload_file.closed is True
