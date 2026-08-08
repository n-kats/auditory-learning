from __future__ import annotations

import os
import re
from pathlib import Path

from v2_auditory_learning.utils.voicevox_url import resolve_voicevox_url

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPO_ROOT / "_data" / "v2_auditory_learning"
DEFAULT_POSTGRES_DSN = "postgresql://v2_auditory_learning:v2_auditory_learning@postgres:5432/v2_auditory_learning"
DEFAULT_PROMPT_EXPLAIN_PATH = REPO_ROOT / "prompt_explain.txt"
DEFAULT_PROMPT_SPEAK_PATH = REPO_ROOT / "prompt_speak.txt"
DEFAULT_MODEL_NAME = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "medium"


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def normalize_reasoning_effort(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if normalized == "middle":
        return "medium"
    return normalized or "medium"


data_dir = Path(os.environ.get("AUDITORY_LEARNING_V2_DATA_DIR", str(DEFAULT_DATA_DIR)))
postgres_dsn = os.environ.get("AUDITORY_LEARNING_V2_POSTGRES_DSN", DEFAULT_POSTGRES_DSN)
prompt_explain_path = _resolve_repo_path(
    os.environ.get("AUDITORY_LEARNING_V2_PROMPT_EXPLAIN_PATH", str(DEFAULT_PROMPT_EXPLAIN_PATH))
)
prompt_speak_path = _resolve_repo_path(
    os.environ.get("AUDITORY_LEARNING_V2_PROMPT_SPEAK_PATH", str(DEFAULT_PROMPT_SPEAK_PATH))
)
prompt_path = prompt_explain_path
frontend_url = os.environ.get("AUDITORY_LEARNING_V2_FRONTEND_URL", "").strip()
frontend_host = os.environ.get("AUDITORY_LEARNING_V2_HOST", "").strip()
frontend_origin_regex = (
    rf"^http://{re.escape(frontend_host)}:\d+$" if frontend_host else ""
)
fallback_voicevox_url = os.environ.get("AUDITORY_LEARNING_V2_FALLBACK_VOICEVOX_URL", "http://voicevox:50021")
requested_voicevox_url = os.environ.get("AUDITORY_LEARNING_V2_VOICEVOX_URL")
voicevox_url = resolve_voicevox_url(requested_voicevox_url, fallback_voicevox_url)
default_model_name = os.environ.get("AUDITORY_LEARNING_V2_DEFAULT_MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
default_reasoning_effort = normalize_reasoning_effort(
    os.environ.get("AUDITORY_LEARNING_V2_DEFAULT_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
)
