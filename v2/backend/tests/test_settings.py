from pathlib import Path

from v2_auditory_learning.settings import REPO_ROOT, _resolve_repo_path, normalize_reasoning_effort


def test_resolve_repo_path_uses_repo_root_for_relative_paths() -> None:
    assert _resolve_repo_path("prompt.txt") == REPO_ROOT / "prompt.txt"
    assert _resolve_repo_path("configs/prompt.txt") == REPO_ROOT / "configs/prompt.txt"


def test_resolve_repo_path_keeps_absolute_paths() -> None:
    absolute_path = Path("/tmp/prompt.txt")
    assert _resolve_repo_path(str(absolute_path)) == absolute_path


def test_normalize_reasoning_effort_maps_middle_to_medium() -> None:
    assert normalize_reasoning_effort("middle") == "medium"


def test_normalize_reasoning_effort_defaults_to_medium() -> None:
    assert normalize_reasoning_effort("") == "medium"
