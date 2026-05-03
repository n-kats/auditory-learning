from __future__ import annotations

from pathlib import Path
from queue import Queue

from v2_auditory_learning.generation_queue import GenerationTaskScheduler


def _job_args(page: int) -> tuple[Path, Path, Path]:
    return (
        Path(f"image-{page:04d}.png"),
        Path(f"cache-{page:04d}.txt"),
        Path(f"audio-{page:04d}.mp3"),
    )


def test_scheduler_prioritizes_active_page_over_prefetch() -> None:
    scheduler = GenerationTaskScheduler[str]()
    prefetch_waiter: Queue[str | Exception] = Queue()
    active_waiter: Queue[str | Exception] = Queue()

    scheduler.reserve("session:0002", _job_args(2), prefetch_waiter, priority=10)
    scheduler.reserve("session:0001", _job_args(1), active_waiter, priority=0)

    first_job = scheduler.get_next_job(timeout=0)
    assert first_job is not None
    assert first_job.task_id == "session:0001"
    assert first_job.force is False

    scheduler.complete(first_job.task_id, "page-1")
    assert active_waiter.get(timeout=0.1) == "page-1"

    second_job = scheduler.get_next_job(timeout=0)
    assert second_job is not None
    assert second_job.task_id == "session:0002"

    scheduler.complete(second_job.task_id, "page-2")
    assert prefetch_waiter.get(timeout=0.1) == "page-2"


def test_scheduler_promotes_pending_task_to_force_when_regenerate_is_reserved() -> None:
    scheduler = GenerationTaskScheduler[str]()
    cacheable_waiter: Queue[str | Exception] = Queue()
    regenerate_waiter: Queue[str | Exception] = Queue()

    scheduler.reserve("session:0003", _job_args(3), cacheable_waiter, priority=10, force=False)
    scheduler.reserve("session:0003", _job_args(3), regenerate_waiter, priority=0, force=True)

    job = scheduler.get_next_job(timeout=0)
    assert job is not None
    assert job.task_id == "session:0003"
    assert job.force is True

    scheduler.complete(job.task_id, "page-3")
    assert cacheable_waiter.get(timeout=0.1) == "page-3"
    assert regenerate_waiter.get(timeout=0.1) == "page-3"


def test_scheduler_completes_all_waiters_for_same_task() -> None:
    scheduler = GenerationTaskScheduler[str]()
    first_waiter: Queue[str | Exception] = Queue()
    second_waiter: Queue[str | Exception] = Queue()

    scheduler.reserve("session:0004", _job_args(4), first_waiter, priority=10)
    scheduler.reserve("session:0004", _job_args(4), second_waiter, priority=10)

    job = scheduler.get_next_job(timeout=0)
    assert job is not None
    assert job.task_id == "session:0004"

    scheduler.complete(job.task_id, "page-4")
    assert first_waiter.get(timeout=0.1) == "page-4"
    assert second_waiter.get(timeout=0.1) == "page-4"
