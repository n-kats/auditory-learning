from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from pathlib import Path
from queue import Queue
from threading import Condition
from typing import Generic, TypeVar


GenerationResultT = TypeVar("GenerationResultT")


@dataclass(frozen=True)
class GenerationJob:
    task_id: str
    args: tuple[Path, Path, Path]
    priority: int
    force: bool = False


@dataclass
class _TaskState(Generic[GenerationResultT]):
    args: tuple[Path, Path, Path]
    priority: int
    version: int
    status: str
    force: bool
    waiters: list[Queue[GenerationResultT | Exception]]


class GenerationTaskScheduler(Generic[GenerationResultT]):
    def __init__(self) -> None:
        self._condition = Condition()
        self._sequence = count()
        self._tasks: dict[str, _TaskState[GenerationResultT]] = {}
        self._heap: list[tuple[int, int, str]] = []

    def reserve(
        self,
        task_id: str,
        args: tuple[Path, Path, Path],
        waiter: Queue[GenerationResultT | Exception],
        *,
        priority: int = 10,
        force: bool = False,
    ) -> None:
        with self._condition:
            state = self._tasks.get(task_id)
            if state is None or state.status == "done":
                version = next(self._sequence)
                self._tasks[task_id] = _TaskState(
                    args=args,
                    priority=priority,
                    version=version,
                    status="pending",
                    force=force,
                    waiters=[waiter],
                )
                heappush(self._heap, (priority, version, task_id))
                self._condition.notify()
                return

            state.waiters.append(waiter)
            if state.status == "pending" and (priority < state.priority or (force and not state.force)):
                state.priority = priority
                state.force = state.force or force
                state.version = next(self._sequence)
                heappush(self._heap, (priority, state.version, task_id))
                self._condition.notify()
                return

            if force and not state.force:
                state.force = True

    def get_next_job(self, timeout: float | None = None) -> GenerationJob | None:
        with self._condition:
            if timeout is None:
                while True:
                    job = self._pop_ready_job()
                    if job is not None:
                        return job
                    self._condition.wait()

            if timeout <= 0:
                return self._pop_ready_job()

            import time

            deadline = time.monotonic() + timeout
            while True:
                job = self._pop_ready_job()
                if job is not None:
                    return job
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

    def complete(self, task_id: str, result: GenerationResultT | Exception) -> None:
        with self._condition:
            state = self._tasks.pop(task_id, None)
            if state is None:
                return
            waiters = list(state.waiters)
            state.status = "done"
        for waiter in waiters:
            waiter.put(result)

    def _pop_ready_job(self) -> GenerationJob | None:
        while self._heap:
            priority, version, task_id = heappop(self._heap)
            state = self._tasks.get(task_id)
            if state is None:
                continue
            if state.status != "pending":
                continue
            if state.version != version or state.priority != priority:
                continue
            state.status = "running"
            return GenerationJob(task_id=task_id, args=state.args, priority=state.priority, force=state.force)
        return None
