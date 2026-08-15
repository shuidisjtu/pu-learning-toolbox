"""Background execution support for the Streamlit application."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from pu_toolbox.progress import CancellationToken, ProgressUpdate

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pu-toolbox-ui")


@dataclass(frozen=True)
class RunSnapshot:
    """Thread-safe UI view of a background run."""

    status: str
    progress: ProgressUpdate
    started_at: str


class BackgroundRun:
    """A submitted future with cooperative cancellation and progress state."""

    def __init__(self, task: Callable[[CancellationToken, Callable[[ProgressUpdate], None]], Any]):
        self.token = CancellationToken()
        self.started_at = datetime.now(UTC).isoformat()
        self._lock = Lock()
        self._progress = ProgressUpdate("queued", 0, 1, "等待执行")
        self.future: Future[Any] = _EXECUTOR.submit(task, self.token, self._update)

    def _update(self, update: ProgressUpdate) -> None:
        with self._lock:
            self._progress = update

    def cancel(self) -> None:
        self.token.cancel()
        self.future.cancel()

    def snapshot(self) -> RunSnapshot:
        with self._lock:
            progress = self._progress
        if self.future.cancelled() or self.token.is_cancelled:
            status = "cancelling" if not self.future.done() else "cancelled"
        elif self.future.done():
            status = "failed" if self.future.exception() is not None else "completed"
        else:
            status = "running"
        return RunSnapshot(status, progress, self.started_at)


def submit_background(
    task: Callable[[CancellationToken, Callable[[ProgressUpdate], None]], Any],
) -> BackgroundRun:
    """Submit one UI analysis without importing Streamlit in the worker."""
    return BackgroundRun(task)
