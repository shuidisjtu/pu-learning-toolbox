"""Shared cooperative cancellation and progress-reporting primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event

from .core.exceptions import RunCancelledError as RunCancelledError

__all__ = [
    "CancellationToken",
    "ProgressCallback",
    "ProgressUpdate",
    "RunCancelledError",
    "emit_progress",
]


class CancellationToken:
    """Thread-safe cancellation flag shared with a running workflow."""

    def __init__(self) -> None:
        self._event = Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise RunCancelledError("run cancelled by user")


@dataclass(frozen=True)
class ProgressUpdate:
    """One immutable progress snapshot."""

    stage: str
    completed: int
    total: int
    message: str

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, max(0.0, self.completed / self.total))


ProgressCallback = Callable[[ProgressUpdate], None]


def emit_progress(
    callback: ProgressCallback | None,
    *,
    stage: str,
    completed: int,
    total: int,
    message: str,
) -> None:
    """Call a progress listener when one was supplied."""
    if callback is not None:
        callback(ProgressUpdate(stage, completed, total, message))
