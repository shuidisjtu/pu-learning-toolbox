# ruff: noqa: S101

"""Background progress and cooperative-cancellation tests."""

from threading import Event

import pytest

from pu_toolbox.progress import CancellationToken, ProgressUpdate, RunCancelledError
from pu_toolbox.ui.runtime import submit_background

pytestmark = pytest.mark.unit


def test_basic_background_run_returns_result():
    handle = submit_background(lambda token, callback: 42)
    assert handle.future.result(timeout=2) == 42
    assert handle.snapshot().status == "completed"


def test_param_background_run_publishes_progress():
    updated = Event()
    release = Event()

    def task(token, callback):
        callback(ProgressUpdate("cv", 2, 5, "fold 2"))
        updated.set()
        release.wait(timeout=2)

    handle = submit_background(task)
    assert updated.wait(timeout=2)
    snapshot = handle.snapshot()
    assert snapshot.progress.fraction == pytest.approx(0.4)
    assert snapshot.progress.message == "fold 2"
    release.set()
    handle.future.result(timeout=2)


def test_edge_cancellation_token_raises_at_safe_boundary():
    token = CancellationToken()
    token.cancel()
    with pytest.raises(RunCancelledError, match="cancelled by user"):
        token.raise_if_cancelled()


def test_deterministic_progress_fraction_is_clamped():
    assert ProgressUpdate("x", -1, 4, "low").fraction == 0.0
    assert ProgressUpdate("x", 8, 4, "high").fraction == 1.0
    assert ProgressUpdate("x", 1, 0, "empty").fraction == 0.0
