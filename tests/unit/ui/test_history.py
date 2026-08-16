# ruff: noqa: S101

"""Process-level run history and run-finalization entry writing (D9 regression)."""

from types import SimpleNamespace

import pytest

from pu_toolbox.ui import history
from pu_toolbox.ui.execution import finalize_run

pytestmark = pytest.mark.unit


def test_history_append_and_snapshot_newest_first():
    history.clear_for_tests()
    history.append({"模式": "pipeline", "状态": "completed"})
    history.append({"模式": "tuning", "状态": "cancelled"})
    snap = history.snapshot()
    assert [entry["模式"] for entry in snap] == ["tuning", "pipeline"]


def test_history_snapshot_is_a_copy():
    history.clear_for_tests()
    history.append({"模式": "pipeline"})
    snap = history.snapshot()
    snap.clear()
    assert len(history.snapshot()) == 1


def test_history_keeps_at_most_20_entries():
    history.clear_for_tests()
    for index in range(25):
        history.append({"index": index})
    assert len(history.snapshot()) == 20
    assert history.snapshot()[0]["index"] == 24
    assert history.snapshot()[-1]["index"] == 5


class _FakeRun:
    """Minimal BackgroundRun stand-in for finalize_run unit tests.

    ``future.result()`` raises RuntimeError exactly when ``result is _ERROR``,
    so the cancelled/failed tests must pass ``result=_ERROR`` — passing only
    an error flag would make result() return None and the test would crash in
    the ``completed`` branch instead of covering the exception path.
    """

    def __init__(self, result=None, cancelled=False):
        self._result = result
        self._cancelled = cancelled

    def snapshot(self):
        return SimpleNamespace(started_at="2026-08-16T00:00:00+00:00")

    @property
    def future(self):
        def result():
            if self._result is _ERROR:
                raise RuntimeError("boom")
            return self._result

        return SimpleNamespace(result=result)

    @property
    def token(self):
        return SimpleNamespace(is_cancelled=self._cancelled)


_ERROR = object()


def test_finalize_run_completed_writes_history():
    history.clear_for_tests()
    run = _FakeRun(result=SimpleNamespace(report=SimpleNamespace(provenance={"classifier": "upu"})))
    entry, analysis, error_message = finalize_run(run, "pipeline")

    assert entry["状态"] == "completed"
    assert entry["结果"] == "upu"
    assert analysis is not None and error_message is None
    assert history.snapshot()[0]["状态"] == "completed"


def test_finalize_run_cancelled_writes_history():
    history.clear_for_tests()
    run = _FakeRun(result=_ERROR, cancelled=True)
    entry, _, error_message = finalize_run(run, "tuning")

    assert entry["状态"] == "cancelled"
    assert error_message is not None
    assert history.snapshot()[0]["状态"] == "cancelled"


def test_finalize_run_failed_writes_history():
    history.clear_for_tests()
    run = _FakeRun(result=_ERROR, cancelled=False)
    entry, _, error_message = finalize_run(run, "pipeline")

    assert entry["状态"] == "failed"
    assert error_message is not None
    assert history.snapshot()[0]["状态"] == "failed"
