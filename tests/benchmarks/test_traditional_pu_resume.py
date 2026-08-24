"""Incremental-persist coverage: interrupted grids keep rows for a resumed run."""

from __future__ import annotations

import contextlib
import threading
import time

import numpy as np
import pandas as pd
import pytest

import benchmarks.traditional_pu.runner as runner
from benchmarks.traditional_pu.runner import load_config, run_trials
from tests.benchmarks.test_traditional_pu_benchmark_runner import _write_config

pytestmark = pytest.mark.unit


class TestIncrementalPersist:
    def test_basic_incremental_rows_hit_disk_before_completion(self, tmp_path, monkeypatch):
        """Rows are persisted as they finish — a mid-run abort does not lose them."""
        # ruff: noqa: N803  # _Slow mirrors the sklearn X/y naming convention

        class _Slow:
            def fit(self, X, y, **kwargs):
                time.sleep(0.15)
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

            def decision_function(self, X):
                return np.zeros(len(X))

            def get_params(self, **kwargs):
                return {}

            def set_params(self, **kwargs):
                return self

        monkeypatch.setitem(runner.ESTIMATOR_FACTORY, "upu", lambda params, **kwargs: _Slow())
        config = load_config(_write_config(tmp_path))
        # 2 methods × 1 prior × 2 scales × 2 kinds (scar+sar) × 2 seeds = 16 cells
        out = tmp_path / "inc"
        result: dict = {}

        def _run():
            result["trials"], result["summary"] = run_trials(
                config, results_dir=out, seed_set="development", resume=False, progress=False
            )

        t = threading.Thread(target=_run)
        t.start()
        partial_rows: list[int] = []
        while t.is_alive():
            csv_path = out / "trials.csv"
            if csv_path.exists() and csv_path.stat().st_size > 0:
                with contextlib.suppress(pd.errors.EmptyDataError, pd.errors.ParserError):
                    # reader races the writer's mid-flush rewrite; retry next poll
                    partial_rows.append(len(pd.read_csv(csv_path)))
            time.sleep(0.05)
        t.join()
        # file appeared mid-run with strictly partial rows, then completed
        assert any(0 < n < 16 for n in partial_rows)
        assert len(result["trials"]) == 16

        # a resumed run skips the persisted cells and lands on the same grid
        trials, _ = run_trials(
            config, results_dir=out, seed_set="development", resume=True, progress=False
        )
        assert len(trials) == 16
