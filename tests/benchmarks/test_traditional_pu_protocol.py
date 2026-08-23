"""Contract §7.4 protocol tests: data isolation (§2.1) and failure stats (§6).

Same-seed determinism across runs is already asserted in the sibling runner
tests (tests/benchmarks/test_traditional_pu_benchmark_runner.py,
TestDeterminism) and is therefore not duplicated here.
"""

# ruff: noqa: N803  # _Boom mirrors the sklearn X/y naming convention

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.traditional_pu.runner import (
    ESTIMATOR_FACTORY,
    METRIC_COLUMNS,
    load_config,
    run_trials,
)

pytestmark = pytest.mark.unit

MINI_CONFIG = {
    "schema_version": 1,
    "protocol": "traditional_pu_baseline",
    "description": "mini protocol test config",
    "seed_set_development": [0, 1],
    "seed_set_confirmation": [2, 3],
    "data": {
        "n_samples_small": 50,
        "n_samples_mid": 50,
        "n_features": 5,
        "class_priors": [0.5],
        "separation": 2.0,
        "label_frequency": 0.5,
        "sar_strength": 1.0,
        "sar_mechanism": "linear",
        "pn_ratios": ["1:1:4"],
    },
    "timeouts": {
        "upu": 60,
        "nnpu": 60,
        "pnu": 60,
        "elkan_noto": 60,
        "ldce": 60,
        "kldce": 60,
        "llsvm": 120,
    },
    "methods": {"upu": {}},
    "limitations": [],
}


@pytest.fixture()
def mini_config(tmp_path):
    """Persist and load a minimal config; returns the validated config dict."""
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(MINI_CONFIG), encoding="utf-8")
    return load_config(path)


class _Boom:
    """Estimator that always raises on fit; injected via ESTIMATOR_FACTORY."""

    def fit(self, X, y, **kwargs):
        raise RuntimeError("boom")

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def decision_function(self, X):
        return np.zeros(len(X))


class TestDataIsolation:
    def test_basic_trials_columns_carry_no_oracle_labels(self, tmp_path, mini_config):
        """Contract §2.1: the runner never persists raw labels into trial rows.

        The trials table carries the state-machine columns, scenario metadata
        and metric columns — and nothing label-like: observed labels (y_pu)
        and ground truth (y_true) stay out of the tuning table.
        """
        trials, _ = run_trials(
            mini_config,
            results_dir=tmp_path / "iso",
            seed_set="development",
            resume=False,
            progress=False,
        )
        expected = (
            {
                "algorithm",
                "scenario",
                "seed",
                "params",
                "elapsed_seconds",
                "warning_count",
                "status",
                "failure_reason",
                "mechanism",
                "class_prior",
                "label_frequency",
                "real_h",
                "pi_h_well_conditioned",
                "ece_buckets_api",
            }
            | set(METRIC_COLUMNS)
            | {
                "brier_score_unavailable_reason",
                "expected_calibration_error_unavailable_reason",
                "diag_converged",
                "diag_n_iter",
                "diag_c_estimate",
                "diag_cond_number",
                "diag_correction_fraction",
                "diag_n_epochs",
                "diag_n_acs_iter",
            }
        )
        assert set(trials.columns) == expected
        assert "y_true" not in trials.columns
        assert "y_pu" not in trials.columns
        assert "y_pu_test" not in trials.columns
        assert "y_pnu" not in trials.columns
        assert not any(str(col).startswith("y_") for col in trials.columns)


class TestPnuRowRecords:
    def test_basic_pnu_rows_carry_group_sizes(self, tmp_path, mini_config):
        """Contract §2.2: P/N/U counts per cell are recorded on the row."""
        cfg = dict(mini_config)
        cfg["methods"] = {"pnu": {}}
        trials, _ = run_trials(
            cfg,
            results_dir=tmp_path / "pnurow",
            seed_set="development",
            resume=False,
            progress=False,
        )
        assert len(trials) == 4  # 1 ratio × 2 scales × 2 seeds
        # mini config: n_samples_small == n_samples_mid == 50, ratio 1:1:4
        assert (trials["n_p"] + trials["n_n"] + trials["n_u"]).eq(50).all()
        assert (trials["n_p"] == 8).all() and (trials["n_u"] == 34).all()


class TestFailureStats:
    def test_param_failed_trials_carry_status_reason_and_zero_rate(
        self, tmp_path, mini_config, monkeypatch
    ):
        """Contract §6: a raising estimator fails the row with status + reason."""
        # Factory call signature: ESTIMATOR_FACTORY[name](params, seed=, prior=, meta=)
        monkeypatch.setitem(ESTIMATOR_FACTORY, "upu", lambda params, **kwargs: _Boom())
        trials, summary = run_trials(
            mini_config,
            results_dir=tmp_path / "fs",
            seed_set="development",
            resume=False,
            progress=False,
        )
        assert (trials["status"] == "failed").all()
        assert trials["failure_reason"].str.contains("boom").all()
        assert (summary["success_rate"] == 0.0).all()

    def test_edge_total_failure_yields_all_nan_metric_means(
        self, tmp_path, mini_config, monkeypatch
    ):
        """Contract §6: failed rows feed the rates, never the metric means.

        A group with zero success rows must report NaN means (excluded), not
        0.0 (which would read as a measured value of zero).
        """
        monkeypatch.setitem(ESTIMATOR_FACTORY, "upu", lambda params, **kwargs: _Boom())
        trials, summary = run_trials(
            mini_config,
            results_dir=tmp_path / "fs2",
            seed_set="development",
            resume=False,
            progress=False,
        )
        assert len(trials) == 8  # 1 prior × 2 scales × (scar + sar) × 2 seeds
        assert (summary["failures"] == 2).all()  # both seeds counted as failures
        for name in METRIC_COLUMNS:
            assert summary[f"mean_{name}"].isna().all(), name
            assert summary[f"std_{name}"].isna().all(), name


class TestDiagnostics:
    def test_smoke_upu_run_emits_diag_columns(self, tmp_path, mini_config):
        """Contract §4: every trial row carries diag_* columns.

        uPU exposes no diagnostic attributes, so its rows record NaN in
        each diag_* column — the column names must still exist so rows
        stay comparable across algorithms.
        """
        trials, _ = run_trials(
            mini_config,
            results_dir=tmp_path / "diag",
            seed_set="development",
            resume=False,
            progress=False,
        )
        diag_cols = [str(col) for col in trials.columns if str(col).startswith("diag_")]
        assert diag_cols
        assert trials[diag_cols].isna().all().all()  # upu: no diagnostics → NaN
        # params snapshot mirrors the config's method entry
        assert (trials["params"] == "{}").all()
