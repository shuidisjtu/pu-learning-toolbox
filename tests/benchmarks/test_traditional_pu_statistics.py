"""Statistics primitive tests (contract §6): rates, CI, paired diff."""

import numpy as np
import pandas as pd
import pytest

from benchmarks.traditional_pu.statistics import _ci95, paired_diff_ci, summarize

pytestmark = pytest.mark.unit


def _trials():
    return pd.DataFrame(
        {
            "algorithm": ["upu", "upu", "upu", "upu", "upu"],
            "scenario": [
                "scar-pi0.5-small",
                "scar-pi0.5-small",
                "scar-pi0.5-mid",
                "scar-pi0.5-mid",
                "scar-pi0.5-mid",
            ],
            "seed": [0, 1, 0, 1, 2],
            "status": ["success", "failed", "success", "success", "success"],
            "elapsed_seconds": [1.0, 1.5, 2.0, 2.5, 3.0],
            "warning_count": [0, 0, 1, 0, 0],
            "pu_zero_one_risk": [0.1, np.nan, 0.2, 0.3, 0.4],
            "pu_auc_roc": [0.9, np.nan, 0.8, 0.7, 0.85],
        }
    )


class TestSummarize:
    def test_basic_rates_include_failures(self):
        out = summarize(_trials(), ["pu_zero_one_risk", "pu_auc_roc"])
        scar_small = out[out["scenario"] == "scar-pi0.5-small"].iloc[0]
        assert scar_small["count"] == 2
        assert scar_small["success"] == 1
        assert scar_small["failures"] == 1
        assert scar_small["success_rate"] == pytest.approx(0.5)
        # non-success rows never enter means
        assert scar_small["mean_pu_zero_one_risk"] == pytest.approx(0.1)

    def test_ci_bounds(self):
        out = summarize(_trials(), ["pu_zero_one_risk"])
        scar_mid = out[out["scenario"] == "scar-pi0.5-mid"].iloc[0]
        assert scar_mid["ci95_low_pu_zero_one_risk"] <= scar_mid["mean_pu_zero_one_risk"]
        assert scar_mid["mean_pu_zero_one_risk"] <= scar_mid["ci95_high_pu_zero_one_risk"]

    def test_edge_inf_success_metric_sanitized(self):
        # Inf in a success-row metric must not propagate into mean/std/CI.
        trials = _trials()
        trials.loc[2, "pu_zero_one_risk"] = np.inf  # success row in scar-pi0.5-mid
        out = summarize(trials, ["pu_zero_one_risk"])
        scar_mid = out[out["scenario"] == "scar-pi0.5-mid"].iloc[0]
        # mean/CI come out finite and coherent (Inf excluded, not propagated)
        assert scar_mid["mean_pu_zero_one_risk"] == pytest.approx(0.35)
        assert np.isfinite(scar_mid["std_pu_zero_one_risk"])
        assert np.isfinite(scar_mid["ci95_low_pu_zero_one_risk"])
        assert np.isfinite(scar_mid["ci95_high_pu_zero_one_risk"])
        assert scar_mid["ci95_low_pu_zero_one_risk"] <= scar_mid["mean_pu_zero_one_risk"]
        assert scar_mid["mean_pu_zero_one_risk"] <= scar_mid["ci95_high_pu_zero_one_risk"]

    def test_determ_repeat_calls_consistent(self):
        a = summarize(_trials(), ["pu_zero_one_risk", "pu_auc_roc"])
        b = summarize(_trials(), ["pu_zero_one_risk", "pu_auc_roc"])
        pd.testing.assert_frame_equal(a, b)


class TestCi95:
    def test_edge_empty_returns_nan(self):
        lo, hi = _ci95(np.array([], dtype=float))
        assert np.isnan(lo) and np.isnan(hi)

    def test_edge_single_value_returns_mean(self):
        lo, hi = _ci95(np.array([3.5]))
        assert lo == hi == pytest.approx(3.5)

    def test_edge_zero_variance_returns_mean(self):
        lo, hi = _ci95(np.array([2.0, 2.0, 2.0]))
        assert lo == hi == pytest.approx(2.0)


class TestPairedDiff:
    def test_known_diff(self):
        mean, lo, hi = paired_diff_ci(np.array([1.0, 2.0]), np.array([2.0, 2.0]))
        assert mean == pytest.approx(0.5)  # (1, 0)/2
        assert lo <= mean <= hi

    def test_raises_on_mismatched_length(self):
        with pytest.raises(ValueError, match="equal"):
            paired_diff_ci(np.array([1.0]), np.array([1.0, 2.0]))
