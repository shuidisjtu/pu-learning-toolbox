"""Metric availability via compute_metric (contract §3 可用条件)."""

import numpy as np
import pytest

from pu_toolbox.workflows._evaluation import compute_metric, extract_proba, resolve_metric_names

pytestmark = [pytest.mark.unit]


class _NoProbaClassifier:
    """Minimal stand-in: decision_function only, no predict_proba."""

    def decision_function(self, X):  # noqa: N803
        return np.asarray(X)[:, 0]

    def predict_proba(self, X):  # noqa: N803
        raise NotImplementedError


class _ProbaClassifier:
    """Minimal stand-in with genuine two-class predict_proba."""

    def decision_function(self, X):  # noqa: N803
        return np.asarray(X)[:, 0]

    def predict_proba(self, X):  # noqa: N803
        x = np.asarray(X, dtype=float)
        proba_pos = 1.0 / (1.0 + np.exp(-x[:, 0]))
        return np.stack([1.0 - proba_pos, proba_pos], axis=1)


class TestMetricAvailability:
    def test_param_brier_unavailable_without_proba(self):
        y_pu = np.array([1, 0, 1, 0])
        pred = np.array([1, 0, 1, 0])
        scores = np.array([0.9, -0.9, 0.8, -0.8])
        y_true = np.array([1, 0, 1, 0])
        value, reason = compute_metric(
            "brier_score", y_pu, pred, scores, y_true, 0.5, proba=None
        )
        assert value is None
        assert reason == "probabilistic metric requires predict_proba"
        value2, reason2 = compute_metric(
            "expected_calibration_error", y_pu, pred, scores, y_true, 0.5, proba=None
        )
        assert value2 is None
        assert reason2 == "probabilistic metric requires predict_proba"

    def test_basic_brier_uses_proba_only(self):
        y_pu = np.array([1, 0, 1, 0])
        pred = np.array([1, 0, 1, 0])
        y_true = np.array([1, 0, 1, 0])
        proba = np.array([0.9, 0.1, 0.9, 0.1])
        value, reason = compute_metric(
            "brier_score", y_pu, pred, np.array([5.0, -5.0, 5.0, -5.0]), y_true, 0.5,
            proba=proba,
        )
        # scores are adversarial but ignored: Brier must come from proba only
        assert reason is None
        assert value == pytest.approx(0.01, abs=1e-10)

    def test_edge_extract_proba_no_fallback(self):
        X = np.array([[1.0], [-1.0]])
        assert extract_proba(_NoProbaClassifier(), X) is None
        p = extract_proba(_ProbaClassifier(), X)
        assert p is not None
        assert p.shape == (2,)

    def test_basic_alias_resolution(self):
        assert resolve_metric_names(["ap", "bacc", "brier", "ece"]) == [
            "average_precision", "balanced_accuracy", "brier_score",
            "expected_calibration_error",
        ]

    def test_determ_default_metrics_unchanged(self):
        # contract §7.2: default report semantics stay additive
        assert resolve_metric_names(None) == [
            "pu_zero_one_risk", "pu_recall", "pu_estimated_precision", "pu_auc_roc",
        ]
