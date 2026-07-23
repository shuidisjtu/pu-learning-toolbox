"""Tests for pu_toolbox.metrics.classification."""

import numpy as np
import pytest

from pu_toolbox.metrics import (
    pu_accuracy,
    pu_auc_roc,
    pu_estimated_precision,
    pu_f1,
    pu_negative_rate,
    pu_recall,
    pu_zero_one_risk,
)

# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def rng():
    return np.random.RandomState(42)


@pytest.fixture()
def pu_data(rng):
    n_p, n_u = 30, 70
    y_pu = np.array([1] * n_p + [0] * n_u)
    y_true = np.array([1] * n_p + [1] * 20 + [0] * 50)
    scores_perfect = np.where(y_true == 1, 1.0, -1.0)
    return y_pu, y_true, scores_perfect


# ── pu_zero_one_risk ────────────────────────────────────────────────────


class TestPUZeroOneRisk:
    @pytest.mark.math
    def test_all_positive_scores(self):
        # All scores > 0: FNR_P = 0, FPR_U = 1.0 → R = 1 - π
        y_pu = np.array([1, 1, 0, 0, 0])
        scores = np.array([1.0, 2.0, 0.5, 1.0, 0.1])
        pi = 0.4
        risk = pu_zero_one_risk(y_pu, scores, class_prior=pi)
        assert risk == pytest.approx(1.0 - pi, abs=1e-10)

    @pytest.mark.math
    def test_all_negative_scores(self):
        # All scores ≤ 0: FNR_P = 1.0, FPR_U = 0 → R = 2π - π = π
        y_pu = np.array([1, 1, 0, 0, 0])
        scores = np.array([-1.0, -2.0, -0.5, -1.0, -0.1])
        pi = 0.4
        risk = pu_zero_one_risk(y_pu, scores, class_prior=pi)
        assert risk == pytest.approx(pi, abs=1e-10)

    @pytest.mark.math
    def test_hand_computed_value(self):
        y_pu = np.array([1, 1, 0, 0, 0, 0])
        scores = np.array([0.5, -0.3, 0.2, -0.1, -0.5, 0.8])
        pi = 0.5
        # FNR_P: 1 of 2 positives has score <= 0 → 0.5
        # FPR_U: 2 of 4 unlabeled have score > 0 → 0.5
        # R = 2*0.5*0.5 + 0.5 - 0.5 = 0.5
        risk = pu_zero_one_risk(y_pu, scores, class_prior=pi)
        assert risk == pytest.approx(0.5, abs=1e-10)

    @pytest.mark.unit
    def test_matches_upu_private_helper(self, rng):
        from pu_toolbox.estimators.risk.upu import _pu_validation_risk

        n = 200
        y_pu = np.array([1] * 60 + [0] * 140)
        scores = rng.randn(n)
        pi = 0.3
        expected = _pu_validation_risk(scores[:60], scores[60:], pi)
        actual = pu_zero_one_risk(y_pu, scores, class_prior=pi)
        assert actual == pytest.approx(expected, abs=1e-12)

    @pytest.mark.unit
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            pu_zero_one_risk(np.array([1, 0]), np.array([0.5]), class_prior=0.5)

    @pytest.mark.unit
    def test_invalid_class_prior_raises(self):
        y_pu = np.array([1, 0])
        scores = np.array([0.5, -0.5])
        with pytest.raises(ValueError, match="class_prior"):
            pu_zero_one_risk(y_pu, scores, class_prior=0.0)
        with pytest.raises(ValueError, match="class_prior"):
            pu_zero_one_risk(y_pu, scores, class_prior=1.0)

    @pytest.mark.unit
    def test_empty_group_returns_inf(self):
        assert pu_zero_one_risk(np.array([1, 1]), np.array([0.5, 0.5]), 0.5) == np.inf
        assert pu_zero_one_risk(np.array([0, 0]), np.array([0.5, 0.5]), 0.5) == np.inf


# ── supervised metrics ──────────────────────────────────────────────────


class TestSupervisedMetrics:
    @pytest.mark.unit
    def test_perfect_accuracy(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert pu_accuracy(y_true, y_pred) == 1.0

    @pytest.mark.unit
    def test_perfect_f1(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert pu_f1(y_true, y_pred) == 1.0

    @pytest.mark.unit
    def test_perfect_auc(self):
        y_true = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        assert pu_auc_roc(y_true, scores) == 1.0

    @pytest.mark.unit
    def test_matches_sklearn(self, rng):
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        y_true = rng.randint(0, 2, size=100)
        y_pred = rng.randint(0, 2, size=100)
        scores = rng.rand(100)
        assert pu_accuracy(y_true, y_pred) == accuracy_score(y_true, y_pred)
        assert pu_f1(y_true, y_pred) == f1_score(y_true, y_pred)
        assert pu_auc_roc(y_true, scores) == roc_auc_score(y_true, scores)


# ── PU-specific observable metrics ─────────────────────────────────────


@pytest.mark.unit
class TestPUObservableMetrics:
    def test_basic_pu_recall(self):
        """PU recall from labeled positives (partial and perfect)."""
        y_pu = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
        assert pu_recall(y_pu, y_pred) == pytest.approx(0.75)

        y_pu2 = np.array([1, 1, 1, 0, 0, 0, 0])
        y_pred2 = np.array([1, 1, 1, 0, 1, 0, 0])
        assert pu_recall(y_pu2, y_pred2) == pytest.approx(1.0)

    def test_basic_pu_estimated_precision(self):
        """Estimated precision with known class prior."""
        y_pu = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
        prec = pu_estimated_precision(y_pu, y_pred, class_prior=0.5)
        assert prec == pytest.approx(1.0)

    def test_param_pu_estimated_precision_low_prior(self):
        """Lower class prior yields lower estimated precision."""
        y_pu = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        prec = pu_estimated_precision(y_pu, y_pred, class_prior=0.3)
        assert prec == pytest.approx(0.6)

    def test_basic_pu_negative_rate(self):
        """Basic negative prediction rate among unlabeled samples."""
        y_pu = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
        assert pu_negative_rate(y_pu, y_pred) == pytest.approx(5.0 / 7.0)
