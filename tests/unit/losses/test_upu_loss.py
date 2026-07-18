"""Golden tests and mathematical-property tests for uPU loss functions.

See tests/unit/losses/test_nnpu_loss.py for authority-level conventions.
"""

# ruff: noqa: N803, N806

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.losses.upu import UPULoss, _sigmoid, _softplus_stable

# ═════════════════════════════════════════════════════════════════════
# Helper
# ═════════════════════════════════════════════════════════════════════


def _σ(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


# ═════════════════════════════════════════════════════════════════════
# MATH — Golden tests
# ═════════════════════════════════════════════════════════════════════


class TestGoldenRiskValues:
    """MATH: Hand-computed expected values from du Plessis et al. 2015."""

    @pytest.mark.math
    def test_golden_logistic_risk_zero_scores(self):
        """All scores = 0. Logistic: pos_term = 0, unlabeled = softplus(0) = log(2)."""
        loss = UPULoss("logistic")
        p = np.array([0.0, 0.0])
        u = np.array([0.0, 0.0, 0.0])
        pi = 0.5

        risk = loss(p, u, class_prior=pi)

        # pos_term = −π·mean(scores) = −0.5·0 = 0
        # unlabeled_term = mean(softplus(0)) = log(1 + exp(0)) = log(2) ≈ 0.693147
        expected = float(np.log(2))
        assert risk == pytest.approx(expected)

    @pytest.mark.math
    def test_golden_logistic_risk_separated(self):
        """P scores positive (+3), U scores negative (−3)."""
        loss = UPULoss("logistic")
        p = np.array([3.0, 3.0])
        u = np.array([-3.0, -3.0, -3.0])
        pi = 0.4

        risk = loss(p, u, class_prior=pi)

        # pos_term = −0.4·3 = −1.2
        # softplus(−3) = log(1 + exp(−3)) = log(1.049787) ≈ 0.048587
        pos_term = -pi * float(np.mean(p))  # −0.4·3 = −1.2
        unl_term = float(np.mean(_softplus_stable(u)))  # ≈ 0.048587
        expected = pos_term + unl_term
        assert risk == pytest.approx(expected)

    @pytest.mark.math
    def test_golden_squared_risk_zero_scores(self):
        """Squared loss with all scores = 0."""
        loss = UPULoss("squared")
        p = np.array([0.0, 0.0])
        u = np.array([0.0, 0.0])
        pi = 0.5

        risk = loss(p, u, class_prior=pi)

        # pos_term = −0.5·0 = 0
        # unlabeled_term = ¼·mean((u + 1)²) = ¼·1² = 0.25
        # risk = 0 + 0.25 = 0.25
        assert risk == pytest.approx(0.25)

    @pytest.mark.math
    def test_deterministic_squared_risk_separated(self):
        """Squared loss: P=+1, U=−1 → minimum risk."""
        loss = UPULoss("squared")
        p = np.array([1.0, 1.0, 1.0])
        u = np.array([-1.0, -1.0, -1.0, -1.0])
        pi = 0.5

        risk = loss(p, u, class_prior=pi)

        # pos_term = −0.5·1 = −0.5
        # unlabeled_term = ¼·mean((−1 + 1)²) = 0
        # risk = −0.5
        assert risk == pytest.approx(-0.5)

    @pytest.mark.math
    def test_validation_logistic_gradient(self):
        """Analytical gradient for logistic loss with known scores."""
        loss = UPULoss("logistic")
        p = np.array([1.0, -2.0])
        u = np.array([0.5, -0.5, 0.0])
        pi = 0.3
        n_P, n_U = len(p), len(u)

        dP, dU = loss.gradient(p, u, class_prior=pi)

        # dP = −π/n_P * ones(n_P)
        assert dP.shape == (n_P,)
        np.testing.assert_array_almost_equal(dP, np.full(n_P, -pi / n_P))

        # dU = sigmoid(u) / n_U
        expected_dU = _sigmoid(u) / n_U
        np.testing.assert_array_almost_equal(dU, expected_dU)


# ═════════════════════════════════════════════════════════════════════
# PROPERTY — Mathematical invariants
# ═════════════════════════════════════════════════════════════════════


class TestRiskInvariants:
    """PROPERTY: Invariants from du Plessis et al. 2015."""

    @pytest.mark.property
    def test_better_separation_lower_risk_logistic(self):
        """Well-separated class → lower PU risk (logistic loss)."""
        loss = UPULoss("logistic")
        r_good = loss(np.array([2.0, 3.0]), np.array([-5.0, -4.0]), class_prior=0.5)
        r_bad = loss(np.array([0.0, 0.0]), np.array([0.0, 0.0]), class_prior=0.5)
        assert r_good < r_bad

    @pytest.mark.property
    def test_higher_class_prior_lower_risk(self):
        """Higher π → more negative pos_term → lower total risk."""
        loss = UPULoss("logistic")
        r_lo = loss(np.array([2.0, 2.0]), np.array([-1.0, -1.0]), class_prior=0.3)
        r_hi = loss(np.array([2.0, 2.0]), np.array([-1.0, -1.0]), class_prior=0.7)
        assert r_lo > r_hi
