# ruff: noqa: N803, N806

"""Tests for LLSVM loss functions — golden values and gradient checks.

Formulas follow the official MATLAB code (method card §4.3):
- P hinge:       alpha * sum_P [max(1 - f, 0)]^2
- U hat:         beta  * sum_U exp(-5 f^2)
- U calibration: (gamma / u) * sum_U [max(A/pi * arctan(f) - t, 0)]^2
- Regularisation: (reg_lambda / 2) * ||w||^2
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.losses.llsvm import (
    calibration_loss,
    llsvm_objective,
    positive_hinge_loss,
    unlabeled_hat_loss,
)


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _numerical_gradient(func, w, eps=1e-7):
    """Central finite-difference gradient."""
    grad = np.zeros_like(w)
    for i in range(len(w)):
        w_plus = w.copy(); w_plus[i] += eps
        w_minus = w.copy(); w_minus[i] -= eps
        grad[i] = (func(w_plus) - func(w_minus)) / (2 * eps)
    return grad


# ═════════════════════════════════════════════════════════════════════
# MATH — Golden tests for each component
# ═════════════════════════════════════════════════════════════════════


class TestPositiveHingeLoss:
    """Golden values for alpha * sum [max(1 - f, 0)]^2."""

    @pytest.mark.math
    def test_golden_all_active(self):
        """All scores < 1 → all hinge terms active."""
        X_p = np.array([[1.0, 0.5]])  # 1 sample, 2 features
        w = np.array([0.0, 0.0])      # f = 0 for all
        alpha = 2.0
        loss, grad = positive_hinge_loss(X_p, w, alpha)
        # f = 0, max(1-0, 0)^2 = 1, loss = 2.0 * 1 = 2.0
        assert loss == pytest.approx(2.0)

    @pytest.mark.math
    def test_golden_none_active(self):
        """All scores >= 1 → hinge inactive, loss = 0."""
        X_p = np.array([[1.0, 0.0]])
        w = np.array([2.0, 0.0])  # f = 2.0
        alpha = 2.0
        loss, grad = positive_hinge_loss(X_p, w, alpha)
        assert loss == pytest.approx(0.0)
        np.testing.assert_array_almost_equal(grad, np.zeros(2))

    @pytest.mark.math
    def test_gradient_finite_diff(self):
        """Gradient matches finite differences."""
        X_p = np.array([[1.0, 0.5], [0.3, -0.2], [0.8, 0.1]])
        w = np.array([0.5, -0.3])
        alpha = 2.0
        _, grad = positive_hinge_loss(X_p, w, alpha)
        num_grad = _numerical_gradient(
            lambda ww: positive_hinge_loss(X_p, ww, alpha)[0], w
        )
        np.testing.assert_allclose(grad, num_grad, rtol=1e-5)


class TestUnlabeledHatLoss:
    """Golden values for beta * sum exp(-5 f^2)."""

    @pytest.mark.math
    def test_golden_zero_scores(self):
        """f = 0 → exp(0) = 1, loss = beta * n_u."""
        X_u = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = np.array([0.0, 0.0])
        beta = 1.0
        loss, grad = unlabeled_hat_loss(X_u, w, beta)
        assert loss == pytest.approx(2.0)

    @pytest.mark.math
    def test_golden_large_scores(self):
        """Large |f| → exp underflows to ~0."""
        X_u = np.array([[1.0, 0.0]])
        w = np.array([10.0, 0.0])  # f = 10, exp(-500) ≈ 0
        beta = 1.0
        loss, grad = unlabeled_hat_loss(X_u, w, beta)
        assert loss == pytest.approx(0.0, abs=1e-100)

    @pytest.mark.math
    def test_gradient_finite_diff(self):
        """Gradient matches finite differences."""
        X_u = np.array([[0.5, -0.3], [0.1, 0.7], [-0.2, 0.4]])
        w = np.array([0.3, 0.5])
        beta = 1.0
        _, grad = unlabeled_hat_loss(X_u, w, beta)
        num_grad = _numerical_gradient(
            lambda ww: unlabeled_hat_loss(X_u, ww, beta)[0], w
        )
        np.testing.assert_allclose(grad, num_grad, rtol=1e-5)


class TestCalibrationLoss:
    """Golden values for (gamma/u) * sum [max(A/pi*arctan(f) - t, 0)]^2."""

    @pytest.mark.math
    def test_golden_inactive(self):
        """When A/pi*arctan(f) <= t, loss = 0."""
        X_u = np.array([[1.0, 0.0]])
        w = np.array([0.0, 0.0])  # f = 0, arctan(0) = 0
        gamma, t, A = 10.0, 0.5, 10.0
        loss, grad = calibration_loss(X_u, w, gamma, t, A, n_unlabeled=1)
        # A/pi * arctan(0) = 0 < t=0.5, so inactive
        assert loss == pytest.approx(0.0)
        np.testing.assert_array_almost_equal(grad, np.zeros(2))

    @pytest.mark.math
    def test_golden_active(self):
        """When A/pi*arctan(f) > t, loss > 0."""
        X_u = np.array([[1.0, 0.0]])
        w = np.array([3.0, 0.0])  # f = 3
        gamma, t, A = 10.0, -2.5, 10.0
        n_u = 1
        loss, grad = calibration_loss(X_u, w, gamma, t, A, n_unlabeled=n_u)
        phi = A / np.pi * np.arctan(3.0)
        expected = gamma / n_u * max(phi - t, 0.0) ** 2
        assert loss == pytest.approx(expected)

    @pytest.mark.math
    def test_gradient_finite_diff(self):
        """Gradient matches finite differences."""
        X_u = np.array([[0.5, -0.3], [0.1, 0.7]])
        w = np.array([1.0, 0.5])
        gamma, t, A = 10.0, -2.5, 10.0
        n_u = len(X_u)
        _, grad = calibration_loss(X_u, w, gamma, t, A, n_unlabeled=n_u)
        num_grad = _numerical_gradient(
            lambda ww: calibration_loss(X_u, ww, gamma, t, A, n_u)[0], w
        )
        np.testing.assert_allclose(grad, num_grad, rtol=1e-5)


class TestLLSVMObjective:
    """Combined objective with regularisation."""

    @pytest.mark.math
    def test_golden_regularisation_only(self):
        """Empty P and U (zero-size arrays) → only L2 reg term."""
        d = 3
        X_p = np.zeros((0, d))
        X_u = np.zeros((0, d))
        w = np.array([1.0, 2.0, 3.0])
        loss, grad = llsvm_objective(
            w, X_p, X_u,
            alpha=2.0, beta=1.0, gamma=10.0,
            t=-2.5, A=10.0, reg_lambda=1.0,
        )
        expected_loss = 0.5 * np.dot(w, w)
        assert loss == pytest.approx(expected_loss)
        np.testing.assert_allclose(grad, w, rtol=1e-10)

    @pytest.mark.math
    def test_gradient_finite_diff_combined(self):
        """Full objective gradient matches finite differences."""
        X_p = np.array([[1.0, 0.5], [0.3, -0.2]])
        X_u = np.array([[0.5, -0.3], [0.1, 0.7], [-0.2, 0.4]])
        w = np.array([0.3, 0.5])
        kwargs = dict(
            alpha=2.0, beta=1.0, gamma=10.0,
            t=-2.5, A=10.0, reg_lambda=1.0,
        )
        _, grad = llsvm_objective(w, X_p, X_u, **kwargs)
        num_grad = _numerical_gradient(
            lambda ww: llsvm_objective(ww, X_p, X_u, **kwargs)[0], w
        )
        np.testing.assert_allclose(grad, num_grad, rtol=1e-5)

    @pytest.mark.math
    def test_components_sum_to_total(self):
        """Sum of three components + reg = total."""
        X_p = np.array([[1.0, 0.5], [0.3, -0.2]])
        X_u = np.array([[0.5, -0.3], [0.1, 0.7]])
        w = np.array([0.3, 0.5])
        alpha, beta, gamma = 2.0, 1.0, 10.0
        t, A, reg_lambda = -2.5, 10.0, 1.0
        n_u = len(X_u)

        l1, _ = positive_hinge_loss(X_p, w, alpha)
        l2, _ = unlabeled_hat_loss(X_u, w, beta)
        l3, _ = calibration_loss(X_u, w, gamma, t, A, n_u)
        reg = 0.5 * reg_lambda * np.dot(w, w)

        total, _ = llsvm_objective(w, X_p, X_u, alpha, beta, gamma, t, A, reg_lambda)
        assert total == pytest.approx(l1 + l2 + l3 + reg)
