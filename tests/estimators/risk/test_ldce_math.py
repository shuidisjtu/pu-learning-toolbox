# ruff: noqa: N802, N803, N806, S101

"""MATH tests for LDCEClassifier — deterministic algorithm core correctness.

Covers Algorithm 1 (MoM centroid), Eq.10 (centroid covariance),
Eq.14 (m-update), and objective/subgradient verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.estimators.risk.ldce import (
    _centroid_covariance,
    _ldce_objective,
    _ldce_subgradient,
    _mom_centroid,
    _update_m,
)


# ═════════════════════════════════════════════════════════════════════
# MATH tests — Algorithm 1: MoM centroid
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.math
class TestMomCentroid:
    """Deterministic and edge-case behaviour of _mom_centroid."""

    def test_g1_returns_simple_mean(self, rng):
        X = rng.randn(20, 3)
        m = _mom_centroid(X, g=1, rng=rng)
        np.testing.assert_allclose(m, X.mean(axis=0))

    def test_fixed_seed_reproducible(self, rng):
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        X = np.random.RandomState(7).randn(100, 4)
        m1 = _mom_centroid(X, g=5, rng=rng1)
        m2 = _mom_centroid(X, g=5, rng=rng2)
        np.testing.assert_array_equal(m1, m2)

    def test_g_exceeds_n_U_raises(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="cannot exceed"):
            _mom_centroid(X, g=5, rng=np.random.RandomState(0))

    def test_output_shape(self, rng):
        X = rng.randn(30, 5)
        m = _mom_centroid(X, g=4, rng=rng)
        assert m.shape == (5,)
        assert np.isfinite(m).all()


# ═════════════════════════════════════════════════════════════════════
# MATH tests — Eq. 10: centroid covariance
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.math
class TestCentroidCovariance:
    """Deterministic behaviour of _centroid_covariance."""

    def test_shape_and_symmetry(self, rng):
        X_U = rng.randn(30, 4)
        S = _centroid_covariance(X_U, ridge=1e-8)
        assert S.shape == (4, 4)
        np.testing.assert_allclose(S, S.T, atol=1e-10)

    def test_ridge_makes_positive_definite(self, rng):
        X_U = rng.randn(50, 3)
        S = _centroid_covariance(X_U, ridge=1.0)
        eigvals = np.linalg.eigvalsh(S)
        assert (eigvals > 0).all()

    def test_singular_input_no_nan(self):
        X_U = np.ones((10, 3))
        S = _centroid_covariance(X_U, ridge=1e-8)
        assert not np.isnan(S).any()
        assert not np.isinf(S).any()

    def test_identity_data(self):
        X_U = np.eye(5)
        S = _centroid_covariance(X_U, ridge=0.0)
        expected = np.eye(5) / 25.0 - np.ones((5, 5)) / 25.0
        np.testing.assert_allclose(S, expected, atol=1e-14)


# ═════════════════════════════════════════════════════════════════════
# MATH tests — Eq. 14: m-update
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.math
class TestMUpdate:
    """Deterministic and edge-case behaviour of _update_m."""

    def test_w_zero_returns_m_hat(self, rng):
        d = 4
        m_hat = rng.randn(d)
        S = np.eye(d)
        w = np.zeros(d)
        m = _update_m(m_hat, S, w, b=1.0)
        np.testing.assert_allclose(m, m_hat)

    def test_constraint_satisfied(self, rng):
        d = 4
        rng2 = np.random.RandomState(99)
        m_hat = rng2.randn(d)
        S = np.eye(d) + 1e-4 * np.eye(d)
        w = rng2.randn(d)
        b = 0.5
        m = _update_m(m_hat, S, w, b)
        diff = m - m_hat
        constraint_val = diff @ S @ diff
        assert constraint_val <= b + 1e-10

    def test_nonzero_w_moves_m(self, rng):
        d = 4
        m_hat = rng.randn(d)
        S = np.eye(d)
        w = np.ones(d)
        m = _update_m(m_hat, S, w, b=1.0)
        assert np.linalg.norm(m - m_hat) > 1e-10


# ═════════════════════════════════════════════════════════════════════
# MATH tests — objective and gradient
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.math
class TestObjectiveAndGradient:
    """Verify objective value and subgradient correctness."""

    def test_objective_scalar(self, rng):
        rng2 = np.random.RandomState(1)
        X_P = rng2.randn(5, 3)
        X_U = rng2.randn(10, 3)
        w = rng2.randn(3)
        m = rng2.randn(3)
        obj = _ldce_objective(w, X_P, X_U, m, n=15, k=5, h=0.3, p=0.5, reg=0.1)
        assert isinstance(obj, float)
        assert np.isfinite(obj)

    def test_gradient_matches_finite_diff(self, rng):
        rng2 = np.random.RandomState(2)
        X_P = rng2.randn(3, 3) + 2.0
        X_U = rng2.randn(6, 3) - 2.0
        w = rng2.randn(3)
        m = rng2.randn(3)
        eps = 1e-6

        g_analytic = _ldce_subgradient(
            w, X_P, X_U, m, n=9, k=3, h=0.3, p=0.5, reg=0.1
        )

        g_numeric = np.zeros(3)
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            obj_plus = _ldce_objective(
                w + e, X_P, X_U, m, n=9, k=3, h=0.3, p=0.5, reg=0.1
            )
            obj_minus = _ldce_objective(
                w - e, X_P, X_U, m, n=9, k=3, h=0.3, p=0.5, reg=0.1
            )
            g_numeric[i] = (obj_plus - obj_minus) / (2.0 * eps)

        np.testing.assert_allclose(g_analytic, g_numeric, rtol=0.05, atol=0.05)

    def test_gradient_shape(self, rng):
        rng2 = np.random.RandomState(3)
        X_P = rng2.randn(3, 5)
        X_U = rng2.randn(6, 5)
        w = rng2.randn(5)
        m = rng2.randn(5)
        g = _ldce_subgradient(
            w, X_P, X_U, m, n=9, k=3, h=0.3, p=0.5, reg=0.1
        )
        assert g.shape == (5,)
        assert np.isfinite(g).all()
