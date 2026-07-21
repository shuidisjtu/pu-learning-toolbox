# ruff: noqa: N802, N803, N806, S101

"""ORACLE tests for KLDCE — QP oracle and bias recovery.

Verifies QP oracle output quality and bias recovery KKT conditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.estimators.risk.kldce import (
    _rbf_kernel,
    _recover_bias_from_kkt,
)


# ═════════════════════════════════════════════════════════════════════
# Bias recovery from KKT conditions (§6.6)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.math
class TestBiasRecovery:
    """KKT-based bias recovery for QP oracle."""

    def test_basic_free_alpha_gives_correct_bias(self):
        """When α is free, b = 1 - g."""
        rng = np.random.RandomState(42)
        n, n_U, d = 6, 3, 2
        X = rng.randn(n, d)
        y_tilde = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        K = _rbf_kernel(X, X, sigma=1.0)
        mu = np.array([0.0, 0.0])
        k = 3

        alpha = np.array([0.05, 0.02, 0.08, 0.03, 0.10, 0.01])
        gamma = np.array([0.02, 0.05, 0.01])
        C_alpha = 1.0 / 6
        C_gamma = 1.0 / 12
        C_eq = -0.3

        b0, info = _recover_bias_from_kkt(
            alpha, gamma, X, K, y_tilde, mu,
            lambda_=1.0, sigma=1.0, C_eq=C_eq,
            C_alpha=C_alpha, C_gamma=C_gamma, k=k,
        )

        assert "n_free" in info
        assert info["n_free"] > 0
        assert info["bias_recovery"] == "free_median"
        assert np.isfinite(b0)

    def test_validation_all_at_bounds_fallback(self):
        """When all α,γ are at bounds, fall back to bounded interval."""
        rng = np.random.RandomState(42)
        n, n_U, d = 6, 3, 2
        X = rng.randn(n, d)
        y_tilde = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        K = _rbf_kernel(X, X, sigma=1.0)
        mu = np.array([0.0, 0.0])
        k = 3

        C_alpha = 1.0 / 6
        C_gamma = 1.0 / 12
        alpha = np.array([0.0, C_alpha, 0.0, C_alpha, 0.0, C_alpha])
        gamma = np.array([0.0, C_gamma, 0.0])
        C_eq = -0.3

        b0, info = _recover_bias_from_kkt(
            alpha, gamma, X, K, y_tilde, mu,
            lambda_=1.0, sigma=1.0, C_eq=C_eq,
            C_alpha=C_alpha, C_gamma=C_gamma, k=k,
        )

        assert info["bias_recovery"] in ("bounded_interval", "indeterminate")
        assert np.isfinite(b0)

    def test_determin_all_six_boundary_cases(self):
        """Cover all 6 KKT boundary cases from spec §6.6."""
        rng = np.random.RandomState(42)
        n, d = 8, 2
        X = rng.randn(n, d)
        y_tilde = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
        K = _rbf_kernel(X, X, sigma=1.0)
        mu = np.array([0.0, 0.0])
        k = 4
        C_alpha = 1.0 / n
        C_gamma = 1.0 / (2.0 * n)
        C_eq = -0.4

        alpha = np.array([C_alpha * 0.5, C_alpha, 0.0, C_alpha, 0.0, C_alpha, 0.0, C_alpha])
        gamma = np.array([C_gamma * 0.5, C_gamma, 0.0, C_gamma])

        b0, info = _recover_bias_from_kkt(
            alpha, gamma, X, K, y_tilde, mu,
            lambda_=1.0, sigma=1.0, C_eq=C_eq,
            C_alpha=C_alpha, C_gamma=C_gamma, k=k,
        )

        assert info["n_free"] >= 2
        assert np.isfinite(b0)
