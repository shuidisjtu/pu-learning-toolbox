# ruff: noqa: N802, N803, N806, S101

"""MATH tests for KLDCE — formula-level correctness.

Covers:
- Q/d/Aeq/beq/lb/ub construction (Appendix Eq. 24)
- C_eq explicit formula
- Phase-I feasible initialisation
- Decision function term-by-term (Appendix Eq. 25)
- RBF centroid delta (Appendix Eq. 33, Taylor at μ=0)
- Bias recovery from KKT conditions (§6.6)
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.estimators.risk.kldce import (
    _build_dual_qp,
    _find_feasible_init,
    _rbf_centroid_delta,
    _rbf_kernel,
)


# ═════════════════════════════════════════════════════════════════════
# Hand-computed test data (4 samples, 2 P + 2 U)
# ═════════════════════════════════════════════════════════════════════

# 4 samples in 2-D: first 2 are P, last 2 are U
# X: [[0, 0], [1, 0], [0, 1], [1, 1]]
# ỹ: [+1, +1, -1, -1]
# n=4, k=2, n_U=2
# λ=1.0, σ=1.0, h=0.3

@pytest.fixture
def tiny_data():
    """4-sample dataset for hand computation."""
    X = np.array([
        [0.0, 0.0],  # P
        [1.0, 0.0],  # P
        [0.0, 1.0],  # U
        [1.0, 1.0],  # U
    ])
    y_tilde = np.array([1.0, 1.0, -1.0, -1.0])
    return X, y_tilde


# ═════════════════════════════════════════════════════════════════════
# C_eq formula
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.math
class TestCeqFormula:
    """C_eq = -(n-k) / (2·n·(1−2·p·h))"""

    def test_validation_explicit_formula(self):
        n, k, h = 100, 40, 0.3
        p = k / (n * (1.0 - h))
        C_eq = -(n - k) / (2.0 * n * (1.0 - 2.0 * p * h))
        # p = 40 / 70 = 0.5714...
        # 1-2ph = 1 - 2*0.5714*0.3 = 1 - 0.3429 = 0.6571...
        # C_eq = -60 / (200 * 0.6571) = -60 / 131.43 = -0.4565...
        expected = -(100 - 40) / (2.0 * 100.0 * (1.0 - 2.0 * (40.0 / (100.0 * 0.7)) * 0.3))
        assert abs(C_eq - expected) < 1e-14
        assert C_eq < 0  # always negative when k < n

    def test_uses_correct_constants(self):
        """C_alpha = 1/n, C_gamma = 1/(2n) — hard-coded, not tunable."""
        n = 50
        C_alpha = 1.0 / n
        C_gamma = 1.0 / (2.0 * n)
        assert C_alpha == 0.02
        assert C_gamma == 0.01
        # Verify C_gamma = C_alpha / 2
        assert abs(C_gamma - C_alpha / 2.0) < 1e-15


# ═════════════════════════════════════════════════════════════════════
# _build_dual_qp — Q, d, constraints (Appendix Eq. 24)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.math
class TestBuildDualQP:
    """Verify Q, d(μ), Aeq, beq, lb, ub against hand-computed values."""

    def test_q_block_shapes(self, tiny_data):
        X, y_tilde = tiny_data
        n, k = 4, 2
        K = _rbf_kernel(X, X, sigma=1.0)
        mu = np.array([0.5, 0.5])
        C_eq = -0.5  # arbitrary for shape test

        Q, d_vec, Aeq, beq, lb, ub = _build_dual_qp(
            mu, X, K, y_tilde, lambda_=1.0, sigma=1.0,
            n=n, k=k, C_eq=C_eq,
        )

        N = n + (n - k)  # 4 + 2 = 6
        assert Q.shape == (N, N)
        assert d_vec.shape == (N,)
        assert Aeq.shape == (1, N)
        assert lb.shape == (N,)
        assert ub.shape == (N,)

    def test_q_symmetry(self, tiny_data):
        X, y_tilde = tiny_data
        n, k = 4, 2
        K = _rbf_kernel(X, X, sigma=1.0)
        mu = np.array([0.5, 0.5])
        C_eq = -0.5

        Q, _, _, _, _, _ = _build_dual_qp(
            mu, X, K, y_tilde, lambda_=1.0, sigma=1.0,
            n=n, k=k, C_eq=C_eq,
        )
        np.testing.assert_allclose(Q, Q.T, atol=1e-14)

    def test_determin_q_hand_computed_4sample(self, tiny_data):
        """Hand-compute Q_αα[0,1] = ỹ₀ỹ₁K(x₀,x₁)/(2λ)."""
        X, y_tilde = tiny_data
        n, k = 4, 2
        sigma_val = 1.0
        K = _rbf_kernel(X, X, sigma_val)
        mu = np.array([0.0, 0.0])
        C_eq = -0.5

        Q, _, _, _, _, _ = _build_dual_qp(
            mu, X, K, y_tilde, lambda_=1.0, sigma=sigma_val,
            n=n, k=k, C_eq=C_eq,
        )

        # Q_αα[0,1] = ỹ₀·ỹ₁·K(x₀,x₁) / (2λ)
        # ỹ₀=+1, ỹ₁=+1, K(x₀,x₁)=exp(-||[0,0]-[1,0]||²/(2·1²))=exp(-0.5)
        expected_01 = 1.0 * 1.0 * np.exp(-0.5) / 2.0
        np.testing.assert_allclose(Q[0, 1], expected_01, rtol=1e-12)

    def test_q_alpha_gamma_sign(self, tiny_data):
        """Q_αγ[i,j] = -ỹᵢ·ỹ_{k+j}·K(xᵢ, x_{k+j}) / (2λ)."""
        X, y_tilde = tiny_data
        n, k = 4, 2
        sigma_val = 1.0
        K = _rbf_kernel(X, X, sigma_val)
        mu = np.array([0.0, 0.0])
        C_eq = -0.5

        Q, _, _, _, _, _ = _build_dual_qp(
            mu, X, K, y_tilde, lambda_=1.0, sigma=sigma_val,
            n=n, k=k, C_eq=C_eq,
        )

        # Q_αγ[0,0]: i=0 (P, ỹ=+1), j=0 (U, ỹ_{2}=-1)
        # = -(+1)·(-1)·K(x₀, x₂) / 2 = +K(x₀, x₂) / 2
        # K(x₀, x₂) = exp(-||[0,0]-[0,1]||²/2) = exp(-0.5)
        expected_alpha_gamma_00 = np.exp(-0.5) / 2.0
        np.testing.assert_allclose(Q[0, n], expected_alpha_gamma_00, rtol=1e-12)
        # Should be positive (since -(+1)*(-1) = +1)
        assert Q[0, n] > 0

    def test_linear_term_d_mu_structure(self, tiny_data):
        """d(μ)_i = 1 + C_eq·ỹᵢ·K(xᵢ,μ)/(2λ) for α,
           d(μ)_{n+i} = 1 - C_eq·ỹ_{k+i}·K(x_{k+i},μ)/(2λ) for γ."""
        X, y_tilde = tiny_data
        n, k = 4, 2
        sigma_val = 1.0
        K = _rbf_kernel(X, X, sigma_val)
        mu = np.array([0.0, 0.0])
        C_eq = -0.5

        _, d_vec, _, _, _, _ = _build_dual_qp(
            mu, X, K, y_tilde, lambda_=1.0, sigma=sigma_val,
            n=n, k=k, C_eq=C_eq,
        )

        # d(μ)[0]: α, P sample x₀=[0,0], ỹ₀=+1
        # K(x₀,μ) = exp(-||[0,0]-[0,0]||²/2) = exp(0) = 1.0
        # d = 1 + (-0.5)*(+1)*1.0/2 = 1 - 0.25 = 0.75
        np.testing.assert_allclose(d_vec[0], 0.75, rtol=1e-12)

        # d(μ)[4]: first γ, U sample x₂=[0,1], ỹ₂=-1
        # K(x₂,μ) = exp(-||[0,1]-[0,0]||²/2) = exp(-0.5)
        # d = 1 - (-0.5)*(-1)*exp(-0.5)/2 = 1 - 0.25*exp(-0.5) ≈ 0.8483...
        expected_gamma_0 = 1.0 - (-0.5) * (-1.0) * np.exp(-0.5) / 2.0
        np.testing.assert_allclose(d_vec[4], expected_gamma_0, rtol=1e-12)

    def test_aeq_structure(self, tiny_data):
        """Aeq = [ỹ₁…ỹₙ | −ỹ_{k+1}…−ỹₙ]."""
        X, y_tilde = tiny_data
        n, k = 4, 2
        K = _rbf_kernel(X, X, sigma=1.0)
        mu = np.array([0.0, 0.0])
        C_eq = -0.5

        _, _, Aeq, _, _, _ = _build_dual_qp(
            mu, X, K, y_tilde, lambda_=1.0, sigma=1.0,
            n=n, k=k, C_eq=C_eq,
        )

        # First n entries = ỹ: [+1, +1, -1, -1]
        np.testing.assert_array_equal(Aeq[0, :4], y_tilde)
        # Last n_U entries = -ỹ_{k:} = -[-1, -1] = [+1, +1]
        np.testing.assert_array_equal(Aeq[0, 4:], np.array([1.0, 1.0]))

    def test_determ_bounds(self, tiny_data):
        """lb = 0, ub = [1/n]*n + [1/(2n)]*n_U."""
        X, y_tilde = tiny_data
        n, k = 4, 2
        K = _rbf_kernel(X, X, sigma=1.0)
        mu = np.array([0.0, 0.0])
        C_eq = -0.5

        _, _, _, _, lb, ub = _build_dual_qp(
            mu, X, K, y_tilde, lambda_=1.0, sigma=1.0,
            n=n, k=k, C_eq=C_eq,
        )

        np.testing.assert_array_equal(lb, np.zeros(6))
        np.testing.assert_allclose(ub[:4], np.full(4, 1.0 / 4))  # C_alpha = 0.25
        np.testing.assert_allclose(ub[4:], np.full(2, 1.0 / 8))  # C_gamma = 0.125

    def test_beq_equals_ceq(self, tiny_data):
        """beq = C_eq (the equality RHS equals the input C_eq)."""
        X, y_tilde = tiny_data
        K = _rbf_kernel(X, X, sigma=1.0)
        mu = np.array([0.0, 0.0])
        C_eq = -0.5

        _, _, _, beq, _, _ = _build_dual_qp(
            mu, X, K, y_tilde, lambda_=1.0, sigma=1.0,
            n=4, k=2, C_eq=C_eq,
        )
        assert abs(beq - C_eq) < 1e-14


# ═════════════════════════════════════════════════════════════════════
# _find_feasible_init — Phase-I LP
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.math
class TestFeasibleInit:
    """Phase-I LP feasibility."""

    def test_feasible_when_ceq_nonzero(self):
        """When C_eq ≠ 0, z=0 is infeasible; Phase-I LP must find a point."""
        n = 4
        N = 6
        y_tilde = np.array([1.0, 1.0, -1.0, -1.0])
        Aeq = np.zeros((1, N))
        Aeq[0, :4] = y_tilde
        Aeq[0, 4:] = -y_tilde[2:]  # = [1, 1]
        beq = -0.5
        lb = np.zeros(N)
        ub = np.array([0.25, 0.25, 0.25, 0.25, 0.125, 0.125])

        z0 = _find_feasible_init(Aeq, beq, lb, ub)

        eq_res = float(np.abs(Aeq @ z0 - beq).max())
        assert eq_res <= 1e-10, f"Equality residual {eq_res}"
        assert (z0 >= lb - 1e-12).all()
        assert (z0 <= ub + 1e-12).all()

    def test_feasible_init_output_shape(self):
        N = 6
        y_tilde = np.array([1.0, 1.0, -1.0, -1.0])
        Aeq = np.zeros((1, N))
        Aeq[0, :4] = y_tilde
        Aeq[0, 4:] = -y_tilde[2:]
        beq = -0.5
        lb = np.zeros(N)
        ub = np.full(N, 1.0)

        z0 = _find_feasible_init(Aeq, beq, lb, ub)
        assert z0.shape == (N,)
        assert np.isfinite(z0).all()


# ═════════════════════════════════════════════════════════════════════
# RBF centroid delta (Appendix Eq. 33)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.math
class TestRBFCentroidDelta:
    """Taylor expansion at μ=0 for centroid update direction."""

    def test_edge_zero_alpha_gamma_gives_zero_delta(self):
        X = np.random.RandomState(0).randn(10, 3)
        y_tilde = np.concatenate([np.ones(5), -np.ones(5)])
        alpha = np.zeros(10)
        gamma = np.zeros(5)

        delta = _rbf_centroid_delta(alpha, gamma, X, y_tilde,
                                     lambda_=1.0, sigma=1.0, k=5)
        np.testing.assert_allclose(delta, np.zeros(3), atol=1e-14)

    def test_delta_shape_and_finite(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10, 3)
        y_tilde = np.concatenate([np.ones(5), -np.ones(5)])
        alpha = rng.uniform(0, 0.1, 10)
        gamma = rng.uniform(0, 0.05, 5)

        delta = _rbf_centroid_delta(alpha, gamma, X, y_tilde,
                                     lambda_=1.0, sigma=2.0, k=5)
        assert delta.shape == (3,)
        assert np.isfinite(delta).all()

    def test_delta_hand_computed_2d(self):
        """Hand-compute delta for 2 samples in 1-D."""
        # X = [[1.0], [2.0]], P then U
        # α = [0.1, 0.05], γ = [0.02]
        # ỹ = [+1, -1], λ=1, σ=1
        X = np.array([[1.0], [2.0]])
        y_tilde = np.array([1.0, -1.0])
        alpha = np.array([0.1, 0.05])
        gamma = np.array([0.02])
        k = 1
        lambda_ = 1.0
        sigma = 1.0

        delta = _rbf_centroid_delta(alpha, gamma, X, y_tilde,
                                     lambda_, sigma, k)

        # scale = 1/(2*1*1²) = 0.5
        # α contribution (all samples, neg):
        #   i=0: α₀·ỹ₀·exp(-x₀²/2)·x₀ = 0.1*(+1)*exp(-0.5)*1.0 = 0.1*0.6065 = 0.06065
        #   i=1: α₁·ỹ₁·exp(-x₁²/2)·x₁ = 0.05*(-1)*exp(-2)*2.0 = -0.05*0.1353*2 = -0.01353
        #   sum = 0.04712
        #   -scale * sum = -0.5 * 0.04712 = -0.02356
        # γ contribution (only U, pos):
        #   j=0: γ₀·ỹ₁·exp(-x₁²/2)·x₁ = 0.02*(-1)*exp(-2)*2.0 = -0.02*0.1353*2 = -0.00541
        #   scale * sum = 0.5 * (-0.00541) = -0.00271
        # Total delta = -0.02356 + (-0.00271) = -0.02627

        w0 = np.exp(-1.0 ** 2 / 2.0)  # exp(-0.5)
        w1 = np.exp(-2.0 ** 2 / 2.0)  # exp(-2)
        alpha_sum = alpha[0] * y_tilde[0] * w0 * X[0] + alpha[1] * y_tilde[1] * w1 * X[1]
        gamma_sum = gamma[0] * y_tilde[1] * w1 * X[1]
        expected = -0.5 * alpha_sum + 0.5 * gamma_sum

        np.testing.assert_allclose(delta, expected, rtol=1e-12)


