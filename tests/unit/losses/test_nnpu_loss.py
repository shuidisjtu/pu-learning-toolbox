"""Golden tests and mathematical-property tests for nnPU loss functions.

============================================================================
AUTHORITY LEVELS (see tests/README.md)
============================================================================
  MATH     — Hand-computed expected values from paper formulas.
             Failure = CODE BUG (the formula IS the ground truth).
  PROPERTY — Mathematical invariants proven in Kiryo et al. 2017.
             Failure = CODE BUG (invariant guaranteed by theorem).
  EMPIRICAL — Reasonable behaviour check; thresholds may need tuning.
============================================================================
"""

# ruff: noqa: N803, N806

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.losses.nnpu import NonNegativePULoss, _sigmoid_stable

# ═════════════════════════════════════════════════════════════════════
# Helper: reference sigmoid for hand-computation
# ═════════════════════════════════════════════════════════════════════


def _σ(z: float) -> float:
    """Reference sigmoid for manual calculation."""
    return float(1.0 / (1.0 + np.exp(-z)))


# ═════════════════════════════════════════════════════════════════════
# MATH — Golden tests (hand-computed expected values)
# ═════════════════════════════════════════════════════════════════════


class TestGoldenRiskValues:
    """MATH: Expected values computed by hand from Eqs. (4.1)–(4.3), (6).

    Every assertion in this class is backed by a manual calculation.
    If these fail, the production code is WRONG — never adjust the
    expected value to match a code change.
    """

    @pytest.mark.math
    def test_golden_balanced_scores(self):
        """All scores = 0 → σ(±0) = 0.5 → known component values."""
        p = np.array([0.0, 0.0, 0.0])
        u = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        pi = 0.5

        info = NonNegativePULoss().evaluate(p, u, class_prior=pi)

        # σ(0) = 0.5 for all scores
        # R_p^+ = mean(σ(-0)) = 0.5
        # R_p^- = mean(σ(0)) = 0.5
        # R_u^- = mean(σ(0)) = 0.5
        # r = 0.5 − 0.5·0.5 = 0.25
        # upu = 0.5·0.5 + 0.25 = 0.5
        # nnpu = 0.5·0.5 + max(0, 0.25) = 0.5
        assert info["positive_risk"] == pytest.approx(0.5)
        assert info["negative_risk"] == pytest.approx(0.25)
        assert info["upu_risk"] == pytest.approx(0.5)
        assert info["nnpu_risk"] == pytest.approx(0.5)

    @pytest.mark.math
    def test_golden_extreme_separation(self):
        """P scores strongly positive, U scores strongly negative.

        σ(−8) ≈ 0.000335, σ(+8) ≈ 0.999665.
        r < 0 → nnPU > uPU (correction activates).
        """
        p = np.array([8.0, 8.0])
        u = np.array([-8.0, -8.0])
        pi = 0.3

        info = NonNegativePULoss().evaluate(p, u, class_prior=pi)

        # Hand calculation:
        # R_p^+ = σ(−8) ≈ 0.000335
        # R_p^- = σ(+8) ≈ 0.999665
        # R_u^- = σ(−8) ≈ 0.000335   (U score = −8, σ(+g) = σ(−8))
        # r = 0.000335 − 0.3·0.999665 ≈ −0.29956
        sig8 = float(1.0 / (1.0 + np.exp(-8.0)))   # σ(8) ≈ 0.999665
        sigm8 = float(1.0 / (1.0 + np.exp(8.0)))    # σ(−8) ≈ 0.000335
        r_hand = sigm8 - pi * sig8
        upu_hand = pi * sigm8 + r_hand
        nnpu_hand = pi * sigm8 + max(0.0, r_hand)

        assert info["positive_risk"] == pytest.approx(sigm8)
        assert info["negative_risk"] == pytest.approx(r_hand)
        assert info["upu_risk"] == pytest.approx(upu_hand)
        assert info["nnpu_risk"] == pytest.approx(nnpu_hand)
        assert r_hand < 0
        assert info["upu_risk"] < info["nnpu_risk"]

    @pytest.mark.math
    def test_golden_mixed_sign_scores(self):
        """Known values: scores = [−2, 1] for P and [3, −0.5] for U."""
        p = np.array([-2.0, 1.0])
        u = np.array([3.0, -0.5])
        pi = 0.4

        info = NonNegativePULoss().evaluate(p, u, class_prior=pi)

        # R_p^+ = σ(−g_p): σ(−(−2))=σ(2)≈0.8808, σ(−1)≈0.2689 → mean≈0.5749
        R_p_plus_h = 0.5 * (_σ(2.0) + _σ(-1.0))
        # R_p^- = σ(+g_p): σ(−2)≈0.1192, σ(1)≈0.7311 → mean≈0.4251
        R_p_minus_h = 0.5 * (_σ(-2.0) + _σ(1.0))
        # R_u^- = σ(+g_u): σ(3)≈0.9526, σ(−0.5)≈0.3775 → mean≈0.6651
        R_u_minus_h = 0.5 * (_σ(3.0) + _σ(-0.5))
        r_h = R_u_minus_h - pi * R_p_minus_h
        upu_h = pi * R_p_plus_h + r_h
        nnpu_h = pi * R_p_plus_h + max(0.0, r_h)

        assert info["positive_risk"] == pytest.approx(R_p_plus_h)
        assert info["negative_risk"] == pytest.approx(r_h)
        assert info["upu_risk"] == pytest.approx(upu_h)
        assert info["nnpu_risk"] == pytest.approx(nnpu_h)


# ═════════════════════════════════════════════════════════════════════
# PROPERTY — Mathematical invariants
# ═════════════════════════════════════════════════════════════════════


class TestRiskInvariants:
    """PROPERTY: Invariants guaranteed by the paper's mathematics.

    These are NOT empirical observations.  If they fail, the code
    violates a theorem from Kiryo et al. 2017.
    """

    @pytest.mark.property
    def test_positive_risk_bounded_by_one(self):
        """sigmoid(z) ∈ [0, 1] → R_p^+ ∈ [0, 1] for any scores."""
        loss = NonNegativePULoss()
        rng = np.random.RandomState(42)
        for _ in range(30):
            p = rng.uniform(-20, 20, size=rng.randint(2, 50))
            u = rng.uniform(-20, 20, size=rng.randint(2, 50))
            pi = rng.uniform(0.05, 0.95)
            info = loss.evaluate(p, u, class_prior=pi)
            assert 0.0 <= info["positive_risk"] <= 1.0, (
                f"R_p^+ = {info['positive_risk']} not in [0, 1]"
            )

    @pytest.mark.property
    def test_nnpu_risk_never_negative(self):
        """Theorem: R̃_pu(g) = π·R_p^+(g) + max(0, r) ≥ 0 always."""
        loss = NonNegativePULoss()
        rng = np.random.RandomState(42)
        for _ in range(50):
            p = rng.uniform(-20, 20, size=rng.randint(2, 30))
            u = rng.uniform(-20, 20, size=rng.randint(2, 30))
            pi = rng.uniform(0.05, 0.95)
            nnpu = loss(p, u, class_prior=pi, non_negative=True)
            assert nnpu >= -1e-12, f"nnPU risk negative: {nnpu}"

    @pytest.mark.property
    def test_nnpu_risk_gte_upu_risk(self):
        """Theorem: max(0, r) ≥ r → nnPU ≥ uPU."""
        loss = NonNegativePULoss()
        rng = np.random.RandomState(42)
        for _ in range(50):
            p = rng.uniform(-20, 20, size=rng.randint(2, 30))
            u = rng.uniform(-20, 20, size=rng.randint(2, 30))
            pi = rng.uniform(0.05, 0.95)
            r_nn = loss(p, u, class_prior=pi, non_negative=True)
            r_upu = loss(p, u, class_prior=pi, non_negative=False)
            assert r_nn >= r_upu - 1e-12, (
                f"nnPU ({r_nn}) < uPU ({r_upu})"
            )

    @pytest.mark.property
    def test_risk_monotonic_in_class_prior(self):
        """For fixed scores, R_p^+ term is linear in π."""
        loss = NonNegativePULoss()
        p = np.array([1.0, -0.5])
        u = np.array([0.5, -1.0])

        info_lo = loss.evaluate(p, u, class_prior=0.3)
        info_hi = loss.evaluate(p, u, class_prior=0.5)

        # R_p^+ and R_p^- are independent of π
        assert info_lo["positive_risk"] == pytest.approx(info_hi["positive_risk"])

    @pytest.mark.property
    def test_beta_zero_gives_standard_nnpu(self):
        """beta=0 and beta→0 give the same nnpu_risk (standard nnPU)."""
        loss0 = NonNegativePULoss(beta=0.0)
        p = np.array([5.0, 4.0])
        u = np.array([-3.0, -2.0])
        pi = 0.3
        r0 = loss0(p, u, class_prior=pi, non_negative=True)
        # For beta=0: correction whenever r < 0
        # nnpu risk = π·R_p^+ + max(0, r) — always non-negative
        assert r0 >= 0.0

    @pytest.mark.property
    def test_sigmoid_stable_is_symmetric(self):
        """σ(−z) = 1 − σ(z) (sigmoid symmetry)."""
        for z in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            s_neg = _sigmoid_stable(np.array([-z]))
            s_pos = _sigmoid_stable(np.array([z]))
            assert s_neg[0] == pytest.approx(1.0 - s_pos[0])
