"""Golden tests and mathematical-property tests for PNU loss functions.

Locks the formulas of the module-level helpers in ``pu_toolbox/losses/pnu.py``
(Sakai, du Plessis, Niu & Sugiyama, ICML 2017):

    R_PNU^eta(g) = (1-gamma)*R_PN + gamma*R_C-PU   (eta >= 0, gamma = eta)
                   (1-gamma)*R_PN + gamma*R_C-NU   (eta <  0, gamma = -eta)

with the squared-loss components R_PN, R_C-PU and R_C-NU.

Note: these pure helpers do NOT validate ``class_prior`` — the open-interval
check lives in ``pu_toolbox/estimators/risk/pnu.py`` (via
``check_scalar_in_range``); any float prior flows through the formula here.
See tests/unit/losses/test_nnpu_loss.py for authority-level conventions.
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.losses.pnu import (
    _compute_nu_risk_squared,
    _compute_pn_risk,
    _compute_pnu_risk,
    _compute_pu_risk_squared,
    _eta_to_gamma,
)

# Module-level marker: the whole module is fast unit tests.
pytestmark = pytest.mark.unit


# ═════════════════════════════════════════════════════════════════════
# eta -> gamma mapping
# ═════════════════════════════════════════════════════════════════════


class TestEtaToGamma:
    """MATH: eta -> (gamma, branch) with gamma = |eta|."""

    @pytest.mark.math
    def test_basic_eta_to_gamma_positive_eta(self):
        """eta >= 0 maps to (eta, "pu"): gamma equals eta itself."""
        # Hand mapping:
        #   eta =  0.5 -> gamma = 0.5, branch "pu"
        #   eta =  2.5 -> gamma = 2.5, branch "pu"
        #   eta =  0.0 -> gamma = 0.0, branch "pu"   (>= 0 branch)
        assert _eta_to_gamma(0.5) == (0.5, "pu")
        assert _eta_to_gamma(2.5) == (2.5, "pu")
        assert _eta_to_gamma(0.0) == (0.0, "pu")

    @pytest.mark.unit
    def test_edge_eta_boundary_zero_and_one(self):
        """Boundaries: gamma = |eta| at eta in {0, +-1}; sign picks branch."""
        # eta =  0.0 -> gamma 0.0, branch "pu"
        # eta = +1.0 -> gamma 1.0, branch "pu"
        # eta = -1.0 -> gamma 1.0, branch "nu"
        assert _eta_to_gamma(0.0) == (0.0, "pu")
        assert _eta_to_gamma(1.0) == (1.0, "pu")
        assert _eta_to_gamma(-1.0) == (1.0, "nu")
        # Sign symmetry: |+eta| == |-eta|; only the branch differs.
        g_plus, b_plus = _eta_to_gamma(1.0)
        g_minus, b_minus = _eta_to_gamma(-1.0)
        assert g_plus == g_minus
        assert b_plus != b_minus

    @pytest.mark.unit
    def test_deterministic_eta_to_gamma_known_mapping(self):
        """Known eta -> known (gamma, branch), identical across repeats."""
        for eta, expected in [(0.5, (0.5, "pu")), (-0.5, (0.5, "nu")), (0.0, (0.0, "pu"))]:
            assert _eta_to_gamma(eta) == expected
            assert _eta_to_gamma(eta) == _eta_to_gamma(eta)  # deterministic


# ═════════════════════════════════════════════════════════════════════
# Component risks (squared loss)
# ═════════════════════════════════════════════════════════════════════


class TestComponentRiskGolden:
    """MATH: hand-computed values for R_PN, R_C-PU and R_C-NU.

    Shared inputs: p = [2, 4] (mean(-p) = -3), n = [1, 3] (mean = 2),
    u = [0.5, 1.5] (mean = 1, mean(-u) = -1), class_prior = 0.25.
    """

    p = np.array([2.0, 4.0])
    n = np.array([1.0, 3.0])
    u = np.array([0.5, 1.5])
    class_prior = 0.25

    @pytest.mark.math
    def test_basic_pn_risk_hand_computed(self):
        """R_PN = theta_P * mean(-p) + theta_N * mean(n)."""
        # Hand computation:
        #   theta_N = 1 - 0.25 = 0.75
        #   mean(-p) = -(2 + 4)/2 = -3.0
        #   mean(n)  = (1 + 3)/2  =  2.0
        #   R_PN = 0.25 * (-3.0) + 0.75 * 2.0 = -0.75 + 1.5 = 0.75
        assert _compute_pn_risk(self.p, self.n, self.class_prior) == pytest.approx(0.75)

    @pytest.mark.math
    def test_basic_pu_risk_squared_hand_computed(self):
        """R_C-PU = theta_P * mean(-p) + mean(u)."""
        # Hand computation:
        #   theta_P = 0.25, mean(-p) = -3.0, mean(u) = 1.0
        #   R_C-PU = 0.25 * (-3.0) + 1.0 = -0.75 + 1.0 = 0.25
        assert _compute_pu_risk_squared(self.p, self.u, self.class_prior) == pytest.approx(0.25)

    @pytest.mark.math
    def test_basic_nu_risk_squared_hand_computed(self):
        """R_C-NU = theta_N * mean(n) + mean(-u)."""
        # Hand computation:
        #   theta_N = 0.75, mean(n) = 2.0, mean(-u) = -(0.5 + 1.5)/2 = -1.0
        #   R_C-NU = 0.75 * 2.0 + (-1.0) = 1.5 - 1.0 = 0.5
        assert _compute_nu_risk_squared(self.n, self.u, self.class_prior) == pytest.approx(0.5)

    @pytest.mark.unit
    def test_edge_single_sample_components(self):
        """n=1 arrays: the mean collapses to the single element."""
        # R_PN = 0.5 * mean(-[3]) + 0.5 * mean([5]) = -1.5 + 2.5 = 1.0
        assert _compute_pn_risk(np.array([3.0]), np.array([5.0]), 0.5) == pytest.approx(1.0)
        # R_C-PU = 0.5 * (-3.0) + mean([1.0]) = -1.5 + 1.0 = -0.5
        assert _compute_pu_risk_squared(np.array([3.0]), np.array([1.0]), 0.5) == pytest.approx(
            -0.5
        )
        # R_C-NU = 0.5 * 5.0 + mean(-[1.0]) = 2.5 - 1.0 = 1.5
        assert _compute_nu_risk_squared(np.array([5.0]), np.array([1.0]), 0.5) == pytest.approx(1.5)


# ═════════════════════════════════════════════════════════════════════
# Combined PNU risk
# ═════════════════════════════════════════════════════════════════════


class TestPnuRiskGolden:
    """MATH: combined R_PNU^eta on both branches.

    Shared inputs as in TestComponentRiskGolden:
    R_PN = 0.75, R_C-PU = 0.25, R_C-NU = 0.5.
    """

    p = np.array([2.0, 4.0])
    n = np.array([1.0, 3.0])
    u = np.array([0.5, 1.5])
    class_prior = 0.25

    @pytest.mark.math
    def test_basic_pnu_risk_pu_branch_hand_computed(self):
        """eta=+0.5 -> gamma=0.5, branch "pu": (1-0.5)*R_PN + 0.5*R_C-PU."""
        risk = _compute_pnu_risk(self.p, self.n, self.u, self.class_prior, 0.5)
        # Hand computation:
        #   (1 - 0.5) * 0.75 + 0.5 * 0.25 = 0.375 + 0.125 = 0.5
        assert risk == pytest.approx(0.5)

    @pytest.mark.math
    def test_basic_pnu_risk_nu_branch_hand_computed(self):
        """eta=-0.5 -> gamma=0.5, branch "nu": (1-0.5)*R_PN + 0.5*R_C-NU."""
        risk = _compute_pnu_risk(self.p, self.n, self.u, self.class_prior, -0.5)
        # Hand computation:
        #   (1 - 0.5) * 0.75 + 0.5 * 0.5 = 0.375 + 0.25 = 0.625
        assert risk == pytest.approx(0.625)

    @pytest.mark.unit
    def test_edge_pnu_risk_eta_zero_degenerates_to_pn(self):
        """eta=0 -> gamma=0: the combination reduces to R_PN exactly.

        Extreme unlabeled scores (100, 200) prove the unlabeled component
        is fully ignored — R_C-PU of the "pu" branch never enters.
        """
        u_extreme = np.array([100.0, 200.0])
        risk = _compute_pnu_risk(self.p, self.n, u_extreme, self.class_prior, 0.0)
        # (1 - 0.0) * 0.75 + 0.0 * (anything) = 0.75
        assert risk == pytest.approx(0.75)

    @pytest.mark.unit
    def test_edge_pnu_risk_eta_one_reduces_to_pu_component(self):
        """eta=1 -> gamma=1: the combination equals R_C-PU exactly."""
        risk = _compute_pnu_risk(self.p, self.n, self.u, self.class_prior, 1.0)
        # (1 - 1.0) * 0.75 + 1.0 * 0.25 = 0.25
        assert risk == pytest.approx(0.25)


# ═════════════════════════════════════════════════════════════════════
# Parameter handling (class_prior)
# ═════════════════════════════════════════════════════════════════════


class TestPriorParameterHandling:
    """Param: any float class_prior flows through — no validation here.

    The module-level helpers are pure formula evaluators; range validation
    (0.0 < class_prior < 1.0, message "class_prior must be in (0.0, 1.0); ...")
    is enforced by the consuming estimator ``pu_toolbox/estimators/risk/pnu.py``
    via ``check_scalar_in_range`` — not by these helpers.
    """

    p = np.array([2.0, 4.0])
    n = np.array([1.0, 3.0])
    u = np.array([0.5, 1.5])

    @pytest.mark.unit
    def test_validation_out_of_range_class_prior_flows_through(self):
        """class_prior = 1.5 (> 1) does not raise; theta_N becomes -0.5."""
        # R_PN = 1.5 * mean(-p) + (1 - 1.5) * mean(n)
        #      = 1.5 * (-3.0) + (-0.5) * 2.0 = -4.5 - 1.0 = -5.5
        assert _compute_pn_risk(self.p, self.n, 1.5) == pytest.approx(-5.5)
        # R_C-PU = 1.5 * (-3.0) + 1.0 = -4.5 + 1.0 = -3.5
        assert _compute_pu_risk_squared(self.p, self.u, 1.5) == pytest.approx(-3.5)
        # R_C-NU = (1 - 1.5) * 2.0 + (-1.0) = -1.0 - 1.0 = -2.0
        assert _compute_nu_risk_squared(self.n, self.u, 1.5) == pytest.approx(-2.0)

    @pytest.mark.unit
    def test_validation_boundary_class_prior_zero_and_one(self):
        """class_prior = 0 -> theta_N = 1; class_prior = 1 -> theta_N = 0."""
        # R_PN(pi=0) = 0 * mean(-p) + 1 * mean(n) = 2.0
        assert _compute_pn_risk(self.p, self.n, 0.0) == pytest.approx(2.0)
        # R_PN(pi=1) = 1 * mean(-p) + 0 * mean(n) = -3.0
        assert _compute_pn_risk(self.p, self.n, 1.0) == pytest.approx(-3.0)
        # R_C-PU(pi=0) = 0 * mean(-p) + mean(u) = 1.0 (positive term vanishes)
        assert _compute_pu_risk_squared(self.p, self.u, 0.0) == pytest.approx(1.0)
        # R_C-NU(pi=1) = 0 * mean(n) + mean(-u) = -1.0 (negative term vanishes)
        assert _compute_nu_risk_squared(self.n, self.u, 1.0) == pytest.approx(-1.0)


# ═════════════════════════════════════════════════════════════════════
# Empty inputs
# ═════════════════════════════════════════════════════════════════════


class TestEmptyInputs:
    """Edge: empty score arrays produce NaN, not a raise (no guard exists)."""

    @pytest.mark.unit
    def test_edge_empty_score_arrays_return_nan(self):
        """np.mean of an empty slice -> NaN with a RuntimeWarning."""
        with pytest.warns(RuntimeWarning):
            risk = _compute_pn_risk(np.array([]), np.array([]), 0.5)
        assert np.isnan(risk)

        with pytest.warns(RuntimeWarning):
            risk = _compute_pnu_risk(np.array([]), np.array([]), np.array([]), 0.5, 0.3)
        assert np.isnan(risk)


# ═════════════════════════════════════════════════════════════════════
# Determinism
# ═════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Determ: pure functions — the same input always gives the same output."""

    p = np.array([2.0, 4.0])
    n = np.array([1.0, 3.0])
    u = np.array([0.5, 1.5])

    @pytest.mark.unit
    def test_deterministic_repeated_calls_identical_output(self):
        """All five helpers return bit-identical results on repeat calls."""
        first = (
            _compute_pn_risk(self.p, self.n, 0.25),
            _compute_pu_risk_squared(self.p, self.u, 0.25),
            _compute_nu_risk_squared(self.n, self.u, 0.25),
            _compute_pnu_risk(self.p, self.n, self.u, 0.25, 0.5),
            _compute_pnu_risk(self.p, self.n, self.u, 0.25, -0.5),
            _eta_to_gamma(-0.5),
        )
        second = (
            _compute_pn_risk(self.p, self.n, 0.25),
            _compute_pu_risk_squared(self.p, self.u, 0.25),
            _compute_nu_risk_squared(self.n, self.u, 0.25),
            _compute_pnu_risk(self.p, self.n, self.u, 0.25, 0.5),
            _compute_pnu_risk(self.p, self.n, self.u, 0.25, -0.5),
            _eta_to_gamma(-0.5),
        )
        assert first == second
