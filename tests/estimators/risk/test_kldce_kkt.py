# ruff: noqa: N802, N803, N806, S101

"""KKT residual utility and ACS convergence behaviour for KLDCE."""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.estimators.risk.kldce import KLDCEClassifier, _true_kkt_residual
from tests.estimators.risk.test_kldce_property import _make_censoring_pu_data


def _small_qp():
    """Q=I2, d=[2,3], z1+z2=3, 0<=z<=5: analytic optimum z*=[1,2] (inner)."""
    Q = np.eye(2)
    d_vec = np.array([2.0, 3.0])
    Aeq = np.array([[1.0, 1.0]])
    lb = np.zeros(2)
    ub = np.full(2, 5.0)
    return Q, d_vec, Aeq, 3.0, lb, ub


@pytest.mark.unit
class TestKktResidual:
    """True KKT residual: exact optimum ~0, perturbations detected."""

    def test_basic_inner_point_kkt_zero(self):
        Q, d_vec, Aeq, beq, lb, ub = _small_qp()
        z_star = np.array([1.0, 2.0])
        kkt, nu, n_free, status = _true_kkt_residual(z_star, Q, d_vec, Aeq, beq, lb, ub)
        assert status == "ok"
        assert n_free == 2
        assert nu == pytest.approx(1.0)
        assert kkt == pytest.approx(0.0, abs=1e-9)

    def test_param_perturbed_z_detects_violation(self):
        Q, d_vec, Aeq, beq, lb, ub = _small_qp()
        z_bad = np.array([0.9, 2.1])  # feasible (sum=3) but not stationary
        kkt, _, _, status = _true_kkt_residual(z_bad, Q, d_vec, Aeq, beq, lb, ub)
        assert status == "ok"
        assert kkt > 0.05

    def test_edge_lower_bound_active_ok(self):
        # d=[2,3], sum=1.0 -> z*=[0.0,1.0] (z1 at lower bound)
        Q = np.eye(2)
        d_vec = np.array([2.0, 3.0])
        Aeq = np.array([[1.0, 1.0]])
        beq = 1.0
        lb = np.zeros(2)
        ub = np.full(2, 5.0)
        z_star = np.array([0.0, 1.0])
        kkt, nu, n_free, status = _true_kkt_residual(z_star, Q, d_vec, Aeq, beq, lb, ub)
        assert status == "ok"
        assert n_free == 1  # z1 at lb is not free
        assert kkt < 1e-9

    def test_edge_upper_bound_active_ok(self):
        # d=[9,2.2], sum=8.0 -> unconstrained proj z*=(7.4,0.6), clamped to
        # z=[5,3] (z1 at upper bound; z2 inner, nu=-0.8)
        Q = np.eye(2)
        d_vec = np.array([9.0, 2.2])
        Aeq = np.array([[1.0, 1.0]])
        beq = 8.0
        lb = np.zeros(2)
        ub = np.full(2, 5.0)
        z_star = np.array([5.0, 3.0])
        kkt, nu, n_free, status = _true_kkt_residual(z_star, Q, d_vec, Aeq, beq, lb, ub)
        assert status == "ok"
        assert n_free == 1
        assert nu == pytest.approx(-0.8)
        assert kkt < 1e-9

    def test_determ_no_free_returns_indeterminate(self):
        Q = np.eye(2)
        d_vec = np.array([10.0, -10.0])
        Aeq = np.array([[1.0, 1.0]])
        lb = np.zeros(2)
        ub = np.full(2, 1e-12)
        z_edge = np.array([0.0, 0.0])
        _, _, n_free, status = _true_kkt_residual(z_edge, Q, d_vec, Aeq, beq=0.0, lb=lb, ub=ub)
        assert status == "indeterminate"


@pytest.mark.unit
class TestACSConvergence:
    """ACS converges quickly after the criterion fix; strict KKT sanity."""

    @pytest.mark.parametrize("seed", [0, 1])
    def test_basic_acs_converges_within_budget(self, seed, rng):
        rng2 = np.random.RandomState(seed)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=10, n_neg=20, h=0.3, d=3)
        clf = KLDCEClassifier(
            flip_probability=0.3,
            sigma=1.0,
            reg_strength=0.1,
            max_acs_iter=50,
            max_dual_variables=2000,
            tol=1e-4,
            random_state=42,
        )
        clf.fit(X, y_pu)
        assert clf.converged_ is True
        assert clf.n_acs_iter_ <= 10

    def test_param_final_kkt_residual_small(self, rng):
        rng2 = np.random.RandomState(17)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=10, n_neg=20, h=0.3, d=3)
        clf = KLDCEClassifier(
            flip_probability=0.3,
            sigma=1.0,
            max_acs_iter=50,
            max_dual_variables=2000,
            tol=1e-4,
            random_state=42,
        )
        clf.fit(X, y_pu)
        last = clf.acs_history_[-1]
        assert 0.0 <= last["kkt_residual"] < 1e-2, last["kkt_residual"]
        assert "gradient_norm" in last

    def test_edge_centroid_violation_negligible(self, rng):
        rng2 = np.random.RandomState(18)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=10, n_neg=20, h=0.3, d=3)
        clf = KLDCEClassifier(
            flip_probability=0.3,
            sigma=1.0,
            centroid_radius=1.0,
            max_acs_iter=50,
            max_dual_variables=2000,
            tol=1e-4,
            random_state=42,
        )
        clf.fit(X, y_pu)
        for entry in clf.acs_history_:
            # Degenerate steps fix mu = m_hat (strictly inside the
            # ellipsoid): violation is exactly -centroid_radius by
            # construction (constraint value 0 minus radius).
            if entry.get("degenerate_centroid_step", False):
                assert entry["centroid_violation"] == pytest.approx(-clf.centroid_radius), entry
                continue
            assert abs(entry["centroid_violation"]) <= 0.05 * entry.get(
                "centroid_constraint_residual", 1.0
            ), entry

    def test_determ_eq_residual_stays_small(self, rng):
        rng2 = np.random.RandomState(19)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=10, n_neg=20, h=0.3, d=3)
        clf = KLDCEClassifier(
            flip_probability=0.3,
            sigma=1.0,
            max_acs_iter=50,
            max_dual_variables=2000,
            tol=1e-4,
            random_state=42,
        )
        clf.fit(X, y_pu)
        for entry in clf.acs_history_:
            assert entry["eq_residual"] < 1e-6, entry
