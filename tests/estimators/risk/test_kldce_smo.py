# ruff: noqa: N802, N803, N806, S101

import numpy as np
import pytest  # noqa: F401

from pu_toolbox.estimators.risk.kldce_smo import (
    _smo_alpha_pair_update,
    _smo_gamma_pair_update,
)


def _simple_qp():
    """2-variable interior test: Q [[2,1],[1,4]], d [1,1], at z=[0.5,0.5]."""
    Q = np.array([[2.0, 1.0], [1.0, 4.0]])
    d = np.array([1.0, 1.0])
    return Q, d


def test_basic_alpha_pair_analytical_step():
    Q, d = _simple_qp()
    z = np.array([0.5, 0.5])
    # 沿约束 s = a1*a2 = 1 (both ±1 same sign): z_j -= δ
    a = np.array([1.0, 1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    z_new = _smo_alpha_pair_update(z, 0, 1, Q, d, a, lb, ub)
    # Equality conserved: a·z unchanged
    assert abs(z_new @ a - z @ a) < 1e-12
    # Objective not increased
    f = lambda v: 0.5 * v @ Q @ v - d @ v  # noqa: E731
    assert f(z_new) <= f(z) + 1e-12


def test_basic_alpha_pair_clips_through_gamma_signed_constraint():
    Q = np.array([[3.0, 0.5], [0.5, 2.0]])
    d = np.array([1.0, 1.0])
    z = np.array([0.0, 0.5])
    a = np.array([1.0, -1.0])  # opposite signs: z_j + δ  (z_j -= s*δ with s=-1)
    lb = np.array([0.0, 0.0])
    ub = np.array([0.5, 0.5])
    z_new = _smo_alpha_pair_update(z, 0, 1, Q, d, a, lb, ub)
    assert abs(z_new @ a - z @ a) < 1e-12
    assert z_new[0] >= lb[0] - 1e-12 and z_new[0] <= ub[0] + 1e-12
    assert z_new[1] >= lb[1] - 1e-12 and z_new[1] <= ub[1] + 1e-12
    f = lambda v: 0.5 * v @ Q @ v - d @ v  # noqa: E731
    assert f(z_new) <= f(z) + 1e-12


def test_basic_alpha_pair_no_feasible_step_returns_unchanged():
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = np.array([0.0, 0.0])
    z = np.array([0.0, 1.0])
    a = np.array([1.0, -1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    # s=-1: z_j += δ; z_i can only increase, z_j can only decrease → 无可行步
    z_new = _smo_alpha_pair_update(z, 0, 1, Q, d, a, lb, ub)
    assert np.allclose(z_new, z)


def test_basic_gamma_pair_update_delegates_same_analytic_step():
    """γ 对与 α 对同一数学(符号差只来自 a);入口分派返回同解。"""
    Q = np.array([[2.0, 0.3], [0.3, 1.5]])
    d = np.array([0.5, 0.7])
    z = np.array([0.2, 0.4])
    a = np.array([1.0, 1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([0.25, 0.25])
    z_alpha = _smo_alpha_pair_update(z.copy(), 0, 1, Q, d, a, lb, ub)
    z_gamma = _smo_gamma_pair_update(z.copy(), 0, 1, Q, d, a, lb, ub)
    assert np.allclose(z_gamma, z_alpha)
    assert abs(z_gamma @ a - z @ a) < 1e-12
    assert z_gamma[0] <= ub[0] + 1e-12
