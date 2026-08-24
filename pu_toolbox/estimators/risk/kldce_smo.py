# ruff: noqa: N803, N806, S101

"""Native paired-SMO solver for the KLDCE fixed-μ dual QP (Appendix Eqs. 21-26).

The analytic two-variable update below is the fixed-μ QP (min ½zᵀQz − dᵀz
s.t. Aeq·z = beq, lb ≤ z ≤ ub) written in gradient form: the update keeps
Aeq·z exactly feasible and is optimal along the one-dimensional restricted
curve, hence numerically equivalent to the paper's paired SMO steps
(Eqs. 21-26, with the paper's E = f − ŷ error quantities).  Equivalent to
the appendix but not a line-by-line reproduction (method card §6).
"""

from __future__ import annotations

import numpy as np


def _smo_alpha_pair_update(
    z: np.ndarray,
    i1: int,
    i2: int,
    Q: np.ndarray,
    d: np.ndarray,
    a: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    """Analytic one-dimensional paired SMO step on variables (i1, i2).

    Returns the updated full variable vector (a new array).  If the pair
    admits no feasible step the input vector is returned unchanged.
    """
    z = z.copy()
    s = a[i1] * a[i2]
    g = Q @ z - d
    # 1D restricted-curve derivative  φ'(δ) = (g[i1] − s·g[i2]) + δ·H
    h = Q[i1, i1] - 2.0 * s * Q[i1, i2] + Q[i2, i2]
    num = g[i1] - s * g[i2]
    if h <= 0.0:
        return z  # non-positive curvature: degenerate, no descent guaranteed
    delta_star = -num / h
    # Bound δ by both boxes along the constraint direction
    low = lb[i1] - z[i1]
    high = ub[i1] - z[i1]
    z2_low = -z[i2] + lb[i2]
    z2_high = -z[i2] + ub[i2]
    if s > 0:  # z_j -= δ
        low = max(low, -z2_high)
        high = min(high, -z2_low)
    else:  # z_j += δ
        low = max(low, z2_low)
        high = min(high, z2_high)
    if low > high:
        return z  # no feasible movement along this pair
    delta = float(np.clip(delta_star, low, high))
    z[i1] += delta
    z[i2] -= s * delta
    return z


def _smo_gamma_pair_update(
    z: np.ndarray,
    i1: int,
    i2: int,
    Q: np.ndarray,
    d: np.ndarray,
    a: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    """γ 对更新:与 α 对同一解析步(由 a 的符号承载差异)。

    Kept as a separate entry point so the gamma-pair semantics (second group
    of the appendix) is explicit at the call site; delegates to the same
    analytic update.
    """
    return _smo_alpha_pair_update(z, i1, i2, Q, d, a, lb, ub)
