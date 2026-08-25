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


def _smo_pair_delta(
    z: np.ndarray,
    i1: int,
    i2: int,
    Q: np.ndarray,
    d: np.ndarray,
    a: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> float | None:
    """Feasible step size for pair (i1, i2), or None if no move is possible.

    Mirrors the box clipping inside ``_smo_alpha_pair_update``; returns None
    for pairs whose step would be zero (box-blocked at the current point) or
    that lie in a non-positive-curvature direction of Q, so callers can skip
    them instead of stalling on a no-progress update.
    """
    s = a[i1] * a[i2]
    g = Q @ z - d
    h = Q[i1, i1] - 2.0 * s * Q[i1, i2] + Q[i2, i2]
    if h <= 0.0:
        return None  # non-positive curvature: degenerate, no descent guaranteed
    # 1D restricted-curve derivative  φ'(δ) = (g[i1] − s·g[i2]) + δ·H
    num = g[i1] - s * g[i2]
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
        return None  # no feasible movement along this pair
    delta = float(np.clip(delta_star, low, high))
    if delta == 0.0:
        return None  # pair already optimal along the restricted curve
    return delta


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
    delta = _smo_pair_delta(z, i1, i2, Q, d, a, lb, ub)
    if delta is None:
        return z
    s = a[i1] * a[i2]
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


def kkt_violation_terms(
    z: np.ndarray,
    Q: np.ndarray,
    d: np.ndarray,
    a: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Per-variable KKT violation for min ½zᵀQz − dᵀz, Aeq·z = beq, lb ≤ z ≤ ub.

    Signs follow the convention used by ``_true_kkt_residual`` (kldce.py):
    with lag = g + ν·a, free variables must have lag ≈ 0; a variable at its
    lower bound must have lag ≥ 0 (violation = max(0, −lag)); at its upper
    bound lag ≤ 0 (violation = max(0, lag)).  ν is recovered as the median
    over free variables where the KKT forces g_i + ν·a_i = 0.
    """
    g = Q @ z - d
    a = np.asarray(a, dtype=float)
    free_mask = (z > lb + eps) & (z < ub - eps)
    free_idx = np.where(free_mask)[0]
    nu = None
    if free_idx.size:
        nu_candidates = [-g[i] / a[i] for i in free_idx if a[i] != 0.0 and free_mask[i]]
        if nu_candidates:
            nu = float(np.median(nu_candidates))
    lag = g.copy()
    if nu is not None:
        lag = g + nu * a
    viol = np.zeros_like(g)
    viol[free_mask] = np.abs(lag[free_mask])
    at_lb = (z <= lb + eps) & ~free_mask
    at_ub = (z >= ub - eps) & ~free_mask
    viol[at_lb] = np.maximum(0.0, -lag[at_lb])
    viol[at_ub] = np.maximum(0.0, lag[at_ub])
    return viol, {"nu": nu, "free_mask": free_mask}


def _smo_pair_select(
    z: np.ndarray,
    Q: np.ndarray,
    d: np.ndarray,
    a: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    eps: float = 1e-8,
) -> tuple[int, int] | None:
    """Select a violating pair (first = max violation, second = max |lag difference|).

    If all violations are ≤ ``eps`` returns None (KKT satisfied).  The
    second variable is chosen by largest |lag difference| among pairs that
    admit a non-zero analytic step (``_smo_pair_delta``); box-blocked or
    null-curvature pairs are skipped — selecting them would stall the solve
    loop on a no-progress update.  If no pair through ``i1`` can move, falls
    back to the largest-step pair anywhere in the vector; if no pair anywhere
    can move, returns None again — this second None means "stuck", NOT
    "KKT satisfied".  Callers must distinguish the two via
    ``kkt_violation_terms`` (viol.max() ≤ eps) rather than via None.
    """
    viol, info = kkt_violation_terms(z, Q, d, a, lb, ub, eps)
    i1 = int(np.argmax(viol))
    if viol[i1] <= eps:
        return None
    lag = Q @ z - d + info["nu"] * a if info["nu"] is not None else Q @ z - d
    for j in sorted(
        (j for j in range(z.shape[0]) if j != i1),
        key=lambda j: abs(lag[j] - lag[i1]),
        reverse=True,
    ):
        if _smo_pair_delta(z, i1, j, Q, d, a, lb, ub) is not None:
            return i1, j
    # No pair through i1 can move: scan the whole vector for the largest step.
    best_pair, best_mag = None, 0.0
    for i in range(z.shape[0]):
        for j in range(i + 1, z.shape[0]):
            delta = _smo_pair_delta(z, i, j, Q, d, a, lb, ub)
            if delta is not None and abs(delta) > best_mag:
                best_mag, best_pair = abs(delta), (i, j)
    return best_pair


def _smo_bias_average(b_values: list[float]) -> float:
    """Four-term bias average (appendix Eqs. 37-40)."""
    if not b_values:
        return 0.0
    return float(np.mean(np.asarray(b_values, dtype=float)))


def _solve_dual_smo(
    Q: np.ndarray,
    d_vec: np.ndarray,
    Aeq: np.ndarray,
    beq: float,
    lb: np.ndarray,
    ub: np.ndarray,
    z0: np.ndarray | None,
    *,
    tol: float = 1e-8,
    max_iter: int = 2000,
) -> tuple[np.ndarray, dict]:
    """Native paired-SMO solver for the KLDCE fixed-μ dual QP.

    Interface-compatible with ``kldce._solve_qp_oracle`` (the SLSQP oracle —
    retained for cross-checking only).  When ``z0`` is None, a feasible
    start point is obtained via ``kldce._find_feasible_init``.

    Convergence is KKT-based, not pair-availability-based: the loop stops
    when ``_smo_pair_select`` returns None, but None also fires when every
    pair is box-blocked / null-curvature while KKT violations remain (a
    stuck point).  ``inner_converged`` is therefore decided by recomputing
    ``kkt_violation_terms`` at the exit point (viol.max() ≤ tol); a stuck
    non-KKT point reports status=1, not a spurious convergence.
    """
    from pu_toolbox.estimators.risk.kldce import _find_feasible_init, _true_kkt_residual

    a = Aeq[0].astype(float) if Aeq.ndim == 2 else Aeq.astype(float)
    if z0 is None:
        z0 = _find_feasible_init(Aeq, beq, lb, ub)
    z = z0.astype(float).copy()

    n_iter = 0
    for _ in range(max_iter):
        pair = _smo_pair_select(z, Q, d_vec, a, lb, ub, eps=tol)
        if pair is None:
            break  # no movable pair: convergence decided by KKT below, not None
        i1, i2 = pair
        z = _smo_alpha_pair_update(z, i1, i2, Q, d_vec, a, lb, ub)
        n_iter += 1

    # KKT-based convergence: pair selection returns None both when the point
    # satisfies KKT and when every pair is stuck (box-blocked / null-curvature)
    # while KKT violations remain — only the former is convergence.
    viol, _ = kkt_violation_terms(z, Q, d_vec, a, lb, ub, eps=tol)
    converged = bool(viol.max() <= tol)

    dual_obj = float(d_vec @ z - 0.5 * z @ Q @ z)
    eq_residual = float(np.abs(Aeq @ z - beq).max())
    box_violation = float(max(np.maximum(lb - z, 0.0).max(), np.maximum(z - ub, 0.0).max()))
    kkt, nu, n_free, kkt_status = _true_kkt_residual(z, Q, d_vec, Aeq, beq, lb, ub)
    diagnostics = {
        "dual_obj": dual_obj,
        "eq_residual": eq_residual,
        "box_violation": box_violation,
        "kkt_residual": kkt,
        "kkt_nu": nu,
        "kkt_n_free": n_free,
        "kkt_status": kkt_status,
        "gradient_norm": float(np.linalg.norm(Q @ z - d_vec)),
        "status": 0 if converged else 1,
        "n_iter": n_iter,
        "inner_converged": converged,
    }
    return z, diagnostics
