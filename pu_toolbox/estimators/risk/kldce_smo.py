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
    g: np.ndarray | None = None,
) -> float | None:
    """Feasible step size for pair (i1, i2), or None if no move is possible.

    Mirrors the box clipping inside ``_smo_alpha_pair_update``; returns None
    for pairs whose step would be zero (box-blocked at the current point) or
    that lie in a non-positive-curvature direction of Q, so callers can skip
    them instead of stalling on a no-progress update.

    ``g`` (optional) is the maintained gradient Q·z − d at the current point
    (see ``_solve_dual_smo``); when None it is recomputed in full here.
    """
    s = a[i1] * a[i2]
    g = Q @ z - d if g is None else g
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
    g: np.ndarray | None = None,
) -> np.ndarray:
    """Analytic one-dimensional paired SMO step on variables (i1, i2).

    Returns the updated full variable vector (a new array).  If the pair
    admits no feasible step the input vector is returned unchanged.

    When ``g`` (the maintained gradient Q·z − d at the current point) is
    given, it is updated in place — only i1/i2 moved by ±δ, so exactly
    g′ = g + δ·(Q[:,i1] − s·Q[:,i2]) (O(n); Platt/LIBSVM kernel-cache style).
    When ``g`` is None the old full recompute path is used verbatim.
    """
    z = z.copy()
    delta = _smo_pair_delta(z, i1, i2, Q, d, a, lb, ub, g)
    if delta is None:
        return z
    s = a[i1] * a[i2]
    z[i1] += delta
    z[i2] -= s * delta
    if g is not None:
        g[:] += delta * (Q[:, i1] - s * Q[:, i2])
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
    g: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Per-variable KKT violation for min ½zᵀQz − dᵀz, Aeq·z = beq, lb ≤ z ≤ ub.

    Signs follow the convention used by ``_true_kkt_residual`` (kldce.py):
    with lag = g + ν·a, free variables must have lag ≈ 0; a variable at its
    lower bound must have lag ≥ 0 (violation = max(0, −lag)); at its upper
    bound lag ≤ 0 (violation = max(0, lag)).

    ν is recovered by intersecting the per-variable KKT obligations, each a
    half-line (or single point) in ν.  With t_i = −g_i/a_i the identity
    lag_i = g_i + ν·a_i = a_i·(ν − t_i) gives: a free variable forces ν = t_i;
    at lb, a_i > 0 forces ν ≥ t_i while a_i < 0 forces ν ≤ t_i; at ub the
    inequalities flip.  Collect lo = max of all "ν ≥ t" obligations and
    hi = min of all "ν ≤ t" obligations (single points count as both): the
    point is KKT-feasible iff lo ≤ hi, and any ν ∈ [lo, hi] is then a valid
    multiplier.  This handles vertex optima (zero free variables), where a
    multiplier exists but cannot be recovered from free-variable medians —
    the old ν = None sign-check degeneracy reported spurious violations at
    exactly such KKT-feasible vertices.

    ν is chosen as the median of the free-variable candidates −g_i/a_i,
    clamped into [lo, hi] when the intersection is non-empty (at a feasible
    point all free t_i coincide, so the clamp is a no-op and the legacy
    medians are preserved); when no free candidate exists, any ν ∈ [lo, hi]
    works and the midpoint is used (the lower endpoint if hi = +∞, the upper
    if lo = −∞).  When the intersection is empty the median is kept, so the
    per-variable signal matches the legacy behaviour, and
    ``nu_feasible_gap = max(0, lo − hi)`` / ``kkt_feasible`` report the
    infeasibility as a scalar.

    ``g`` (optional) is the maintained gradient Q·z − d at the current point
    (see ``_solve_dual_smo``); when None it is recomputed in full here.
    """
    g = Q @ z - d if g is None else g
    a = np.asarray(a, dtype=float)
    free_mask = (z > lb + eps) & (z < ub - eps)
    at_lb = (z <= lb + eps) & ~free_mask
    at_ub = (z >= ub - eps) & ~free_mask

    # Per-variable ν obligations via lag_i = a_i·(ν − t_i), t_i = −g_i/a_i.
    lo = -np.inf
    hi = np.inf
    for i in range(z.shape[0]):
        if a[i] == 0.0:
            continue  # no ν dependence; the lag itself is checked below
        t = -g[i] / a[i]
        if free_mask[i]:
            lo = max(lo, t)
            hi = min(hi, t)
        elif at_lb[i]:  # lag_i ≥ 0
            if a[i] > 0.0:
                lo = max(lo, t)
            else:
                hi = min(hi, t)
        elif at_ub[i]:  # lag_i ≤ 0
            if a[i] > 0.0:
                hi = min(hi, t)
            else:
                lo = max(lo, t)

    nu_candidates = [-g[i] / a[i] for i in np.where(free_mask)[0] if a[i] != 0.0]
    if nu_candidates:
        nu = float(np.median(nu_candidates))
        if lo <= hi:  # feasible: project the median into the interval (no-op)
            nu = float(min(max(nu, lo), hi))
    elif np.isfinite(lo) and np.isfinite(hi):
        nu = 0.5 * (lo + hi)
    elif np.isfinite(lo):
        nu = lo
    elif np.isfinite(hi):
        nu = hi
    else:
        nu = 0.0

    lag = g + nu * a
    viol = np.zeros_like(g)
    viol[free_mask] = np.abs(lag[free_mask])
    viol[at_lb] = np.maximum(0.0, -lag[at_lb])
    viol[at_ub] = np.maximum(0.0, lag[at_ub])
    return viol, {
        "nu": nu,
        "free_mask": free_mask,
        "nu_interval": (lo, hi),
        "kkt_feasible": bool(lo <= hi),
        "nu_feasible_gap": float(max(0.0, lo - hi)),
    }


def _smo_pair_select(
    z: np.ndarray,
    Q: np.ndarray,
    d: np.ndarray,
    a: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    eps: float = 1e-8,
    g: np.ndarray | None = None,
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

    ``g`` (optional) is the maintained gradient Q·z − d at the current point
    (see ``_solve_dual_smo``); when None every use recomputes it in full.
    """
    viol, info = kkt_violation_terms(z, Q, d, a, lb, ub, eps, g)
    i1 = int(np.argmax(viol))
    if viol[i1] <= eps:
        return None
    lag = (g if g is not None else (Q @ z - d)) + info["nu"] * a
    for j in sorted(
        (j for j in range(z.shape[0]) if j != i1),
        key=lambda j: abs(lag[j] - lag[i1]),
        reverse=True,
    ):
        if _smo_pair_delta(z, i1, j, Q, d, a, lb, ub, g) is not None:
            return i1, j
    # No pair through i1 can move: scan the whole vector for the largest step.
    best_pair, best_mag = None, 0.0
    for i in range(z.shape[0]):
        for j in range(i + 1, z.shape[0]):
            delta = _smo_pair_delta(z, i, j, Q, d, a, lb, ub, g)
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
    tol: float = 1e-6,
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
    non-KKT point reports status=1, not a spurious convergence.  Points with
    no free variables (vertex optima) are handled via the ν-interval
    semantics of ``kkt_violation_terms``: at a KKT-feasible vertex any
    ν in the non-empty allowed interval gives zero violations, so the
    recompute correctly reports convergence there.

    Gradient maintenance: each pair update moves only two components, so the
    gradient g = Q·z − d is maintained incrementally — g′ = g + δ·(Q[:,i1] −
    s·Q[:,i2]), exact up to machine precision (Platt/LIBSVM kernel-cache
    style).  g is fully recomputed once per solve and again every 256 pairs
    as a drift guard; drift is O(pairs·ε) ≈ 1e-12 ≪ tol either way.
    """
    from pu_toolbox.estimators.risk.kldce import _find_feasible_init, _true_kkt_residual

    a = Aeq[0].astype(float) if Aeq.ndim == 2 else Aeq.astype(float)
    if z0 is None:
        z0 = _find_feasible_init(Aeq, beq, lb, ub)
    z = z0.astype(float).copy()

    n_iter = 0
    g = Q @ z - d_vec  # full recompute per solve; incremental in the loop
    for _ in range(max_iter):
        pair = _smo_pair_select(z, Q, d_vec, a, lb, ub, eps=tol, g=g)
        if pair is None:
            break  # no movable pair: convergence decided by KKT below, not None
        i1, i2 = pair
        z = _smo_alpha_pair_update(z, i1, i2, Q, d_vec, a, lb, ub, g=g)
        n_iter += 1
        if n_iter % 256 == 0:
            g = Q @ z - d_vec  # periodic full recompute: drift guard (≈1e-12)

    # KKT-based convergence: pair selection returns None both when the point
    # satisfies KKT and when every pair is stuck (box-blocked / null-curvature)
    # while KKT violations remain — only the former is convergence.  The ν
    # interval semantics of kkt_violation_terms make viol ≈ 0 at KKT-feasible
    # vertices (zero free variables), so those report convergence too.
    viol, info = kkt_violation_terms(z, Q, d_vec, a, lb, ub, eps=tol)
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
        "nu_feasible_gap": info["nu_feasible_gap"],
        "kkt_feasible": info["kkt_feasible"],
        "status": 0 if converged else 1,
        "n_iter": n_iter,
        "inner_converged": converged,
    }
    return z, diagnostics
