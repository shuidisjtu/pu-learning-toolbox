# ruff: noqa: N803, N806, S101

"""KLDCE: Kernelized Loss Decomposition and Centroid Estimation for PU Learning.

Implements the kernelized LDCE classifier (RBF kernel) from:

    Gong, C., Shi, H., Liu, T., Zhang, C., Yang, J., & Tao, D.
    "Loss Decomposition and Centroid Estimation for Positive and
    Unlabeled Learning."  IEEE TPAMI, 2021.
    (Online supplementary appendix — Algorithm 1, Eqs. 21–40.)

The method alternates between two subproblems in an ACS (Alternating
Convex Search) outer loop:

1. **Fixed centroid μ**: solve a joint dual QP over α and γ via
   ``scipy.optimize.minimize`` (QP oracle).
2. **Fixed α, γ**: update the centroid μ under the ellipsoid constraint
   using the RBF Taylor expansion (Appendix Eq. 33–35).

**Delivery strategy (first version):**
- ACS outer loop + scipy QP oracle for the fixed-μ dual QP.
- RBF kernel only (centroid update relies on Gaussian Taylor expansion).
- Appendix-native SMO solver deferred to a follow-up PR.

**Note on QP oracle vs paper Algorithm 1:**
The paper's Algorithm 1 first updates μ, then runs SMO on α/γ.
Here we first fix μ and solve the joint QP, then update μ — both are
valid block-coordinate orderings within ACS, but this implementation
is NOT a line-by-line reproduction of Algorithm 1.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.optimize
import scipy.spatial.distance
from scipy.linalg import solve

from ...core.base import BasePUClassifier
from ...core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    ImplementationStatus,
    Maturity,
    Scenario,
    SourceStatus,
)
from ...core.validation import validate_pu_X_y
from ...utils.centroid import _centroid_covariance, _mom_centroid

# ═════════════════════════════════════════════════════════════════════
# Kernel
# ═════════════════════════════════════════════════════════════════════


def _rbf_kernel(
    X: np.ndarray,
    Z: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """RBF / Gaussian kernel.

    .. math::

        K(x,z) = \\exp\\left(-\\frac{\\|x-z\\|^2}{2\\sigma^2}\\right)

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
    Z : np.ndarray of shape (m, d)
    sigma : float
        Bandwidth.  Relation to sklearn gamma:
        :math:`\\gamma = 1 / (2\\sigma^2)`.

    Returns
    -------
    K : np.ndarray of shape (n, m)
    """
    sqdist = scipy.spatial.distance.cdist(X, Z, "sqeuclidean")
    return np.exp(-sqdist / (2.0 * sigma ** 2))


# ═════════════════════════════════════════════════════════════════════
# Phase-I feasible initialisation
# ═════════════════════════════════════════════════════════════════════


def _find_feasible_init(
    Aeq: np.ndarray,
    beq: float,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    """Find a feasible initial point z₀ for the dual QP.

    Solves the Phase-I LP:

        min 0  s.t.  Aeq @ z = beq,  lb ≤ z ≤ ub

    via ``scipy.optimize.linprog`` (HiGHS method).

    Returns a feasible point or raises ``RuntimeError``.
    """
    N = lb.shape[0]
    result = scipy.optimize.linprog(
        c=np.zeros(N),
        A_eq=Aeq,
        b_eq=np.array([beq]),
        bounds=list(zip(lb, ub, strict=True)),
        method="highs",
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(
            f"Phase-I feasible initialisation failed: {result.message}. "
            "Aeq·z = beq may be infeasible for the given box constraints."
        )
    z0 = result.x

    eq_res = float(np.abs(Aeq @ z0 - beq).max())
    if eq_res > 1e-10:
        raise RuntimeError(
            f"Phase-I LP returned point with equality residual {eq_res:.2e} > 1e-10."
        )
    if (z0 < lb - 1e-12).any() or (z0 > ub + 1e-12).any():
        raise RuntimeError(
            "Phase-I LP returned point violating box constraints."
        )

    return z0


# ═════════════════════════════════════════════════════════════════════
# Dual QP builder (Appendix Eq. 24)
# ═════════════════════════════════════════════════════════════════════


def _build_dual_qp(
    mu: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    y_tilde: np.ndarray,
    lambda_: float,
    sigma: float,
    n: int,
    k: int,
    C_eq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Build the fixed-μ dual QP (Appendix Eq. 24).

    Let :math:`N = n + n_U` where :math:`n_U = n - k`.

    The dual variable vector:
        :math:`z = [\\alpha_1 \\ldots \\alpha_n \\mid \\gamma_{k+1} \\ldots \\gamma_n]`

    The QP:
        max  d(μ)ᵀ z − ½ zᵀ Q z
        s.t. Aeq·z = C_eq,  0 ≤ z ≤ ub

    Parameters
    ----------
    mu : np.ndarray of shape (d,)
        Current centroid estimate.
    X : np.ndarray of shape (n, d)
        Feature matrix.
    K : np.ndarray of shape (n, n)
        Precomputed kernel matrix K(x_i, x_j).
    y_tilde : np.ndarray of shape (n,)
        Observed labels: +1 for P, -1 for U.
    lambda_ : float
        Regularisation λ.
    sigma : float
        RBF bandwidth (for K(x_i, μ)).
    n : int
        Total samples.
    k : int
        Number of labeled positives.
    C_eq : float
        Equality constraint RHS.

    Returns
    -------
    Q : np.ndarray of shape (N, N)
    d_vec : np.ndarray of shape (N,)
    Aeq : np.ndarray of shape (1, N)
    beq : float
    lb : np.ndarray of shape (N,)
    ub : np.ndarray of shape (N,)
    """
    n_U = n - k
    N = n + n_U

    C_alpha = 1.0 / n
    C_gamma = 1.0 / (2.0 * n)

    # ── Q = 1/(2λ) · [[Q_αα, Q_αγ], [Q_γα, Q_γγ]] ─────────────────
    yy = np.outer(y_tilde, y_tilde)  # (n, n)

    Q_alpha_alpha = yy * K  # (n, n)
    Q_alpha_gamma = -(yy[:n, k:] * K[:n, k:])  # (n, n_U)
    Q_gamma_gamma = yy[k:, k:] * K[k:, k:]  # (n_U, n_U)

    N_total = N
    Q = np.zeros((N_total, N_total), dtype=float)
    Q[:n, :n] = Q_alpha_alpha
    Q[:n, n:] = Q_alpha_gamma
    Q[n:, :n] = Q_alpha_gamma.T
    Q[n:, n:] = Q_gamma_gamma
    Q *= 1.0 / (2.0 * lambda_)

    # ── Linear term d(μ) ───────────────────────────────────────────
    # d(μ)_i = 1 + C_eq·ỹᵢ·K(xᵢ, μ)/(2λ)               (α part)
    # d(μ)_{n+i} = 1 - C_eq·ỹ_{k+i}·K(x_{k+i}, μ)/(2λ)   (γ part)
    K_mu = _rbf_kernel(X, mu.reshape(1, -1), sigma).ravel()  # (n,)

    d_vec = np.zeros(N_total, dtype=float)
    d_vec[:n] = 1.0 + C_eq * y_tilde * K_mu / (2.0 * lambda_)
    # γ part: only U samples (indices k..n-1)
    d_vec[n:] = 1.0 - C_eq * y_tilde[k:] * K_mu[k:] / (2.0 * lambda_)

    # ── Constraints ────────────────────────────────────────────────
    # Aeq = [ỹ₁…ỹₙ | −ỹ_{k+1}…−ỹₙ]
    Aeq = np.zeros((1, N_total), dtype=float)
    Aeq[0, :n] = y_tilde
    Aeq[0, n:] = -y_tilde[k:]
    beq = C_eq

    lb = np.zeros(N_total, dtype=float)
    ub = np.concatenate([
        np.full(n, C_alpha),
        np.full(n_U, C_gamma),
    ])

    return Q, d_vec, Aeq, beq, lb, ub


# ═════════════════════════════════════════════════════════════════════
# QP oracle (scipy SLSQP)
# ═════════════════════════════════════════════════════════════════════


def _solve_qp_oracle(
    Q: np.ndarray,
    d_vec: np.ndarray,
    Aeq: np.ndarray,
    beq: float,
    lb: np.ndarray,
    ub: np.ndarray,
    z0: np.ndarray,
    *,
    tol: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Solve the dual QP via scipy SLSQP.

    Converts the maximisation problem:

        max  dᵀ z − ½ zᵀ Q z   s.t.  Aeq·z = beq,  lb ≤ z ≤ ub

    into minimisation of ½ zᵀ Q z − dᵀ z.

    Parameters
    ----------
    Q : np.ndarray of shape (N, N)
    d_vec : np.ndarray of shape (N,)
    Aeq : np.ndarray of shape (1, N)
    beq : float
    lb, ub : np.ndarray of shape (N,)
    z0 : np.ndarray of shape (N,)
        Warm-start point (must be feasible).
    tol : float
        Tolerance passed to SLSQP.

    Returns
    -------
    z : np.ndarray of shape (N,)
        Optimal dual variables [α; γ].
    diagnostics : dict
        Keys: dual_obj, eq_residual, box_violation, kkt_residual, status, n_iter.
    """
    N = len(d_vec)

    def objective(z: np.ndarray) -> float:
        return float(0.5 * z @ Q @ z - d_vec @ z)

    def gradient(z: np.ndarray) -> np.ndarray:
        return Q @ z - d_vec

    constraints = [
        {"type": "eq", "fun": lambda z: Aeq @ z - beq,
         "jac": lambda z: Aeq[0]},
    ]
    bounds = scipy.optimize.Bounds(lb, ub)

    result = scipy.optimize.minimize(
        objective,
        z0,
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": tol, "maxiter": 1000, "disp": False},
    )

    z = result.x
    dual_obj = float(d_vec @ z - 0.5 * z @ Q @ z)
    eq_residual = float(np.abs(Aeq @ z - beq).max())
    box_violation = float(
        max(np.maximum(lb - z, 0.0).max(), np.maximum(z - ub, 0.0).max())
    )

    # Approximate KKT residual: ‖Qz − d + Aeqᵀν + λ_upper − λ_lower‖
    # We approximate via the gradient of the Lagrangian at the solution.
    kkt_residual = float(np.linalg.norm(result.jac if hasattr(result, 'jac') else np.zeros(N)))

    diagnostics = {
        "dual_obj": dual_obj,
        "eq_residual": eq_residual,
        "box_violation": box_violation,
        "kkt_residual": kkt_residual,
        "status": result.status,
        "n_iter": getattr(result, "nit", 0),
    }

    return z, diagnostics


# ═════════════════════════════════════════════════════════════════════
# RBF centroid delta (Appendix Eq. 33, Taylor expansion at μ=0)
# ═════════════════════════════════════════════════════════════════════


def _rbf_centroid_delta(
    alpha: np.ndarray,
    gamma: np.ndarray,
    X: np.ndarray,
    y_tilde: np.ndarray,
    lambda_: float,
    sigma: float,
    k: int,
) -> np.ndarray:
    """Compute the centroid update direction Δ (Appendix Eq. 33).

    Taylor expansion of K(x, μ) around μ = 0:

    .. math::

        \\frac{\\partial K(x, \\mu)}{\\partial \\mu}\\bigg|_{\\mu=0}
        = \\exp\\left(-\\frac{\\|x\\|^2}{2\\sigma^2}\\right) \\cdot
        \\frac{x}{\\sigma^2}

    Then:

    .. math::

        \\Delta = -\\frac{1}{2\\lambda\\sigma^2}
            \\sum_i \\alpha_i \\tilde{y}_i
            \\exp(-\\|x_i\\|^2/(2\\sigma^2)) x_i \\\\
            + \\frac{1}{2\\lambda\\sigma^2}
            \\sum_{i=k+1}^n \\gamma_i \\tilde{y}_i
            \\exp(-\\|x_i\\|^2/(2\\sigma^2)) x_i

    Parameters
    ----------
    alpha : np.ndarray of shape (n,)
        Dual variables for all samples.
    gamma : np.ndarray of shape (n_U,)
        Dual variables for unlabeled samples.
    X : np.ndarray of shape (n, d)
        Feature matrix.
    y_tilde : np.ndarray of shape (n,)
        Observed labels (+1 for P, -1 for U).
    lambda_ : float
        Regularisation λ.
    sigma : float
        RBF bandwidth.
    k : int
        Number of labeled positives (first k rows of X).

    Returns
    -------
    delta : np.ndarray of shape (d,)
        Update direction (row vector).
    """
    n, d = X.shape
    scale = 1.0 / (2.0 * lambda_ * sigma ** 2)

    # exp(-||x_i||^2 / (2σ^2)) for all samples
    sq_norms = np.sum(X ** 2, axis=1)  # (n,)
    weights = np.exp(-sq_norms / (2.0 * sigma ** 2))  # (n,)

    # α contribution (all samples, negative sign)
    alpha_weighted = alpha * y_tilde * weights  # (n,)
    delta = -scale * (alpha_weighted @ X)  # (d,)

    # γ contribution (only U samples, positive sign)
    gamma_weighted = gamma * y_tilde[k:] * weights[k:]  # (n_U,)
    delta += scale * (gamma_weighted @ X[k:])  # (d,)

    return delta


# ═════════════════════════════════════════════════════════════════════
# Centroid update (Appendix Eq. 35)
# ═════════════════════════════════════════════════════════════════════


def _update_centroid(
    m_hat: np.ndarray,
    S_raw: np.ndarray,
    S_solve: np.ndarray,
    delta: np.ndarray,
    centroid_radius: float,
    tol: float,
) -> tuple[np.ndarray, dict]:
    """Update centroid μ under ellipsoid constraint (Appendix Eq. 35).

    1. Solve :math:`S_{\\text{solve}} \\cdot u = \\Delta^\\top`
    2. Compute :math:`q = u^\\top S_{\\text{raw}} u`
       (constraint scaling always uses S_raw)
    3. If :math:`q \\le \\text{tol}`: :math:`\\mu = m_{\\text{hat}}`
       (degenerate step)
    4. Else: :math:`\\mu = m_{\\text{hat}} - u \\cdot
       \\sqrt{\\text{centroid\\_radius} / q}`

    Parameters
    ----------
    m_hat : np.ndarray of shape (d,)
        MoM initial centroid.
    S_raw : np.ndarray of shape (d, d)
        Raw centroid covariance (without ridge).
    S_solve : np.ndarray of shape (d, d)
        Covariance used for linear solve (may include ridge).
        With ``covariance_ridge=0``, S_solve == S_raw.
    delta : np.ndarray of shape (d,)
        Update direction from _rbf_centroid_delta.
    centroid_radius : float
        Ellipsoid radius b.
    tol : float
        Tolerance for degenerate step detection.

    Returns
    -------
    mu : np.ndarray of shape (d,)
        Updated centroid.
    info : dict
        Keys: degenerate_centroid_step, constraint_residual,
        constraint_violated, centroid_solver.

    Raises
    ------
    np.linalg.LinAlgError
        If ``covariance_ridge=0`` and S_raw is near-singular.
    """
    info: dict = {
        "degenerate_centroid_step": False,
        "constraint_residual": 0.0,
        "constraint_violated": False,
        "centroid_solver": "solve",
    }

    # ── Solve S_solve · u = delta ──────────────────────────────────
    try:
        u = solve(S_solve, delta)
    except np.linalg.LinAlgError:
        cond = float(np.linalg.cond(S_raw))
        raise np.linalg.LinAlgError(
            f"S_raw is near-singular (condition number = {cond:.2e}). "
            "Use covariance_ridge > 0 for numerical stabilization."
        ) from None

    # ── Constraint scaling: q = uᵀ · S_raw · u ─────────────────────
    q = float(u @ S_raw @ u)

    if q <= tol:
        # Degenerate step — fall back to m_hat
        info["degenerate_centroid_step"] = True
        info["constraint_residual"] = 0.0
        return m_hat.copy(), info

    # ── μ = m_hat - u · √(b / q) ───────────────────────────────────
    mu = m_hat - u * np.sqrt(centroid_radius / q)

    # ── Verify constraint: (μ-m̂)ᵀ S_raw (μ-m̂) ≤ b + tol ──────────
    diff = mu - m_hat
    constraint = float(diff @ S_raw @ diff)
    info["constraint_residual"] = constraint

    if constraint > centroid_radius + tol:
        info["constraint_violated"] = True
        # Project onto ellipsoid boundary
        if constraint > 1e-15:
            mu = m_hat + diff * np.sqrt(centroid_radius / constraint)
            # Recompute after projection
            diff2 = mu - m_hat
            constraint = float(diff2 @ S_raw @ diff2)
            info["constraint_residual"] = constraint
            info["centroid_solver"] = "projected"

    return mu, info


# ═════════════════════════════════════════════════════════════════════
# Bias recovery from KKT conditions (QP oracle version)
# ═════════════════════════════════════════════════════════════════════


def _recover_bias_from_kkt(
    alpha: np.ndarray,
    gamma: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    y_tilde: np.ndarray,
    mu: np.ndarray,
    lambda_: float,
    sigma: float,
    C_eq: float,
    C_alpha: float,
    C_gamma: float,
    k: int,
) -> tuple[float, dict]:
    """Recover bias b₀ from KKT margin conditions (QP oracle version).

    For free support vectors (0 < αᵢ < C_alpha or 0 < γⱼ < C_gamma),
    the KKT conditions imply a margin of exactly 1:

        bᵢ = 1 − gᵢ   where gᵢ = f(xᵢ) − b₀

    We compute the bias-free decision scores gᵢ for all samples, then
    take the median of {bᵢ} from free variables.

    When no free variables exist, we fall back to the bounded-interval
    method using KKT inequality conditions.

    Parameters
    ----------
    alpha : np.ndarray of shape (n,)
    gamma : np.ndarray of shape (n_U,)
    X : np.ndarray of shape (n, d)
    K : np.ndarray of shape (n, n)
    y_tilde : np.ndarray of shape (n,)
    mu : np.ndarray of shape (d,)
    lambda_ : float
    sigma : float
    C_eq : float
    C_alpha, C_gamma : float
        Box constraint upper bounds.
    k : int
        Number of labeled positives.

    Returns
    -------
    b0 : float
        Estimated bias.
    info : dict
        Keys: bias_recovery, n_free.
    """
    # ── Compute bias-free decision scores gᵢ = f(xᵢ) − b₀ ─────────
    # f(x) = [ΣαᵢỹᵢK(x,xᵢ) − Σγⱼỹ_{k+j}K(x,x_{k+j}) − C_eq·K(x,μ)]/(2λ) + b₀
    # So g = f − b₀ = above expression without b₀.
    K_mu = _rbf_kernel(X, mu.reshape(1, -1), sigma).ravel()  # (n,)

    alpha_y = alpha * y_tilde  # (n,)

    # contribution from α terms
    g_alpha = K @ alpha_y  # (n,)

    # contribution from γ terms (subtract)
    gamma_y = gamma * y_tilde[k:]  # (n_U,)
    g_gamma = K[:, k:] @ gamma_y  # (n,)

    # centroid term
    g_centroid = C_eq * K_mu  # (n,)

    g = (g_alpha - g_gamma - g_centroid) / (2.0 * lambda_)

    # ── Collect free support vectors ────────────────────────────────
    info: dict = {"bias_recovery": "free_median", "n_free": 0}

    b_estimates = []

    # Free α (all samples): 0 < αᵢ < C_alpha, ỹᵢ = +1
    free_alpha_mask = (alpha > 1e-12) & (alpha < C_alpha - 1e-12)
    for i in np.where(free_alpha_mask)[0]:
        b_estimates.append(1.0 - g[i])

    # Free γ (U samples): 0 < γⱼ < C_gamma
    free_gamma_mask = (gamma > 1e-12) & (gamma < C_gamma - 1e-12)
    for j in np.where(free_gamma_mask)[0]:
        # ỹ_{k+j} = -1
        b_estimates.append(1.0 - g[k + j])

    if len(b_estimates) > 0:
        b0 = float(np.median(b_estimates))
        info["n_free"] = len(b_estimates)
        info["bias_recovery"] = "free_median"
        return b0, info

    # ── Fallback: bounded interval from KKT inequalities ───────────
    # L = lower bound on b₀, U = upper bound
    # margin = 1 − gᵢ for ỹ=+1, margin = y·(g+b) ≥ 1 → free: b ≥ 1−g
    # ỹ=+1: KKT condition αᵢ(f(xᵢ)−1)≥0
    #   αᵢ=0: f(xᵢ) ≥ 1 → b₀ ≥ 1 − gᵢ  (lower bound)
    #   αᵢ=C: f(xᵢ) ≤ 1 → b₀ ≤ 1 − gᵢ  (upper bound)
    # ỹ=−1: KKT condition γⱼ(−f(x_{k+j})−1)≥0
    #   γⱼ=0: −f ≥ 1 → f ≤ −1 → b₀ ≤ −1 − g  (upper bound)
    #   γⱼ=C: −f ≤ 1 → f ≥ −1 → b₀ ≥ −1 − g  (lower bound)

    L_parts: list[float] = []
    U_parts: list[float] = []

    # α at lower bound (α=0)
    alpha_lo = alpha <= 1e-12
    for i in np.where(alpha_lo)[0]:
        if y_tilde[i] > 0:
            L_parts.append(1.0 - g[i])  # ỹ=+1, α=0 → b ≥ 1−g
        else:
            U_parts.append(-1.0 - g[i])  # ỹ=−1, α=0 → b ≤ −1−g

    # α at upper bound (α=C_alpha)
    alpha_hi = alpha >= C_alpha - 1e-12
    for i in np.where(alpha_hi)[0]:
        if y_tilde[i] > 0:
            U_parts.append(1.0 - g[i])  # ỹ=+1, α=C → b ≤ 1−g
        else:
            L_parts.append(-1.0 - g[i])  # ỹ=−1, α=C → b ≥ −1−g

    # γ at lower bound (γ=0, U samples only)
    gamma_lo = gamma <= 1e-12
    for j in np.where(gamma_lo)[0]:
        # ỹ_{k+j} = −1 (always for U samples)
        U_parts.append(1.0 - g[k + j])  # γ=0, ỹ=−1 → b ≤ 1−g

    # γ at upper bound (γ=C_gamma, U samples only)
    gamma_hi = gamma >= C_gamma - 1e-12
    for j in np.where(gamma_hi)[0]:
        # ỹ_{k+j} = −1 (always for U samples)
        L_parts.append(1.0 - g[k + j])  # γ=C, ỹ=−1 → b ≥ 1−g

    if L_parts and U_parts:
        L_val = max(L_parts)
        U_val = min(U_parts)
        if L_val <= U_val:
            b0 = (L_val + U_val) / 2.0
            info["bias_recovery"] = "bounded_interval"
            return b0, info

    b0 = 0.0
    info["bias_recovery"] = "indeterminate"
    return b0, info


# ═════════════════════════════════════════════════════════════════════
# KLDCEClassifier
# ═════════════════════════════════════════════════════════════════════


class KLDCEClassifier(BasePUClassifier):
    """Kernelized LDCE classifier for censoring PU learning.

    Implements the kernelized version of Loss Decomposition and Centroid
    Estimation (KLDCE) from Gong et al. (TPAMI 2021).  Uses an RBF kernel
    and alternates between solving a dual QP (fixed centroid) and updating
    the centroid via RBF Taylor expansion (fixed dual variables).

    Parameters
    ----------
    flip_probability : float
        Probability *h* that a true positive is flipped to an observed
        negative (censoring rate).  Must be in ``(0, 1)``.  **Required.**
    sigma : float or ``"scale"``, default ``"scale"``
        RBF kernel bandwidth σ.  ``"scale"`` (default) uses the heuristic
        ``σ = 1 / sqrt(n_features)``.
    reg_strength : float, default 1.0
        L2 regularisation coefficient λ.
    centroid_radius : float, default 1.0
        Ellipsoid radius *b* for the centroid constraint.
    mom_groups : int, default 10
        Number of groups *g* for median-of-means centroid estimation.
        ``g=1`` degenerates to the ordinary mean.
    covariance_ridge : float, default 0.0
        Ridge penalty added to the centroid covariance diagonal.
        Default 0.0 matches the paper formula; values > 0 are an opt-in
        numerical stabilisation variant.
    max_acs_iter : int, default 50
        Maximum number of ACS outer loop iterations.
    max_dual_variables : int, default 1000
        Hard limit on the number of dual variables ``n + n_U``.
    tol : float, default 1e-6
        Convergence tolerance for the ACS loop (relative objective change,
        centroid change, KKT violation).
    random_state : int or None, default None
        Seed for MoM shuffling.

    Attributes
    ----------
    alpha_full_ : np.ndarray of shape (n,)
        Dual variables α for all samples.
    gamma_unlabeled_ : np.ndarray of shape (n_U,)
        Dual variables γ for unlabeled samples.
    unlabeled_indices_ : np.ndarray of shape (n_U,)
        Indices of U samples in the training data.
    support_indices_ : np.ndarray of shape (n_sv,)
        Indices where α≠0 or γ≠0.
    bias_ : float
        Recovered bias b₀.
    class_prior_ : float
        Estimated positive class prior p = k/[n(1−h)].
    flip_probability_ : float
        Validated flip probability.
    centroid_hat_ : np.ndarray of shape (n_features,)
        MoM initial centroid m̂.
    centroid_opt_ : np.ndarray of shape (n_features,)
        Optimised centroid μ.
    centroid_covariance_raw_ : np.ndarray of shape (n_features, n_features)
        Raw centroid covariance Ŝ (no ridge).
    C_eq_ : float
        Equality constraint RHS constant.
    n_acs_iter_ : int
        Number of ACS iterations performed.
    acs_history_ : list of dict
        Per-iteration diagnostics.
    converged_ : bool
        Whether ACS converged.
    classes_ : np.ndarray of shape (2,)
        ``np.array([0, 1])``.

    Notes
    -----
    - First version supports RBF kernel only (centroid update is
      Gaussian-Taylor-specific).
    - The QP oracle uses scipy SLSQP, not the paper's native SMO.
    - ``predict_proba`` is not implemented (raises NotImplementedError).
    """

    # ── Class-level metadata ──────────────────────────────────────────
    family: AlgorithmFamily = AlgorithmFamily.RISK_ESTIMATION
    assumption: tuple[Assumption, ...] = (Assumption.SCAR,)
    scenario: tuple[Scenario, ...] = (Scenario.SINGLE_TRAINING_SET,)
    requires_class_prior: bool = False
    implementation_status: ImplementationStatus = ImplementationStatus.NATIVE
    source_status: SourceStatus = SourceStatus.OFFICIAL_RELATED
    backend: Backend = Backend.NUMPY
    maturity: Maturity = Maturity.RESEARCH

    def __init__(
        self,
        flip_probability: float,
        *,
        sigma: float | str = "scale",
        reg_strength: float = 1.0,
        centroid_radius: float = 1.0,
        mom_groups: int = 10,
        covariance_ridge: float = 0.0,
        max_acs_iter: int = 50,
        max_dual_variables: int = 1000,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.flip_probability = flip_probability
        self.sigma = sigma
        self.reg_strength = reg_strength
        self.centroid_radius = centroid_radius
        self.mom_groups = mom_groups
        self.covariance_ridge = covariance_ridge
        self.max_acs_iter = max_acs_iter
        self.max_dual_variables = max_dual_variables
        self.tol = tol
        self.random_state = random_state

    # ── fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> KLDCEClassifier:
        """Fit the KLDCE classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.  Must be dense.
        y_pu : np.ndarray of shape (n_samples,)
            PU labels ``{+1, 0}`` or ``{+1, -1}``.
        class_prior : float, optional
            Override the derived class prior.
        sample_weight : np.ndarray, optional
            Accepted for sklearn API compatibility; currently ignored.

        Returns
        -------
        self : KLDCEClassifier
        """
        X, y_pu = validate_pu_X_y(
            X, y_pu,
            accept_sparse=False,
            estimator_name="KLDCEClassifier",
        )

        # ── Input sanity ─────────────────────────────────────────────
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values.")
        if not np.isfinite(y_pu).all():
            raise ValueError("y_pu contains NaN or Inf values.")

        # ── Validate flip_probability ─────────────────────────────────
        h = float(self.flip_probability)
        if not (0.0 < h < 1.0):
            raise ValueError(
                f"flip_probability must be in (0, 1); got {h}."
            )
        self.flip_probability_ = h

        # ── Split P / U ──────────────────────────────────────────────
        mask_P = y_pu == 1
        X_P = X[mask_P]
        X_U = X[~mask_P]
        k = X_P.shape[0]
        n_U = X_U.shape[0]
        n = X.shape[0]
        d = X.shape[1]

        if k == 0:
            raise ValueError("Need at least 1 labeled positive sample.")
        if n_U == 0:
            raise ValueError("Need at least 1 unlabeled sample.")

        # ── Input validation (§4 step 2) ─────────────────────────────
        if self.mom_groups < 1:
            raise ValueError(
                f"mom_groups must be >= 1; got {self.mom_groups}."
            )
        if self.mom_groups > n_U:
            raise ValueError(
                f"mom_groups ({self.mom_groups}) cannot exceed n_U ({n_U})."
            )
        n_dual = n + n_U
        if n_dual > self.max_dual_variables:
            raise ValueError(
                f"Number of dual variables ({n_dual}) exceeds "
                f"max_dual_variables ({self.max_dual_variables})."
            )

        # ── Class prior (§4 step 3) ──────────────────────────────────
        if class_prior is not None:
            p = float(class_prior)
            if not (0.0 < p <= 1.0):
                raise ValueError(
                    f"class_prior must be in (0, 1]; got {p}."
                )
        else:
            p = k / (n * (1.0 - h))
            if not (0.0 < p <= 1.0):
                raise ValueError(
                    f"Derived class prior p = {p} is out of (0, 1]. "
                    f"Check flip_probability (h={h}) and data: "
                    f"k={k}, n={n}. Formula: p = k / [n·(1−h)]."
                )
        self.class_prior_ = p

        # ── Check near-singular denominator (§4 step 3) ──────────────
        denom = 1.0 - 2.0 * p * h
        if abs(denom) < 1e-12:
            raise ValueError(
                f"Denominator 1−2ph = {denom:.2e} is near-zero "
                f"(h={h}, p={p}). The centroid term is ill-conditioned."
            )

        # ── Validate other hyper-parameters ───────────────────────────
        if self.reg_strength <= 0:
            raise ValueError(
                f"reg_strength must be > 0; got {self.reg_strength}."
            )
        if self.centroid_radius <= 0:
            raise ValueError(
                f"centroid_radius must be > 0; got {self.centroid_radius}."
            )

        # ── Random state ──────────────────────────────────────────────
        rng = np.random.RandomState(self.random_state)

        # ── ỹ construction (§4 step 4) ────────────────────────────────
        y_tilde = np.concatenate([
            np.ones(k, dtype=float),    # P: +1
            -np.ones(n_U, dtype=float),  # U: -1
        ])

        # ── MoM centroid (§4 step 5) ─────────────────────────────────
        # Paper: corrupted set = {ỹ_i · x_i | ỹ_i = -1} = {-x_i}
        m_hat = _mom_centroid(-X_U, self.mom_groups, rng)
        self.centroid_hat_ = m_hat.copy()

        # ── Centroid covariance (§4 step 5) ──────────────────────────
        S_raw = _centroid_covariance(X_U)
        self.centroid_covariance_raw_ = S_raw

        S_solve = (
            S_raw + self.covariance_ridge * np.eye(d)
            if self.covariance_ridge > 0
            else S_raw
        )

        # ── Kernel (§4 step 6) ───────────────────────────────────────
        sigma_val = self.sigma
        if sigma_val == "scale":
            sigma_val = 1.0 / np.sqrt(d)
        sigma_val = float(sigma_val)

        K = _rbf_kernel(X, X, sigma_val)

        # ── Constants (§4 step 7) ────────────────────────────────────
        C_alpha = 1.0 / n
        C_gamma = 1.0 / (2.0 * n)
        c = -(n - k) / (2.0 * n)
        C_eq = c / denom  # = -(n-k) / (2·n·(1−2·p·h))
        self.C_eq_ = C_eq

        # ── Initial feasible point (§4 step 8) ───────────────────────
        N_total = n + n_U
        Aeq_init = np.zeros((1, N_total), dtype=float)
        Aeq_init[0, :n] = y_tilde
        Aeq_init[0, n:] = -y_tilde[k:]
        lb_init = np.zeros(N_total, dtype=float)
        ub_init = np.concatenate([
            np.full(n, C_alpha),
            np.full(n_U, C_gamma),
        ])

        z0 = _find_feasible_init(Aeq_init, C_eq, lb_init, ub_init)
        mu = m_hat.copy()

        # ── ACS outer loop (§4 step 8a–8h) ────────────────────────────
        acs_history: list[dict] = []
        converged = False
        prev_obj: float | None = None
        z = z0.copy()

        for t in range(self.max_acs_iter):
            # (a) Fixed μ: build and solve QP
            Q, d_vec, Aeq, beq, lb, ub = _build_dual_qp(
                mu, X, K, y_tilde, self.reg_strength, sigma_val,
                n, k, C_eq,
            )

            z, qp_diag = _solve_qp_oracle(
                Q, d_vec, Aeq, beq, lb, ub, z,
                tol=self.tol,
            )
            dual_obj = qp_diag["dual_obj"]

            # (b) Record diagnostics
            iter_info: dict = {
                "iter": t,
                "dual_obj": dual_obj,
                "eq_residual": qp_diag["eq_residual"],
                "box_violation": qp_diag["box_violation"],
                "centroid_constraint_residual": 0.0,
                "degenerate_centroid_step": False,
            }

            # (c) Extract α, γ
            alpha = z[:n]
            gamma = z[n:]

            # (d) Compute Δ from α, γ
            delta = _rbf_centroid_delta(
                alpha, gamma, X, y_tilde,
                self.reg_strength, sigma_val, k,
            )

            # (e) Update μ via Appendix Eq. 35
            mu_new, cent_info = _update_centroid(
                m_hat, S_raw, S_solve, delta,
                self.centroid_radius, self.tol,
            )

            iter_info["centroid_constraint_residual"] = cent_info["constraint_residual"]
            iter_info["degenerate_centroid_step"] = cent_info["degenerate_centroid_step"]

            # (f) Bias recovery
            b0, bias_info = _recover_bias_from_kkt(
                alpha, gamma, X, K, y_tilde, mu_new,
                self.reg_strength, sigma_val, C_eq,
                C_alpha, C_gamma, k,
            )

            # ── Convergence check ────────────────────────────────────
            mu_change = float(np.linalg.norm(mu_new - mu))
            max_kkt = qp_diag["kkt_residual"]
            rel_obj_change = 0.0
            if prev_obj is not None and abs(prev_obj) > 1e-15:
                rel_obj_change = abs(dual_obj - prev_obj) / abs(prev_obj)

            mu = mu_new
            acs_history.append(iter_info)

            if max(rel_obj_change, mu_change, max_kkt) < self.tol:
                converged = True
                break

            prev_obj = dual_obj

        # ── Store fitted attributes (§4 step 9) ───────────────────────
        self.alpha_full_ = alpha
        self.gamma_unlabeled_ = gamma
        self.unlabeled_indices_ = np.where(~mask_P)[0]
        self.support_indices_ = np.where(
            (abs(alpha) > 1e-12) | (
                np.concatenate([np.zeros(k), abs(gamma)]) > 1e-12
            )
        )[0]
        self.bias_ = b0
        self.centroid_opt_ = mu
        self.n_acs_iter_ = len(acs_history)
        self.acs_history_ = acs_history
        self.converged_ = converged

        # ── Finalise (§4 step 10) ─────────────────────────────────────
        self._X_shape_ = X.shape
        self._X_train = X
        self._sigma_val_ = sigma_val
        self._K_ = K
        self._y_tilde_ = y_tilde
        self.classes_ = np.array([0, 1])
        self._is_fitted = True

        if not converged:
            warnings.warn(
                f"KLDCE did not converge within {self.max_acs_iter} "
                f"ACS iterations. Results may be suboptimal.",
                UserWarning,
                stacklevel=2,
            )

        return self

    # ── Decision function / predict ──────────────────────────────────

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision scores f(x) (Appendix Eq. 25).

        .. math::

            f(x) = \\frac{1}{2\\lambda}\\Big[
                \\sum_i \\alpha_i \\tilde{y}_i K(x, x_i)
                - \\sum_{j=1}^{n_U} \\gamma_j \\tilde{y}_{k+j} K(x, x_{k+j})
                - C_{\\text{eq}} \\cdot K(x, \\mu)
            \\Big] + b_0
        """
        self._check_is_fitted()
        K_test = _rbf_kernel(X, self._X_train, self._sigma_val_)

        alpha_y = self.alpha_full_ * self._y_tilde_  # (n,)
        g_alpha = K_test @ alpha_y  # (m,)

        k = len(self.alpha_full_) - len(self.gamma_unlabeled_)
        gamma_y = self.gamma_unlabeled_ * self._y_tilde_[k:]  # (n_U,)
        g_gamma = K_test[:, k:] @ gamma_y  # (m,)

        K_mu = _rbf_kernel(X, self.centroid_opt_.reshape(1, -1), self._sigma_val_).ravel()
        g_centroid = self.C_eq_ * K_mu  # (m,)

        return (g_alpha - g_gamma - g_centroid) / (2.0 * self.reg_strength) + self.bias_

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Binary labels: 1 if f(x) ≥ 0, else 0."""
        return (self._decision_function(X) >= 0.0).astype(int)

    # ── Metadata ─────────────────────────────────────────────────────

    def get_pu_metadata(self) -> dict:
        """Return PU metadata including KLDCE-specific diagnostics."""
        meta = super().get_pu_metadata()
        meta.update({
            "flip_probability": getattr(self, "flip_probability_", None),
            "class_prior": getattr(self, "class_prior_", None),
            "centroid_radius": self.centroid_radius,
            "reg_strength": self.reg_strength,
            "n_acs_iter": getattr(self, "n_acs_iter_", None),
            "converged": getattr(self, "converged_", False),
            "bias": getattr(self, "bias_", None),
        })
        return meta
