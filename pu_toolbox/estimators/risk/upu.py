# ruff: noqa: N803, N806, S101

"""Unbiased PU (uPU) classifier — convex PU learning.

Implements the convex formulation from du Plessis, Niu, Sugiyama
(ICML 2015) with three loss variants:

* **C-DH** (double hinge) — QP-based, main variant.
* **C-LL** (logistic) — smooth convex, L-BFGS.
* **Squared** — closed-form linear solve.

Reference
---------
du Plessis, M. C., Niu, G., & Sugiyama, M.
"Convex Formulation for Learning from Positive and Unlabeled Data."
ICML, 2015.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize

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
from ...losses.upu import _sigmoid, _softplus_stable
from ...utils.basis import (
    build_linear_basis,
    build_rbf_basis,
    subsample_centers,
)

# ── Backwards-compatible private aliases (used internally) ──────────
_build_linear_basis = build_linear_basis
_build_rbf_basis = build_rbf_basis
_subsample_centers = subsample_centers


# ═════════════════════════════════════════════════════════════════════
# PU risk / score helpers
# ═════════════════════════════════════════════════════════════════════


def _pu_validation_risk(
    scores_P: np.ndarray,
    scores_U: np.ndarray,
    class_prior: float,
) -> float:
    """Compute PU zero-one validation risk (paper Eq. 2 / method card §4.6).

    R = 2π · frac(g(x_P) ≤ 0) + frac(g(x_U) > 0) − π

    Note: this can be *negative* in finite samples; do NOT clip before
    comparing across hyper-parameter candidates.
    """
    n_P = len(scores_P)
    n_U = len(scores_U)
    if n_P == 0 or n_U == 0:
        return np.inf
    f_n = float(np.mean(scores_P <= 0.0))  # P classified as negative
    f_pu = float(np.mean(scores_U > 0.0))  # U classified as positive
    return 2.0 * class_prior * f_n + f_pu - class_prior


# ═════════════════════════════════════════════════════════════════════
# Optimisation helpers
# ═════════════════════════════════════════════════════════════════════


def _pack_theta(alpha: np.ndarray, b: float, has_b: bool) -> np.ndarray:
    """Pack (α, b) into a single parameter vector for the optimiser."""
    return np.append(alpha, b) if has_b else alpha.copy()


def _unpack_theta(
    theta: np.ndarray,
    n_basis: int,
    has_b: bool,
) -> tuple[np.ndarray, float]:
    """Unpack optimiser vector into (α, b)."""
    if has_b:
        return theta[:n_basis], float(theta[-1])
    return theta, 0.0


# ═════════════════════════════════════════════════════════════════════
# UPUClassifier
# ═════════════════════════════════════════════════════════════════════


class UPUClassifier(BasePUClassifier):
    """Unbiased PU classifier with convex risk (uPU / Convex PU).

    Fits a linear-in-parameters model

        g(x) = αᵀ φ(x) + b

    by minimising the convex PU risk

        J(α, b) = −(π/n_P) Σ_P g(x) + (1/n_U) Σ_U ℓ(−g(x)) + (λ/2)‖α‖²

    where the margin loss ℓ is chosen so that ℓ̃(z) = ℓ(z)−ℓ(−z) = −z,
    guaranteeing convexity of the overall objective.

    Parameters
    ----------
    class_prior : float
        Class prior π = P(y=1).  Required.  May be overridden in
        :meth:`fit` via the ``class_prior`` kwarg.
    loss : {"double_hinge", "logistic", "squared"}, default "double_hinge"
        * ``"double_hinge"`` — ℓ(z) = max{−z, 0, (1−z)/2}.  Convex QP.
          Recommended primary variant (C-DH).
        * ``"logistic"`` — ℓ(z) = log(1+exp(−z)).  Smooth convex, L-BFGS.
          Alternative when QP solver is unavailable (C-LL).
        * ``"squared"`` — ℓ(z) = ¼(z−1)².  Closed-form linear solve.
          Fastest but penalises correct large-margin predictions.
    reg_lambda : float, default 1e-3
        ℓ₂ regularisation coefficient for α.  Must be > 0.  The intercept
        *b* is never regularised (paper convention).
    basis : {"linear", "rbf"}, default "linear"
        Basis function type.  ``"linear"`` → φ(x) = x.  ``"rbf"`` →
        Gaussian kernel with *n_centers* subsampled from unlabeled data.
    kernel_width : float or None, default None
        Gaussian kernel width σ.  Required when ``basis="rbf"``.
    n_centers : int or None, default None
        Number of RBF centers to subsample from unlabeled data.
        Default: ``min(200, n_U)``.  Ignored for ``basis="linear"``.
    fit_intercept : bool, default True
        Whether to fit the intercept *b*.  Setting to ``False`` yields
        the simplified model g(x) = αᵀ φ(x).
    max_iter : int, default 1000
        Maximum iterations for the optimiser (L-BFGS / SLSQP).
    tol : float, default 1e-6
        Convergence tolerance for the optimiser.
    random_state : int or None, default None
        Random seed for center subsampling.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_basis,)
        Fitted basis coefficients α.
    intercept_ : float
        Fitted intercept b (0 when ``fit_intercept=False``).
    n_iter_ : int
        Number of optimiser iterations performed.
    opt_status_ : int
        Optimiser exit status (0 = success).  See :mod:`scipy.optimize`.
    opt_message_ : str
        Human-readable optimiser termination message.
    """

    # ── Class-level metadata ──────────────────────────────────────────
    family: AlgorithmFamily = AlgorithmFamily.RISK_ESTIMATION
    assumption: tuple[Assumption, ...] = (Assumption.SCAR,)
    scenario: tuple[Scenario, ...] = (Scenario.CASE_CONTROL,)
    requires_class_prior: bool = True
    implementation_status: ImplementationStatus = ImplementationStatus.NATIVE
    source_status: SourceStatus = SourceStatus.OFFICIAL_BUNDLE
    backend: Backend = Backend.NUMPY
    maturity: Maturity = Maturity.STABLE

    def __init__(
        self,
        class_prior: float,
        *,
        loss: Literal["double_hinge", "logistic", "squared"] = "double_hinge",
        reg_lambda: float = 1e-3,
        basis: Literal["linear", "rbf"] = "linear",
        kernel_width: float | None = None,
        n_centers: int | None = None,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.class_prior = class_prior
        self.loss = loss
        self.reg_lambda = reg_lambda
        self.basis = basis
        self.kernel_width = kernel_width
        self.n_centers = n_centers
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
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
    ) -> UPUClassifier:
        """Fit the uPU classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y_pu : np.ndarray of shape (n_samples,)
            PU labels.  +1 = labeled positive, 0 = unlabeled.
        class_prior : float, optional
            Override the constructor's ``class_prior``.  Must be in (0, 1).
        sample_weight : np.ndarray, optional
            Accepted for API compatibility; currently ignored.

        Returns
        -------
        self : UPUClassifier
        """
        X, y_pu = validate_pu_X_y(X, y_pu, estimator_name="UPUClassifier")

        # ── Input sanity ─────────────────────────────────────────────
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values.")
        if not np.isfinite(y_pu).all():
            raise ValueError("y_pu contains NaN or Inf values.")

        # ── Resolve class_prior ───────────────────────────────────────
        pi = class_prior if class_prior is not None else self.class_prior
        if not (0.0 < pi < 1.0):
            raise ValueError(f"class_prior must be in (0, 1); got {pi}.")
        self._class_prior = pi
        self.class_prior_ = pi

        # ── Parameter validation ─────────────────────────────────────
        if self.reg_lambda <= 0:
            raise ValueError(f"reg_lambda must be > 0; got {self.reg_lambda}.")
        if self.loss not in ("double_hinge", "logistic", "squared"):
            raise ValueError(f"Unknown loss {self.loss!r}.")

        # ── Split P / U ──────────────────────────────────────────────
        mask_P = y_pu == 1
        X_P = X[mask_P]
        X_U = X[~mask_P]
        n_P = X_P.shape[0]
        n_U = X_U.shape[0]
        d = X.shape[1]

        if n_P == 0:
            raise ValueError("Need at least 1 labeled positive sample.")
        if n_U == 0:
            raise ValueError("Need at least 1 unlabeled sample.")

        rng = np.random.RandomState(self.random_state)

        # ── Build basis ───────────────────────────────────────────────
        if self.basis == "linear":
            _phi = _build_linear_basis
            n_basis = d
            centers = None
        elif self.basis == "rbf":
            if self.kernel_width is None or self.kernel_width <= 0:
                raise ValueError(
                    f"kernel_width must be > 0 for basis='rbf'; got {self.kernel_width}."
                )
            n_centers_val = self.n_centers if self.n_centers is not None else min(200, n_U)
            centers = _subsample_centers(X_U, n_centers_val, rng)
            n_basis = centers.shape[0]
            kw = self.kernel_width

            def _phi(X_in: np.ndarray) -> np.ndarray:
                return _build_rbf_basis(X_in, centers, kw)
        else:
            raise ValueError(f"Unknown basis {self.basis!r}.")

        Phi_P = _phi(X_P)  # (n_P, n_basis)
        Phi_U = _phi(X_U)  # (n_U, n_basis)

        self._n_basis_ = n_basis
        self._centers_ = centers
        self._kw_ = self.kernel_width

        # ── Solve ─────────────────────────────────────────────────────
        loss_fn = self.loss
        if loss_fn == "squared":
            self._fit_squared(Phi_P, Phi_U, n_P, n_U, pi)
        elif loss_fn == "logistic":
            self._fit_logistic(Phi_P, Phi_U, n_P, n_U, pi)
        else:  # double_hinge
            self._fit_double_hinge(Phi_P, Phi_U, n_P, n_U, pi)

        # ── Finalise ──────────────────────────────────────────────────
        self._X_shape_ = X.shape
        self.classes_ = np.array([0, 1])
        self._is_fitted = True
        return self

    # ── Squared loss (closed form) ───────────────────────────────────

    def _fit_squared(
        self,
        Phi_P: np.ndarray,
        Phi_U: np.ndarray,
        n_P: int,
        n_U: int,
        pi: float,
    ) -> None:
        """Squared-loss closed form (no intercept, per paper)."""
        if self.fit_intercept:
            warnings.warn(
                "Squared loss closed form does not support an intercept; "
                "fit_intercept is ignored. Set fit_intercept=False to "
                "suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
        n_basis = Phi_P.shape[1]
        lam = self.reg_lambda

        # H = (1/(2 n_U)) Φ_Uᵀ Φ_U + λ I
        H = (Phi_U.T @ Phi_U) / (2.0 * n_U) + lam * np.eye(n_basis)
        # h = (π/n_P) Φ_Pᵀ 1 − (1/(2 n_U)) Φ_Uᵀ 1
        h = (pi / n_P) * Phi_P.sum(axis=0) - (1.0 / (2.0 * n_U)) * Phi_U.sum(axis=0)

        self.coef_ = solve(H, h, assume_a="pos")
        self.intercept_ = 0.0
        self.n_iter_ = 0
        self.opt_status_ = 0
        self.opt_message_ = "Closed-form solution."

    # ── Logistic loss (L-BFGS) ──────────────────────────────────────

    def _fit_logistic(
        self,
        Phi_P: np.ndarray,
        Phi_U: np.ndarray,
        n_P: int,
        n_U: int,
        pi: float,
    ) -> None:
        """L-BFGS minimisation of the smooth C-LL objective."""
        lam = self.reg_lambda
        has_b = self.fit_intercept

        n_basis = Phi_P.shape[1]

        # sum over P: cached once
        sum_Phi_P = Phi_P.sum(axis=0)  # (n_basis,)

        def objective(theta: np.ndarray) -> float:
            alpha, b = _unpack_theta(theta, n_basis, has_b)
            g_U = Phi_U @ alpha + b  # (n_U,)
            pos_term = -(pi / n_P) * (alpha @ sum_Phi_P + n_P * b)
            unlabeled_term = float(np.mean(_softplus_stable(g_U)))
            reg_term = 0.5 * lam * (alpha @ alpha)
            return pos_term + unlabeled_term + reg_term

        def gradient(theta: np.ndarray) -> np.ndarray:
            alpha, b = _unpack_theta(theta, n_basis, has_b)
            g_U = Phi_U @ alpha + b
            sigma_U = _sigmoid(g_U)  # (n_U,)

            grad_alpha = (
                -(pi / n_P) * sum_Phi_P
                + (1.0 / n_U) * (Phi_U.T @ sigma_U)
                + lam * alpha
            )
            grad_b = -pi + float(np.mean(sigma_U))

            return _pack_theta(grad_alpha, grad_b, has_b)

        theta0 = _pack_theta(np.zeros(n_basis), 0.0, has_b)

        res = minimize(
            objective,
            theta0,
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )

        self.coef_, self.intercept_ = _unpack_theta(res.x, n_basis, has_b)
        self.n_iter_ = res.nit if hasattr(res, "nit") else res.nfev
        self.opt_status_ = res.status
        self.opt_message_ = (
            res.message.decode() if isinstance(res.message, bytes) else str(res.message)
        )

        if not res.success:
            warnings.warn(
                f"L-BFGS solver did not converge: {self.opt_message_} "
                f"(status={self.opt_status_}). Results may be suboptimal.",
                UserWarning,
                stacklevel=2,
            )

    # ── Double hinge loss (SLSQP) ────────────────────────────────────

    def _fit_double_hinge(
        self,
        Phi_P: np.ndarray,
        Phi_U: np.ndarray,
        n_P: int,
        n_U: int,
        pi: float,
    ) -> None:
        """Minimise the C-DH objective via constrained QP (SLSQP).

        The problem is equivalent to:

            min_{α,b}  −(π/n_P) Σ_P g(x_P) − π b
                      + (1/n_U) Σ_U max{0, g(x_U), (1+g(x_U))/2}
                      + (λ/2) ‖α‖²

        where g(x) = αᵀ φ(x) + b.  Although non-smooth, the convexity
        guarantees that SLSQP (which handles inequality constraints
        natively) converges reliably in practice.
        """
        lam = self.reg_lambda
        has_b = self.fit_intercept
        n_basis = Phi_P.shape[1]

        sum_Phi_P = Phi_P.sum(axis=0)  # (n_basis,)

        def objective(theta: np.ndarray) -> float:
            alpha, b = _unpack_theta(theta, n_basis, has_b)
            g_U = Phi_U @ alpha + b

            # double hinge: max{0, g, (1+g)/2}
            dh_u = np.maximum(np.maximum(0.0, g_U), 0.5 + 0.5 * g_U)

            pos_term = -(pi / n_P) * (alpha @ sum_Phi_P + n_P * b)
            unlabeled_term = float(np.mean(dh_u))
            reg_term = 0.5 * lam * (alpha @ alpha)
            return pos_term + unlabeled_term + reg_term

        theta0 = _pack_theta(np.zeros(n_basis), 0.0, has_b)

        res = minimize(
            objective,
            theta0,
            method="SLSQP",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        self.coef_, self.intercept_ = _unpack_theta(res.x, n_basis, has_b)
        self.n_iter_ = res.nit if hasattr(res, "nit") else res.nfev
        self.opt_status_ = res.status
        self.opt_message_ = (
            res.message.decode() if isinstance(res.message, bytes) else str(res.message)
        )

        if not res.success:
            warnings.warn(
                f"QP solver did not converge: {self.opt_message_} "
                f"(status={self.opt_status_}). Results may be suboptimal.",
                UserWarning,
                stacklevel=2,
            )

    # ── Decision function / predict ──────────────────────────────────

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """g(x) = αᵀ φ(x) + b."""
        self._check_is_fitted()
        Phi = (
            _build_linear_basis(X)
            if self.basis == "linear"
            else _build_rbf_basis(X, self._centers_, self._kw_)
        )
        return Phi @ self.coef_ + self.intercept_

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Binary labels: 1 if g(x) >= 0 else 0."""
        return (self._decision_function(X) >= 0.0).astype(int)

    # ── PU validation risk ───────────────────────────────────────────

    def pu_validation_risk(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
    ) -> float:
        """Compute PU zero-one validation risk (paper Eq. 2).

        Used for hyper-parameter selection (PU-CV).  Low values
        (possibly negative) indicate better models.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y_pu : np.ndarray of shape (n_samples,)
            PU labels in {+1, 0} format.

        Returns
        -------
        float
        """
        from ...core.labels import normalize_pu_labels

        y_pu = normalize_pu_labels(y_pu)
        scores = self._decision_function(X)
        mask_P = y_pu == 1
        return _pu_validation_risk(scores[mask_P], scores[~mask_P], self._class_prior)

    # ── Metadata ─────────────────────────────────────────────────────

    def get_pu_metadata(self) -> dict:
        """Return PU metadata including solver diagnostics."""
        meta = super().get_pu_metadata()
        meta.update(
            {
                "loss": self.loss,
                "reg_lambda": self.reg_lambda,
                "basis": self.basis,
                "fit_intercept": self.fit_intercept,
                "n_basis": getattr(self, "_n_basis_", None),
                "n_iter": getattr(self, "n_iter_", None),
                "opt_status": getattr(self, "opt_status_", None),
            }
        )
        return meta
