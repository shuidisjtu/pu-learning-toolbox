# ruff: noqa: N803, N806

"""PNU (Positive-Negative-Unlabeled) classifier — closed-form solution.

Implements the convex PNU formulation from Sakai, du Plessis, Niu,
Sugiyama (ICML 2017) with the squared loss, yielding a closed-form
linear solve analogous to :class:`UPUClassifier`'s squared variant.

Reference
---------
Sakai, T., du Plessis, M. C., Niu, G., & Sugiyama, M.
"Semi-Supervised Classification Based on Classification from
Positive and Unlabeled Data."  ICML / PMLR 70, 2017.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
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
from ...core.validation import validate_pnu_X_y
from ...losses.pnu import (
    _compute_nu_risk_squared,
    _compute_pn_risk,
    _compute_pnu_risk,
    _compute_pu_risk_squared,
)
from ...utils.basis import build_linear_basis, build_rbf_basis, subsample_centers

# ═════════════════════════════════════════════════════════════════════
# PNUClassifier
# ═════════════════════════════════════════════════════════════════════


class PNUClassifier(BasePUClassifier):
    """PNU semi-supervised classifier (Sakai et al., ICML 2017).

    Fits a linear-in-parameters model

        g(x) = alpha^T phi(x) + b

    by minimising the convex PNU risk with squared loss in closed form.
    The formulation is mathematically equivalent to :class:`pywsl`'s
    ``PNU_SL`` class.

    Parameters
    ----------
    class_prior : float
        Class prior theta_P = P(y=1).  **Required** — must be in (0, 1).
    eta : float, default 0.0
        PNU trade-off parameter in [-1, 1].  ``eta=0`` is PN (supervised),
        ``eta=+1`` is PU, ``eta=-1`` is NU.
    reg_lambda : float, default 1e-3
        ell_2 regularisation coefficient for alpha.  Must be > 0.  The
        intercept *b* is never regularised (paper convention).
    basis : {"linear", "rbf"}, default "linear"
        Basis function type.  ``"linear"`` → phi(x) = x.  ``"rbf"`` →
        Gaussian kernel with *n_centers* subsampled from unlabeled data.
    kernel_width : float or None, default None
        Gaussian kernel width sigma.  Required when ``basis="rbf"``.
    n_centers : int or None, default None
        Number of RBF centers to subsample from unlabeled data.
        Default: ``min(200, n_U)``.  Ignored for ``basis="linear"``.
    fit_intercept : bool, default True
        Whether to fit the intercept *b*.  The intercept is implemented
        by augmenting the basis with a constant column of ones whose
        coefficient is never regularised (matches pywsl convention).
    random_state : int or None, default None
        Random seed for center subsampling.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_basis,)
        Fitted basis coefficients alpha (including intercept if enabled).
    intercept_ : float
        Fitted intercept b (0 when ``fit_intercept=False``).
    class_prior_ : float
        Class prior used during training.
    eta_ : float
        PNU trade-off parameter used during training.
    risk_components_ : dict
        Component risks after fitting: ``pn_risk``, ``pu_risk``,
        ``nu_risk``, ``pnu_risk``.
    n_positive_ : int
        Number of positive samples.
    n_negative_ : int
        Number of negative samples.
    n_unlabeled_ : int
        Number of unlabeled samples.
    """

    # ── Class-level metadata ──────────────────────────────────────────
    family: AlgorithmFamily = AlgorithmFamily.RISK_ESTIMATION
    assumption: tuple[Assumption, ...] = (Assumption.SCAR,)
    scenario: tuple[Scenario, ...] = (Scenario.CASE_CONTROL,)
    requires_class_prior: bool = True
    implementation_status: ImplementationStatus = ImplementationStatus.NATIVE
    source_status: SourceStatus = SourceStatus.OFFICIAL_EXACT
    backend: Backend = Backend.NUMPY
    maturity: Maturity = Maturity.RESEARCH

    def __init__(
        self,
        class_prior: float,
        *,
        eta: float = 0.0,
        reg_lambda: float = 1e-3,
        basis: Literal["linear", "rbf"] = "linear",
        kernel_width: float | None = None,
        n_centers: int | None = None,
        fit_intercept: bool = True,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.class_prior = class_prior
        self.eta = eta
        self.reg_lambda = reg_lambda
        self.basis = basis
        self.kernel_width = kernel_width
        self.n_centers = n_centers
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    # ── fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> PNUClassifier:
        """Fit the PNU classifier via closed-form linear solve.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            P/N/U labels in {+1, -1, 0} format.
        class_prior : float, optional
            Override the constructor's ``class_prior``.  Must be in (0, 1).
        sample_weight : np.ndarray, optional
            Accepted for API compatibility; currently ignored.

        Returns
        -------
        self : PNUClassifier
        """
        # ── Validate ────────────────────────────────────────────────
        X, y = validate_pnu_X_y(
            X, y, estimator_name="PNUClassifier"
        )
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values.")

        # ── Resolve parameters ──────────────────────────────────────
        pi = class_prior if class_prior is not None else self.class_prior
        if not (0.0 < pi < 1.0):
            raise ValueError(
                f"class_prior must be in (0, 1); got {pi}."
            )
        if not (-1.0 <= self.eta <= 1.0):
            raise ValueError(
                f"eta must be in [-1, 1]; got {self.eta}."
            )
        if self.reg_lambda <= 0:
            raise ValueError(
                f"reg_lambda must be > 0; got {self.reg_lambda}."
            )

        # ── Split P / N / U ─────────────────────────────────────────
        mask_P = y == 1
        mask_N = y == -1
        mask_U = y == 0
        X_P = X[mask_P]
        X_N = X[mask_N]
        X_U = X[mask_U]
        n_P, n_N, n_U = X_P.shape[0], X_N.shape[0], X_U.shape[0]
        d = X.shape[1]

        self._class_prior = pi
        self.class_prior_ = pi
        self.eta_ = self.eta
        self.n_positive_ = n_P
        self.n_negative_ = n_N
        self.n_unlabeled_ = n_U

        rng = np.random.RandomState(self.random_state)

        # ── Build basis ─────────────────────────────────────────────
        if self.basis == "linear":
            _phi = build_linear_basis
            n_basis = d
            centers = None
        elif self.basis == "rbf":
            if self.kernel_width is None or self.kernel_width <= 0:
                raise ValueError(
                    f"kernel_width must be > 0 for basis='rbf'; "
                    f"got {self.kernel_width}."
                )
            n_centers_val = (
                self.n_centers
                if self.n_centers is not None
                else min(200, n_U)
            )
            centers = subsample_centers(X_U, n_centers_val, rng)
            n_basis = centers.shape[0]
            kw = self.kernel_width

            def _phi(X_in: np.ndarray) -> np.ndarray:
                return build_rbf_basis(X_in, centers, kw)
        else:
            raise ValueError(f"Unknown basis {self.basis!r}.")

        Phi_P = _phi(X_P)  # (n_P, n_basis)
        Phi_N = _phi(X_N)  # (n_N, n_basis)
        Phi_U = _phi(X_U)  # (n_U, n_basis)

        # ── Augment with intercept column ───────────────────────────
        has_b = self.fit_intercept
        if has_b:
            Phi_P = np.column_stack([Phi_P, np.ones(n_P)])
            Phi_N = np.column_stack([Phi_N, np.ones(n_N)])
            Phi_U = np.column_stack([Phi_U, np.ones(n_U)])
            n_basis += 1

        self._n_basis_ = n_basis
        self._centers_ = centers
        self._kw_ = self.kernel_width

        # ── Build linear system (matching pywsl PNU_SL math) ───────
        theta_P = pi
        theta_N = 1.0 - pi

        # Hessian blocks
        H_p = (Phi_P.T @ Phi_P) / n_P  # (n_basis, n_basis)
        H_n = (Phi_N.T @ Phi_N) / n_N
        H_u = (Phi_U.T @ Phi_U) / n_U
        H_pn = theta_P * H_p + theta_N * H_n

        # Gradient blocks (mean of each basis column)
        h_p = theta_P * np.mean(Phi_P, axis=0)  # (n_basis,)
        h_n = theta_N * np.mean(Phi_N, axis=0)
        h_u = np.mean(Phi_U, axis=0)
        h_pn = h_p - h_n

        # Regularisation matrix (don't regularise intercept)
        Reg = self.reg_lambda * np.eye(n_basis)
        if has_b:
            Reg[-1, -1] = 0.0

        # Branch on eta
        if self.eta >= 0.0:
            gamma = self.eta
            h_xu = 2.0 * h_p - h_u  # PN+PU branch
        else:
            gamma = -self.eta
            h_xu = h_u - 2.0 * h_n  # PN+NU branch

        H_total = (1.0 - gamma) * H_pn + gamma * H_u + Reg
        h_total = (1.0 - gamma) * h_pn + gamma * h_xu

        # ── Solve ───────────────────────────────────────────────────
        theta_vec = solve(H_total, h_total, assume_a="pos")

        if has_b:
            self.coef_ = theta_vec[:-1]
            self.intercept_ = float(theta_vec[-1])
        else:
            self.coef_ = theta_vec
            self.intercept_ = 0.0

        # ── Compute risk components (diagnostics) ───────────────────
        scores_P = Phi_P @ theta_vec
        scores_N = Phi_N @ theta_vec
        scores_U = Phi_U @ theta_vec

        self.risk_components_ = {
            "pn_risk": _compute_pn_risk(scores_P, scores_N, pi),
            "pu_risk": _compute_pu_risk_squared(scores_P, scores_U, pi),
            "nu_risk": _compute_nu_risk_squared(scores_N, scores_U, pi),
            "pnu_risk": _compute_pnu_risk(
                scores_P, scores_N, scores_U, class_prior=pi, eta=self.eta
            ),
        }

        # ── Finalise ────────────────────────────────────────────────
        self._X_shape_ = X.shape
        self.classes_ = np.array([0, 1])
        self._is_fitted = True
        return self

    # ── Decision function / predict ─────────────────────────────────

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """g(x) = alpha^T phi(x) + b."""
        self._check_is_fitted()

        if self.basis == "linear":
            Phi = build_linear_basis(X)
        else:
            Phi = build_rbf_basis(X, self._centers_, self._kw_)

        return Phi @ self.coef_ + self.intercept_

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Binary labels: 1 if g(x) >= 0 else 0."""
        return (self._decision_function(X) >= 0.0).astype(int)

    # ── Metadata ────────────────────────────────────────────────────

    def get_pu_metadata(self) -> dict:
        """Return PU metadata including PNU-specific diagnostics."""
        meta = super().get_pu_metadata()
        meta.update(
            {
                "eta": self.eta_,
                "reg_lambda": self.reg_lambda,
                "basis": self.basis,
                "fit_intercept": self.fit_intercept,
                "n_basis": getattr(self, "_n_basis_", None),
                "n_positive": getattr(self, "n_positive_", None),
                "n_negative": getattr(self, "n_negative_", None),
                "n_unlabeled": getattr(self, "n_unlabeled_", None),
                "risk_components": getattr(
                    self, "risk_components_", None
                ),
            }
        )
        return meta
