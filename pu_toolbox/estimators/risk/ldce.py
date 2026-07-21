# ruff: noqa: N803, N806, S101

"""LDCE: Loss Decomposition and Centroid Estimation for PU Learning.

Implements the linear LDCE classifier from:

    Gong, C., Shi, H., Liu, T., Zhang, C., Yang, J., & Tao, D.
    "Loss Decomposition and Centroid Estimation for Positive and
    Unlabeled Learning."  IEEE TPAMI, 2021.

The method treats unlabeled data as a corrupted negative set with
one-sided label noise (censoring PU) and estimates the true centroid
of the unlabeled set via median-of-means (MoM) with an ellipsoid
constraint.  Optimization alternates between a closed-form centroid
update and subgradient descent on the linear discriminant *w*.

This is a clean-room implementation based on the paper's Algorithm 1
(MoM centroid) and Algorithm 2 (alternating optimisation).  The
official code archive (CEGE_PAMI20.rar) implements an earlier
conference version with bilateral noise and no ellipsoid constraint;
it is *not* the basis for this module.
"""

from __future__ import annotations

import warnings

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
from ...core.validation import validate_pu_X_y

# ═════════════════════════════════════════════════════════════════════
# Private algorithm helpers
# ═════════════════════════════════════════════════════════════════════


def _mom_centroid(
    X_U: np.ndarray,
    g: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Median-of-means centroid estimate (Algorithm 1).

    Parameters
    ----------
    X_U : np.ndarray of shape (n_U, d)
        Unlabeled samples (corrupted negative set S̃_N).
    g : int
        Number of groups.  ``g=1`` degenerates to the ordinary mean.
    rng : np.random.RandomState
        Seeded random state for reproducible shuffling.

    Returns
    -------
    m_hat : np.ndarray of shape (d,)
        MoM centroid estimate.

    Raises
    ------
    ValueError
        If *g* exceeds the number of unlabeled samples.
    """
    n_U = X_U.shape[0]

    if g == 1:
        return X_U.mean(axis=0)

    if g > n_U:
        raise ValueError(
            f"mom_groups ({g}) cannot exceed the number of unlabeled "
            f"samples ({n_U})."
        )

    # Shuffle and split into g approximately equal groups
    indices = rng.permutation(n_U)
    groups = np.array_split(indices, g)
    means = np.array([X_U[grp].mean(axis=0) for grp in groups])  # (g, d)

    # Pairwise L2 distances between group means
    diffs = np.linalg.norm(
        means[:, None, :] - means[None, :, :], axis=-1
    )  # (g, g)

    # r_i = median_{j != i} ||m_i - m_j||
    r = np.array([
        np.median(np.delete(diffs[i], i)) for i in range(g)
    ])

    i_star = int(np.argmin(r))
    return means[i_star]


def _centroid_covariance(
    X_U: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Empirical covariance of the corrupted centroid (Eq. 10).

    .. math::

        \\hat{S} = \\frac{X_U^\\top X_U}{|U|^2}
                 - \\frac{(\\sum X_U)(\\sum X_U)^\\top}{|U|^2}
                 + \\rho I

    where :math:`\\rho` is the ridge penalty.

    Note: because all unlabeled samples have corrupted label
    :math:`\\tilde{y}_i = -1`, the term :math:`(\\sum x_i \\tilde{y}_i)`
    equals :math:`-\\sum x_i`, and its outer product is identical to
    :math:`(\\sum x_i)(\\sum x_i)^\\top`.

    Parameters
    ----------
    X_U : np.ndarray of shape (n_U, d)
        Unlabeled samples.
    ridge : float
        Ridge penalty added to the diagonal for numerical stability.

    Returns
    -------
    S_hat : np.ndarray of shape (d, d)
        Regularised centroid covariance matrix.
    """
    n_U = X_U.shape[0]
    d = X_U.shape[1]
    sum_X = X_U.sum(axis=0)  # (d,)

    S = (X_U.T @ X_U) / (n_U ** 2) - np.outer(sum_X, sum_X) / (n_U ** 2)
    S += ridge * np.eye(d)
    return S


def _update_m(
    m_hat: np.ndarray,
    S_ridge: np.ndarray,
    w: np.ndarray,
    b: float,
) -> np.ndarray:
    """Closed-form centroid update under ellipsoid constraint (Eq. 14).

    .. math::

        m \\leftarrow \\hat{m}
        + \\hat{S}^{-1} w \\sqrt{\\frac{b}{w^\\top \\hat{S}^{-1} w}}

    Uses a linear solve instead of explicit matrix inversion.
    Falls back to *m_hat* when the direction is degenerate.

    Parameters
    ----------
    m_hat : np.ndarray of shape (d,)
        MoM initial centroid.
    S_ridge : np.ndarray of shape (d, d)
        Regularised covariance matrix (must be positive definite).
    w : np.ndarray of shape (d,)
        Current linear discriminant.
    b : float
        Ellipsoid radius.

    Returns
    -------
    m : np.ndarray of shape (d,)
        Updated centroid satisfying the ellipsoid constraint.
    """
    try:
        v = solve(S_ridge, w)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Covariance solve failed in m-update; falling back to m_hat.",
            UserWarning,
            stacklevel=2,
        )
        return m_hat.copy()

    wTv = float(w @ v)
    if wTv < 1e-15:
        warnings.warn(
            "w^T S^{-1} w is near-zero in m-update; falling back to m_hat.",
            UserWarning,
            stacklevel=2,
        )
        return m_hat.copy()

    return m_hat + v * np.sqrt(b / wTv)


def _ldce_objective(
    w: np.ndarray,
    X_P: np.ndarray,
    X_U: np.ndarray,
    m: np.ndarray,
    n: int,
    k: int,
    h: float,
    p: float,
    reg: float,
) -> float:
    """Full LDCE objective value for a fixed *w* and *m*.

    .. math::

        J(w) = \\frac{1}{n}\\sum_{P}\\ell(w^\\top x_i)
             + \\frac{1}{2n}\\sum_{U}\\phi(-w^\\top x_i)
             + \\frac{c}{1-2ph} w^\\top m + \\lambda\\|w\\|^2

    where :math:`\\ell(z)=[1-z]_+`,
    :math:`\\phi(z)=[1-z]_++[1+z]_+`, and
    :math:`c = -(n-k)/(2n)`.
    """
    scores_P = X_P @ w
    scores_U = X_U @ w

    # P term (labeled positives, hinge loss)
    loss_P = np.maximum(0.0, 1.0 - scores_P).sum() / n

    # U term (corrupted negatives, φ(-score) since ỹ=-1)
    phi_U = np.maximum(0.0, 1.0 + scores_U) + np.maximum(0.0, 1.0 - scores_U)
    loss_U = phi_U.sum() / (2.0 * n)

    # Centroid term
    c = -(n - k) / (2.0 * n)
    denom = 1.0 - 2.0 * p * h
    cent_term = c / denom * float(w @ m)

    # Regularisation
    reg_term = reg * float(w @ w)

    return float(loss_P + loss_U + cent_term + reg_term)


def _ldce_subgradient(
    w: np.ndarray,
    X_P: np.ndarray,
    X_U: np.ndarray,
    m: np.ndarray,
    n: int,
    k: int,
    h: float,
    p: float,
    reg: float,
) -> np.ndarray:
    """Subgradient of the LDCE objective w.r.t. *w* (fixed *m*).

    At kink points (where :math:`w^\\top x = \\pm 1`) the subgradient
    contribution is taken as zero — a valid choice for any convex
    hinge-like function.
    """
    scores_P = X_P @ w
    scores_U = X_U @ w
    d = w.shape[0]
    grad = np.zeros(d, dtype=float)

    # ── P term: -(1/n) Σ_{score < 1} x_i ──────────────────────────
    mask_P = scores_P < 1.0
    if mask_P.any():
        grad -= X_P[mask_P].sum(axis=0) / n

    # ── U term: (1/(2n)) [Σ_{score>1} x_i − Σ_{score<-1} x_i] ────
    mask_gt_1 = scores_U > 1.0
    mask_lt_m1 = scores_U < -1.0
    if mask_gt_1.any():
        grad += X_U[mask_gt_1].sum(axis=0) / (2.0 * n)
    if mask_lt_m1.any():
        grad -= X_U[mask_lt_m1].sum(axis=0) / (2.0 * n)

    # ── Centroid term: c/(1-2ph) · m ──────────────────────────────
    c = -(n - k) / (2.0 * n)
    denom = 1.0 - 2.0 * p * h
    grad += c / denom * m

    # ── Regularisation: 2λ w ──────────────────────────────────────
    grad += 2.0 * reg * w

    return grad


def _solve_w_subproblem(
    w: np.ndarray,
    X_P: np.ndarray,
    X_U: np.ndarray,
    m: np.ndarray,
    n: int,
    k: int,
    h: float,
    p: float,
    reg: float,
    outer_iter: int,
    learning_rate: float,
    n_inner_iter: int,
) -> np.ndarray:
    """Minimise the LDCE objective w.r.t. *w* via subgradient descent.

    Uses schedule-based learning rate decay:
    ``lr = learning_rate / sqrt(1 + outer_iter)``.
    """
    lr = learning_rate / np.sqrt(1.0 + outer_iter)
    for _ in range(n_inner_iter):
        g = _ldce_subgradient(w, X_P, X_U, m, n, k, h, p, reg)

        # Gradient clipping for stability.
        # Threshold 1e3 is generous enough for well-conditioned problems
        # while preventing overflow in extreme cases.
        g_norm = float(np.linalg.norm(g))
        if g_norm > 1e3:
            g *= 1e3 / g_norm

        w = w - lr * g

    return w


# ═════════════════════════════════════════════════════════════════════
# LDCEClassifier
# ═════════════════════════════════════════════════════════════════════


class LDCEClassifier(BasePUClassifier):
    """Linear LDCE classifier for censoring PU learning.

    Implements Loss Decomposition and Centroid Estimation (LDCE) from
    Gong et al. (TPAMI 2021).  Treats unlabeled data as a corrupted
    negative set with one-sided label noise (flip probability *h*) and
    recovers the true unlabeled centroid via median-of-means with an
    ellipsoid constraint.

    Parameters
    ----------
    flip_probability : float
        Probability *h* that a true positive is flipped to an observed
        negative (censoring rate).  Must be in ``(0, 1)``.  **Required.**
    reg_strength : float, default 1.0
        L2 regularisation coefficient λ for the linear weights.
    centroid_radius : float, default 1.0
        Ellipsoid radius *b* for the centroid constraint.  Selected via
        cross-validation in the paper.
    mom_groups : int, default 10
        Number of groups *g* for median-of-means centroid estimation.
        ``g=1`` degenerates to the ordinary mean.
    covariance_ridge : float, default 1e-4
        Ridge penalty added to the diagonal of the centroid covariance
        matrix for numerical stability.  1e-4 is chosen over 1e-8 for
        practical robustness on near-singular covariance matrices.
    learning_rate : float, default 0.01
        Initial step size for subgradient descent on *w*.
    n_inner_iter : int, default 50
        Number of gradient descent steps per outer iteration.
    max_iter : int, default 100
        Maximum number of outer alternating-optimisation iterations.
    tol : float, default 1e-6
        Relative change in objective below which convergence is declared.
    random_state : int or None, default None
        Seed for MoM group shuffling and weight initialisation.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Fitted linear discriminant weights *w*.
    class_prior_ : float
        Estimated positive class prior π = k / [n (1−h)].
    flip_probability_ : float
        Validated flip probability used for fitting.
    corrupted_centroid_ : np.ndarray of shape (n_features,)
        MoM initial centroid of the corrupted negative set (m̂).
    true_unlabeled_centroid_ : np.ndarray of shape (n_features,)
        Optimised centroid of the true unlabeled set (m).
    centroid_covariance_ : np.ndarray of shape (n_features, n_features)
        Regularised centroid covariance matrix Ŝ.
    n_labeled_ : int
        Number of labeled positive samples (k).
    n_unlabeled_ : int
        Number of unlabeled samples (n−k).
    n_iter_ : int
        Number of outer iterations actually performed.
    objective_history_ : list of float
        Objective value after each outer iteration.
    converged_ : bool
        Whether the optimisation converged within ``max_iter``.
    classes_ : np.ndarray of shape (2,)
        ``np.array([0, 1])``.

    Notes
    -----
    This method is only applicable to **censoring PU** (single
    i.i.d. training set with one-sided SCAR label noise).  It is *not*
    designed for case-control PU data.

    The kernelised variant (KLDCE) is not yet implemented.

    References
    ----------
    .. [1] Gong, C., Shi, H., Liu, T., Zhang, C., Yang, J., & Tao, D.
           "Loss Decomposition and Centroid Estimation for Positive and
           Unlabeled Learning."  IEEE TPAMI, 43(3), 918–932, 2021.
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
        reg_strength: float = 1.0,
        centroid_radius: float = 1.0,
        mom_groups: int = 10,
        covariance_ridge: float = 1e-4,
        learning_rate: float = 0.01,
        n_inner_iter: int = 50,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.flip_probability = flip_probability
        self.reg_strength = reg_strength
        self.centroid_radius = centroid_radius
        self.mom_groups = mom_groups
        self.covariance_ridge = covariance_ridge
        self.learning_rate = learning_rate
        self.n_inner_iter = n_inner_iter
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
    ) -> LDCEClassifier:
        """Fit the LDCE classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.  Must be dense (sparse not supported).
        y_pu : np.ndarray of shape (n_samples,)
            PU labels in any convention accepted by
            :func:`~pu_toolbox.core.validation.validate_pu_X_y`
            (``{+1, 0}``, ``{+1, -1}``, ``{1, 0}``, ``{1, -1}``).
        class_prior : float, optional
            Override the class prior derived from *flip_probability*.
            Providing this bypasses the internal prior formula.
        sample_weight : np.ndarray, optional
            Accepted for sklearn API compatibility; currently ignored.

        Returns
        -------
        self : LDCEClassifier
            Fitted estimator.
        """
        X, y_pu = validate_pu_X_y(
            X, y_pu,
            accept_sparse=False,
            estimator_name="LDCEClassifier",
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
        n_P = X_P.shape[0]
        n_U = X_U.shape[0]
        n = X.shape[0]
        d = X.shape[1]

        if n_P == 0:
            raise ValueError("Need at least 1 labeled positive sample.")
        if n_U == 0:
            raise ValueError("Need at least 1 unlabeled sample.")

        # ── Compute / override class prior ────────────────────────────
        if class_prior is not None:
            p = float(class_prior)
            if not (0.0 < p <= 1.0):
                raise ValueError(
                    f"class_prior must be in (0, 1]; got {p}."
                )
        else:
            p = n_P / (n * (1.0 - h))
            if not (0.0 < p <= 1.0):
                raise ValueError(
                    f"Derived class prior p = {p} is out of (0, 1]. "
                    f"Check flip_probability (h={h}) and data: "
                    f"k={n_P}, n={n}. "
                    f"Formula: p = k / [n·(1−h)]."
                )
        self.class_prior_ = p

        # ── Check near-singular denominator ───────────────────────────
        denom = 1.0 - 2.0 * p * h  # = 1 - 2k/n when h from data
        if abs(denom) < 1e-12:
            raise ValueError(
                f"Denominator 1−2ph = {denom:.2e} is near-zero "
                f"(h={h}, p={p}).  The centroid term is ill-conditioned "
                f"for this data / flip_probability combination."
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
        if self.mom_groups < 1:
            raise ValueError(
                f"mom_groups must be >= 1; got {self.mom_groups}."
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0; got {self.learning_rate}."
            )
        if self.n_inner_iter < 1:
            raise ValueError(
                f"n_inner_iter must be >= 1; got {self.n_inner_iter}."
            )
        if self.max_iter < 2:
            raise ValueError(
                f"max_iter must be >= 2; got {self.max_iter}. "
                f"At least 2 iterations are required for convergence checking."
            )

        rng = np.random.RandomState(self.random_state)

        # ── Algorithm 1: MoM centroid of corrupted negative set ───────
        m_hat = _mom_centroid(X_U, self.mom_groups, rng)
        self.corrupted_centroid_ = m_hat.copy()

        # ── Eq. 10: centroid covariance ───────────────────────────────
        S_hat = _centroid_covariance(X_U, self.covariance_ridge)
        self.centroid_covariance_ = S_hat

        # ── Initialise w ──────────────────────────────────────────────
        w = rng.randn(d) * 0.01
        m = m_hat.copy()

        # ── Algorithm 2: alternating optimisation ─────────────────────
        objective_history: list[float] = []
        converged = False

        for t in range(self.max_iter):
            # (a) Fix w, update m via closed form
            m = _update_m(m_hat, S_hat, w, self.centroid_radius)

            # (b) Fix m, update w via subgradient descent
            w = _solve_w_subproblem(
                w, X_P, X_U, m,
                n=n, k=n_P, h=h, p=p, reg=self.reg_strength,
                outer_iter=t,
                learning_rate=self.learning_rate,
                n_inner_iter=self.n_inner_iter,
            )

            # (c) Compute objective and check convergence
            obj = _ldce_objective(
                w, X_P, X_U, m,
                n=n, k=n_P, h=h, p=p, reg=self.reg_strength,
            )
            objective_history.append(obj)

            if t > 0:
                prev_obj = objective_history[-2]
                rel_change = abs(obj - prev_obj) / max(1.0, abs(prev_obj))
                if rel_change < self.tol:
                    converged = True
                    break

        # ── Store fitted attributes ───────────────────────────────────
        self.coef_ = w
        self.true_unlabeled_centroid_ = m
        self.n_labeled_ = n_P
        self.n_unlabeled_ = n_U
        self.n_iter_ = len(objective_history)
        self.objective_history_ = objective_history
        self.converged_ = converged

        if not converged and len(objective_history) >= 2:
            rel_change = abs(
                objective_history[-1] - objective_history[-2]
            ) / max(1.0, abs(objective_history[-2]))
            warnings.warn(
                f"LDCE did not converge within {self.max_iter} "
                f"iterations (last relative change = {rel_change:.2e} "
                f"> tol={self.tol}).  Results may be suboptimal.",
                UserWarning,
                stacklevel=2,
            )

        # ── Finalise ──────────────────────────────────────────────────
        self._X_shape_ = X.shape
        self.classes_ = np.array([0, 1])
        self._is_fitted = True
        return self

    # ── Decision function / predict ──────────────────────────────────

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision scores :math:`w^\\top x`."""
        self._check_is_fitted()
        return X @ self.coef_

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Binary labels: 1 if :math:`w^\\top x \\ge 0`, else 0."""
        return (self._decision_function(X) >= 0.0).astype(int)

    # ── Metadata ─────────────────────────────────────────────────────

    def get_pu_metadata(self) -> dict:
        """Return PU metadata including LDCE-specific diagnostics."""
        meta = super().get_pu_metadata()
        meta.update({
            "flip_probability": getattr(self, "flip_probability_", None),
            "class_prior": getattr(self, "class_prior_", None),
            "centroid_radius": self.centroid_radius,
            "reg_strength": self.reg_strength,
            "n_iter": getattr(self, "n_iter_", None),
            "converged": getattr(self, "converged_", False),
            "n_labeled": getattr(self, "n_labeled_", None),
            "n_unlabeled": getattr(self, "n_unlabeled_", None),
        })
        return meta
