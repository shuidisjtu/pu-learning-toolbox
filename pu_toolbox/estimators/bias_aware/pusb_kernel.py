# ruff: noqa: N803, N806

"""Official-aligned kernel estimator for selection-biased PU learning.

This is a clean-room adapter of the RBF implementation released with PUSB.
It keeps the paper code's basis selection, shuffled-fold cross-validation,
PU objective, and class-prior quantile rule while exposing a stable sklearn
estimator API.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit

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

_OFFICIAL_SIGMA_GRID = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
_OFFICIAL_REG_GRID = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)


def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return pairwise squared Euclidean distances without negative roundoff."""
    distances = (
        np.sum(X * X, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * X @ centers.T
    )
    return np.maximum(distances, 0.0)


def _rbf_design(distances: np.ndarray, sigma: float) -> np.ndarray:
    """Build the official RBF design matrix and append an intercept column.

    Formula identical to ``utils.basis.build_rbf_basis``, but the input is
    a precomputed squared-distance matrix (plus an appended intercept
    column), so the two are intentionally not merged.
    """
    kernel = np.exp(-distances / (2.0 * sigma**2))
    return np.column_stack((kernel, np.ones(kernel.shape[0], dtype=float)))


def _pu_objective_and_gradient(
    coef: np.ndarray,
    design: np.ndarray,
    y_pu: np.ndarray,
    class_prior: float,
    reg_lambda: float,
) -> tuple[float, np.ndarray]:
    """Evaluate the numerically stable PUSB risk and its exact gradient."""
    positive = design[y_pu == 1]
    unlabeled = design[y_pu == 0]
    positive_scores = positive @ coef
    unlabeled_scores = unlabeled @ coef

    objective = (
        -class_prior * float(np.mean(positive_scores))
        + float(np.mean(np.logaddexp(0.0, unlabeled_scores)))
        + 0.5 * reg_lambda * float(coef @ coef)
    )
    gradient = (
        -class_prior * np.mean(positive, axis=0)
        + np.mean(expit(unlabeled_scores)[:, None] * unlabeled, axis=0)
        + reg_lambda * coef
    )
    return objective, gradient


def prior_quantile_predict(scores: np.ndarray, class_prior: float) -> tuple[np.ndarray, float]:
    """Apply the official batch-level class-prior quantile decision rule."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    if not 0.0 < class_prior < 1.0:
        raise ValueError("class_prior must be in (0, 1)")
    index = int(np.floor(scores.size * (1.0 - class_prior)))
    threshold = float(np.sort(scores)[index])
    return (scores > threshold).astype(int), threshold


def _fit_coefficients(
    design: np.ndarray,
    y_pu: np.ndarray,
    class_prior: float,
    reg_lambda: float,
    max_iter: int,
    tol: float,
) -> OptimizeResult:
    initial = np.zeros(design.shape[1], dtype=float)

    def objective(coef):
        return _pu_objective_and_gradient(coef, design, y_pu, class_prior, reg_lambda)

    result = minimize(
        objective,
        initial,
        method="BFGS",
        jac=True,
        options={"maxiter": max_iter, "gtol": tol},
    )
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("PUSB optimization produced non-finite values")
    return result


class PUSBKernelClassifier(BasePUClassifier):
    """Official-aligned RBF PUSB classifier with deterministic CV.

    Parameters mirror the released implementation. The official objective and
    gradient disagree by a factor of two in the regularizer; this adapter uses
    ``0.5 * reg_lambda * ||coef||^2``, whose derivative is the released
    ``reg_lambda * coef`` gradient.
    """

    family = AlgorithmFamily.BIAS_AWARE
    assumption = (Assumption.SAR,)
    scenario = (Scenario.SELECTION_BIASED,)
    requires_class_prior = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_RELATED
    backend = Backend.NUMPY
    maturity = Maturity.RESEARCH

    def __init__(
        self,
        *,
        n_basis: int = 300,
        cv: int = 5,
        sigma_grid: Sequence[float] = _OFFICIAL_SIGMA_GRID,
        reg_grid: Sequence[float] = _OFFICIAL_REG_GRID,
        random_state: int | None = 2018,
        max_iter: int = 200,
        tol: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_basis = n_basis
        self.cv = cv
        self.sigma_grid = sigma_grid
        self.reg_grid = reg_grid
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def _validate_parameters(
        self, X: np.ndarray, class_prior: float | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if class_prior is None or not 0.0 < class_prior < 1.0:
            raise ValueError("PUSBKernelClassifier requires class_prior in (0, 1)")
        if not isinstance(self.n_basis, int) or self.n_basis <= 0:
            raise ValueError("n_basis must be a positive integer")
        if not isinstance(self.cv, int) or self.cv < 2 or self.cv > len(X):
            raise ValueError("cv must be an integer in [2, n_samples]")
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        sigma_grid = np.asarray(self.sigma_grid, dtype=float)
        reg_grid = np.asarray(self.reg_grid, dtype=float)
        if sigma_grid.ndim != 1 or sigma_grid.size == 0 or np.any(sigma_grid <= 0):
            raise ValueError("sigma_grid must be a non-empty sequence of positive values")
        if reg_grid.ndim != 1 or reg_grid.size == 0 or np.any(reg_grid <= 0):
            raise ValueError("reg_grid must be a non-empty sequence of positive values")
        if (
            not np.isfinite(X).all()
            or not np.isfinite(sigma_grid).all()
            or not np.isfinite(reg_grid).all()
        ):
            raise ValueError("X and parameter grids must contain only finite values")
        return sigma_grid, reg_grid

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        X, y_pu = validate_pu_X_y(
            X, y_pu, accept_sparse=False, estimator_name="PUSBKernelClassifier"
        )
        X = np.asarray(X, dtype=float)
        if not np.any(y_pu == 0):
            raise ValueError("PUSBKernelClassifier requires unlabeled samples")
        if sample_weight is not None:
            raise NotImplementedError("The official PUSB objective does not define sample_weight")
        sigma_grid, reg_grid = self._validate_parameters(X, class_prior)

        rng = np.random.RandomState(self.random_state)
        center_indices = rng.permutation(len(X))[: min(self.n_basis, len(X))]
        centers = X[center_indices].copy()
        distances = _squared_distances(X, centers)

        fold_ids = np.floor(np.arange(len(X)) * self.cv / len(X)).astype(int)
        fold_ids = fold_ids[rng.permutation(len(X))]
        for fold in range(self.cv):
            for mask_name, mask in (
                ("training", fold_ids != fold),
                ("validation", fold_ids == fold),
            ):
                if np.unique(y_pu[mask]).size != 2:
                    raise ValueError(
                        f"CV {mask_name} fold {fold} lacks a P or U class; reduce cv or add samples"
                    )

        cv_scores = np.empty((len(sigma_grid), len(reg_grid)), dtype=float)
        cv_convergence = np.empty_like(cv_scores, dtype=bool)
        for sigma_index, sigma in enumerate(sigma_grid):
            design = _rbf_design(distances, float(sigma))
            for reg_index, reg_lambda in enumerate(reg_grid):
                score = 0.0
                converged = True
                for fold in range(self.cv):
                    train = fold_ids != fold
                    validation = ~train
                    result = _fit_coefficients(
                        design[train],
                        y_pu[train],
                        float(class_prior),
                        float(reg_lambda),
                        self.max_iter,
                        self.tol,
                    )
                    fold_score, _ = _pu_objective_and_gradient(
                        result.x,
                        design[validation],
                        y_pu[validation],
                        float(class_prior),
                        float(reg_lambda),
                    )
                    score += fold_score
                    converged = converged and bool(result.success)
                cv_scores[sigma_index, reg_index] = score
                cv_convergence[sigma_index, reg_index] = converged

        best_sigma_index, best_reg_index = np.unravel_index(np.argmin(cv_scores), cv_scores.shape)
        self.sigma_ = float(sigma_grid[best_sigma_index])
        self.reg_lambda_ = float(reg_grid[best_reg_index])
        self.centers_ = centers
        self.center_indices_ = center_indices
        self.fold_ids_ = fold_ids
        self.cv_scores_ = cv_scores
        self.cv_convergence_ = cv_convergence

        design = _rbf_design(distances, self.sigma_)
        result = _fit_coefficients(
            design,
            y_pu,
            float(class_prior),
            self.reg_lambda_,
            self.max_iter,
            self.tol,
        )
        self.coef_ = result.x
        self.optimization_result_ = result
        train_scores = design @ self.coef_
        _, self.threshold_ = prior_quantile_predict(train_scores, float(class_prior))
        self.classes_ = np.array([0, 1])
        self._class_prior = float(class_prior)
        self._X_shape_ = X.shape
        self._is_fitted = True
        return self

    def _decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._X_shape_[1]:
            raise ValueError(f"X must have shape (n_samples, {self._X_shape_[1]})")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return _rbf_design(_squared_distances(X, self.centers_), self.sigma_) @ self.coef_

    def _predict(self, X):
        return (self._decision_function(X) > self.threshold_).astype(int)

    def predict_with_prior_quantile(self, X) -> np.ndarray:
        """Predict with the released code's batch-dependent quantile rule."""
        predictions, _ = prior_quantile_predict(self._decision_function(X), self._class_prior)
        return predictions

    def get_pu_metadata(self) -> dict:
        metadata = super().get_pu_metadata()
        metadata["official_compatibility"] = {
            "regularizer": "0.5 * lambda * ||coef||^2, matching the released gradient",
            "public_threshold": "frozen training-score prior quantile",
            "batch_quantile_api": "predict_with_prior_quantile",
        }
        return metadata
