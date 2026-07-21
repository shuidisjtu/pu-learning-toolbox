# ruff: noqa: N803, N806, E501

"""Penalized-L1 class-prior estimation for PU data.

This is the closed-form penL1 estimator described by du Plessis, Niu and
Sugiyama.  The implementation uses a Gaussian basis and searches a supplied
grid of candidate priors; it intentionally does not hide cross-validation
inside ``fit``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

from ..core.base import BasePriorEstimator
from ..core.exceptions import NotFittedError
from ..core.validation import validate_pu_X_y


class ClassPriorEstimator(BasePriorEstimator):
    """Estimate ``pi=P(y=1)`` with the paper's penalized-L1 objective."""

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        reg_lambda: float = 1e-2,
        theta_grid: np.ndarray | None = None,
        n_centers: int | None = 200,
        standardize: bool = True,
    ) -> None:
        self.sigma = sigma
        self.reg_lambda = reg_lambda
        self.theta_grid = theta_grid
        self.n_centers = n_centers
        self.standardize = standardize

    def fit(self, X: np.ndarray, y_pu: np.ndarray) -> ClassPriorEstimator:
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="ClassPriorEstimator")
        if self.sigma <= 0 or self.reg_lambda <= 0:
            raise ValueError("sigma and reg_lambda must be positive")
        X = np.asarray(X, dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values")
        P, U = X[y_pu == 1], X[y_pu == 0]
        if len(U) == 0:
            raise ValueError("ClassPriorEstimator requires unlabeled samples")
        if self.standardize:
            mean, scale = X.mean(axis=0), X.std(axis=0)
            scale = np.where(scale > 1e-12, scale, 1.0)
            X, P, U = (X - mean) / scale, (P - mean) / scale, (U - mean) / scale
            self.mean_, self.scale_ = mean, scale
        centers = X if self.n_centers is None else X[: min(self.n_centers, len(X))]
        phi_p = np.exp(-pairwise_distances(P, centers, metric="sqeuclidean") / (2.0 * self.sigma**2))
        phi_u = np.exp(-pairwise_distances(U, centers, metric="sqeuclidean") / (2.0 * self.sigma**2))
        theta_grid = np.asarray(
            np.linspace(0.01, 0.99, 99) if self.theta_grid is None else self.theta_grid,
            dtype=float,
        )
        if theta_grid.ndim != 1 or len(theta_grid) == 0 or np.any((theta_grid < 0) | (theta_grid > 1)):
            raise ValueError("theta_grid must be a non-empty one-dimensional grid in [0, 1]")
        beta_u = phi_u.mean(axis=0)
        objectives = []
        for theta in theta_grid:
            beta = theta * phi_p.mean(axis=0) - beta_u
            positive_beta = np.maximum(beta, 0.0)
            objectives.append(float(np.dot(positive_beta, beta) / self.reg_lambda - theta + 1.0))
        best = int(np.argmin(objectives))
        self.class_prior_ = float(theta_grid[best])
        self.objective_values_ = np.asarray(objectives)
        self.theta_grid_ = theta_grid
        self.n_features_in_ = X.shape[1]
        self.n_centers_ = len(centers)
        self._is_fitted = True
        return self

    def estimate(self) -> float:
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("ClassPriorEstimator is not fitted. Call fit() first.")
        return self.class_prior_


PenL1Estimator = ClassPriorEstimator
