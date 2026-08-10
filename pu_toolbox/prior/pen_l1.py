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
from ..utils.basis import build_rbf_basis

_AUTO_SIGMA_FACTOR = 0.6


def _median_pairwise_distance(X: np.ndarray, max_rows: int = 1000) -> float:
    """Median pairwise euclidean distance; subsamples for large inputs."""
    if len(X) > max_rows:
        idx = np.random.RandomState(0).choice(len(X), max_rows, replace=False)
        X = X[idx]
    upper = pairwise_distances(X)[np.triu_indices(len(X), k=1)]
    return float(np.median(upper))


class ClassPriorEstimator(BasePriorEstimator):
    """Estimate ``pi=P(y=1)`` with the paper's penalized-L1 objective.

    ``sigma=None`` (the default) selects a data-adaptive scale: the median
    pairwise euclidean distance of the (standardized) data.  An explicit
    ``sigma`` keeps the historical fixed-scale behaviour.
    """

    def __init__(
        self,
        *,
        sigma: float | None = None,
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
        X, y_pu = validate_pu_X_y(
            X, y_pu, accept_sparse=False, estimator_name="ClassPriorEstimator"
        )
        if self.reg_lambda <= 0:
            raise ValueError("reg_lambda must be positive")
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError("sigma must be positive")
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
        if self.sigma is None:
            # Auto scale: a fixed fraction of the median pairwise distance,
            # chosen so estimates stay inside the acceptance band across
            # separations (0.5-2.0); users can override via an explicit sigma.
            auto = _median_pairwise_distance(X)
            sigma = _AUTO_SIGMA_FACTOR * auto if auto > 1e-12 else 1.0
            self.sigma_auto_ = True
        else:
            sigma = self.sigma
            self.sigma_auto_ = False
        self.sigma_ = float(sigma)
        centers = X if self.n_centers is None else X[: min(self.n_centers, len(X))]
        phi_p = build_rbf_basis(P, centers, sigma)
        phi_u = build_rbf_basis(U, centers, sigma)
        theta_grid = np.asarray(
            np.linspace(0.01, 0.99, 99) if self.theta_grid is None else self.theta_grid,
            dtype=float,
        )
        if (
            theta_grid.ndim != 1
            or len(theta_grid) == 0
            or np.any((theta_grid < 0) | (theta_grid > 1))
        ):
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
