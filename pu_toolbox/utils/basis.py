# ruff: noqa: N803, N806

"""Basis-function builders for linear-in-parameter models.

These functions construct the feature mapping φ(x) used by kernel-based
PU estimators (uPU, PNU, LLSVM, Centroid PU, etc.).  They are extracted
to a shared location so that every risk-estimation classifier can reuse
the same implementations.

All functions operate on NumPy arrays and have no trainable parameters.
"""

from __future__ import annotations

import numpy as np


def build_linear_basis(X: np.ndarray) -> np.ndarray:
    """Linear basis: φ(x) = x.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        Linear basis expansion (identity).
    """
    return X


def build_rbf_basis(
    X: np.ndarray,
    centers: np.ndarray,
    kernel_width: float,
) -> np.ndarray:
    """Gaussian / RBF basis: φ_j(x) = exp(−||x − c_j||² / (2σ²)).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    centers : np.ndarray of shape (n_centers, n_features)
        RBF centre points.
    kernel_width : float
        Kernel bandwidth σ.  Must be > 0.

    Returns
    -------
    np.ndarray of shape (n_samples, n_centers)
        RBF basis expansion.
    """
    # squared_dist: ||X||² − 2 X·C^T + ||C||²
    sq_X = np.sum(X**2, axis=1, keepdims=True)  # (n, 1)
    sq_C = np.sum(centers**2, axis=1, keepdims=True).T  # (1, m)
    dist2 = sq_X - 2.0 * X.dot(centers.T) + sq_C  # (n, m)
    return np.exp(-dist2 / (2.0 * kernel_width**2))


def subsample_centers(
    X_pool: np.ndarray,
    n_centers: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Randomly subsample *n_centers* rows from *X_pool*.

    Parameters
    ----------
    X_pool : np.ndarray of shape (n_samples, n_features)
        Candidate centre pool (typically U samples).
    n_centers : int
        Desired number of centres.  Capped at ``len(X_pool)``.
    rng : np.random.RandomState
        Seeded random state for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_centers, n_features)
    """
    n = X_pool.shape[0]
    n_centers = min(n_centers, n)
    idx = rng.choice(n, size=n_centers, replace=False)
    return X_pool[idx]
