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
        Linear basis expansion (identity — returns a copy).
    """
    return X.copy()


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


def resolve_basis_fn(
    basis: str,
    X_pool: np.ndarray,
    *,
    kernel_width: float | None = None,
    n_centers: int | None = None,
    rng: np.random.RandomState,
) -> tuple[callable, int, np.ndarray | None]:
    """Resolve basis-function callable and metadata from config.

    Single source of truth for the ``basis=`` dispatch logic shared by
    ``UPUClassifier``, ``PNUClassifier``, and future risk-estimation
    classifiers.

    Parameters
    ----------
    basis : {"linear", "rbf"}
        Basis type.
    X_pool : np.ndarray of shape (n_samples, n_features)
        Pool for RBF centre subsampling (typically U samples).  Ignored
        for ``basis="linear"``.
    kernel_width : float or None
        Required when ``basis="rbf"``; must be > 0.
    n_centers : int or None
        Number of RBF centres.  Default: ``min(200, len(X_pool))``.
        Ignored for ``basis="linear"``.  Must be > 0 if given.
    rng : np.random.RandomState
        Seeded random state for centre subsampling.

    Returns
    -------
    phi_fn : callable
        ``phi_fn(X) -> np.ndarray`` — basis expansion.
    n_basis : int
        Dimensionality of the basis expansion.
    centers : np.ndarray or None
        RBF centres (``None`` for linear).

    Raises
    ------
    ValueError
        If *basis* is unrecognised, *kernel_width* is missing/invalid for
        RBF, or *n_centers* <= 0.
    """
    n_samples = X_pool.shape[0]

    if basis == "linear":
        return build_linear_basis, X_pool.shape[1], None

    if basis == "rbf":
        if kernel_width is None or kernel_width <= 0:
            raise ValueError(
                f"kernel_width must be > 0 for basis='rbf'; "
                f"got {kernel_width}."
            )
        if n_centers is not None and n_centers <= 0:
            raise ValueError(
                f"n_centers must be > 0; got {n_centers}."
            )
        n_centers_val = (
            n_centers if n_centers is not None else min(200, n_samples)
        )
        centers = subsample_centers(X_pool, n_centers_val, rng)
        n_basis = centers.shape[0]

        def _rbf_fn(X_in: np.ndarray) -> np.ndarray:
            return build_rbf_basis(X_in, centers, kernel_width)

        return _rbf_fn, n_basis, centers

    raise ValueError(f"Unknown basis {basis!r}.")
