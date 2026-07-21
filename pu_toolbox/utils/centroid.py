# ruff: noqa: N803, N806, S101

"""Shared centroid estimation primitives for LDCE / KLDCE.

Provides median-of-means (MoM) centroid estimation and empirical centroid
covariance — both extracted from ``ldce.py`` so that KLDCE can reuse them
without duplicating code.

.. note::

   ``_centroid_covariance`` returns the **raw** covariance matrix
   (no ridge).  Callers are responsible for adding ridge if needed.
"""

from __future__ import annotations

import numpy as np


def _mom_centroid(
    X_U: np.ndarray,
    g: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Median-of-means centroid estimate (Algorithm 1 in Gong et al. 2021).

    Parameters
    ----------
    X_U : np.ndarray of shape (n_U, d)
        Transformed unlabeled samples.  For LDCE/KLDCE the caller must
        pass ``-X_U`` because the paper's MoM operates on
        {:math:`\\tilde{y}_i \\cdot x_i \\mid \\tilde{y}_i = -1`} = {-x_i}.
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
        If *g* exceeds the number of samples.
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
) -> np.ndarray:
    """Empirical covariance of the corrupted centroid (Eq. 10, raw).

    .. math::

        \\hat{S}_{\\text{raw}} =
        \\frac{X_U^\\top X_U}{|U|^2}
        - \\frac{(\\sum X_U)(\\sum X_U)^\\top}{|U|^2}

    **No ridge is applied** — callers add ``ridge * I`` themselves
    depending on their numerical requirements.

    Note: because all unlabeled samples have corrupted label
    :math:`\\tilde{y}_i = -1`, the term
    :math:`(\\sum x_i \\tilde{y}_i)` equals :math:`-\\sum x_i`, and its
    outer product is identical to :math:`(\\sum x_i)(\\sum x_i)^\\top`.

    Parameters
    ----------
    X_U : np.ndarray of shape (n_U, d)
        Unlabeled samples.

    Returns
    -------
    S_hat : np.ndarray of shape (d, d)
        Raw centroid covariance matrix (no ridge).
    """
    n_U = X_U.shape[0]
    sum_X = X_U.sum(axis=0)  # (d,)

    S = (X_U.T @ X_U) / (n_U ** 2) - np.outer(sum_X, sum_X) / (n_U ** 2)
    return S
