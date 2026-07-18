# ruff: noqa: N802, N803, N806

"""PU / PNU label generation and synthetic data construction.

This module provides reusable utilities to convert fully-labeled binary
datasets into PU or PNU labels under different labeling mechanisms (SCAR,
case-control), and to generate synthetic Gaussian PU datasets for
benchmarks and examples.

All functions return labels in canonical form (``{+1, 0}`` for PU,
``{+1, -1, 0}`` for PNU), compatible with :mod:`pu_toolbox.core.labels`.
"""

from __future__ import annotations

import numpy as np

from pu_toolbox.core.config import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    UNLABELED_LABEL,
)
from pu_toolbox.core.random import check_random_state

__all__ = [
    "make_case_control_labels",
    "make_gaussian_pu_data",
    "make_pnu_labels",
    "make_pu_labels",
    "make_scar_dataset",
    "make_scar_labels",
]

# ═════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════


def _validate_binary_labels(y_true: np.ndarray, *, require_both: bool = True) -> None:
    """Validate that *y_true* is a 1-D binary label vector.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    require_both : bool
        If ``True``, require both classes (0 and 1) to be present.

    Raises
    ------
    ValueError
        If *y_true* fails validation.
    """
    y = np.asarray(y_true, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"y_true must be 1-D; got ndim={y.ndim}.")
    unique = set(np.unique(y))
    if not unique <= {0.0, 1.0}:
        raise ValueError(f"y_true must contain only {{0, 1}} values; got {sorted(unique)}.")
    if require_both and len(unique) < 2:
        raise ValueError("y_true must contain both 0 (negative) and 1 (positive) samples.")


# ═════════════════════════════════════════════════════════════════════
# SCAR labeling
# ═════════════════════════════════════════════════════════════════════


def make_scar_labels(
    y_true: np.ndarray,
    c: float = 0.5,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Convert true binary labels to PU labels under the SCAR assumption.

    Under SCAR (Selected Completely At Random), each positive sample is
    labeled independently with constant probability *c*:

        P(s=1 | y=1, x) = c

    All true negatives (``y_true == 0``) become unlabeled (0).

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels in ``{0, 1}``.
    c : float, default 0.5
        Labeling propensity.  Must satisfy ``0 < c <= 1``.
    random_state : int or np.random.RandomState or None, optional
        Random seed or RandomState for reproducibility.

    Returns
    -------
    y_pu : np.ndarray of shape (n_samples,) and dtype int
        PU labels in canonical ``{+1, 0}`` form.

    Raises
    ------
    ValueError
        If *c* is not in ``(0, 1]``, *y_true* has invalid shape/values,
        or no positives remain after labeling.
    """
    _validate_binary_labels(y_true, require_both=False)

    if not (0.0 < c <= 1.0):
        raise ValueError(f"c must be in (0, 1]; got {c}.")

    rng = check_random_state(random_state)
    y_true = np.asarray(y_true, dtype=int)
    pos_mask = y_true == 1

    if not np.any(pos_mask):
        # No true positives — all unlabeled.
        return np.zeros(len(y_true), dtype=int)

    y_pu = np.zeros(len(y_true), dtype=int)
    n_pos = int(np.sum(pos_mask))
    n_labeled = max(1, int(np.round(n_pos * c)))
    # Ensure n_labeled ≤ n_pos (can happen with c > 0 but very few positives).
    n_labeled = min(n_labeled, n_pos)

    pos_idx = np.where(pos_mask)[0]
    labeled_idx = rng.choice(pos_idx, size=n_labeled, replace=False)
    y_pu[labeled_idx] = POSITIVE_LABEL

    return y_pu


# ═════════════════════════════════════════════════════════════════════
# Case-control labeling
# ═════════════════════════════════════════════════════════════════════


def make_case_control_labels(
    y_true: np.ndarray,
    n_labeled: int,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Create PU labels by selecting a fixed number of positives to label.

    In the case-control scenario, labeled positives and unlabeled data
    are collected separately.  This function mirrors that by randomly
    choosing exactly *n_labeled* true-positive samples to receive label
    ``+1``; all other samples (remaining positives + all negatives) become
    unlabeled (``0``).

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels in ``{0, 1}``.
    n_labeled : int
        Number of true positives to label.  Must be ``>= 1`` and
        ``<=`` the total number of true positives.
    random_state : int or np.random.RandomState or None, optional
        Random seed or RandomState for reproducibility.

    Returns
    -------
    y_pu : np.ndarray of shape (n_samples,) and dtype int
        PU labels in canonical ``{+1, 0}`` form.

    Raises
    ------
    ValueError
        If *n_labeled* is out of range or *y_true* is invalid.
    """
    _validate_binary_labels(y_true)

    if n_labeled < 1:
        raise ValueError(f"n_labeled must be >= 1; got {n_labeled}.")

    y_true_arr = np.asarray(y_true, dtype=int)
    pos_idx = np.where(y_true_arr == 1)[0]

    if n_labeled > len(pos_idx):
        raise ValueError(
            f"n_labeled ({n_labeled}) exceeds the number of true positives ({len(pos_idx)})."
        )

    rng = check_random_state(random_state)
    chosen = rng.choice(pos_idx, size=n_labeled, replace=False)

    y_pu = np.zeros(len(y_true_arr), dtype=int)
    y_pu[chosen] = POSITIVE_LABEL
    return y_pu


# ═════════════════════════════════════════════════════════════════════
# Unified PU label dispatcher
# ═════════════════════════════════════════════════════════════════════


def make_pu_labels(
    y_true: np.ndarray,
    mechanism: str = "scar",
    c: float | None = None,
    n_labeled: int | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Convert true binary labels to PU labels via the specified mechanism.

    This is the recommended entry point.  Use *mechanism* to select the
    labeling scheme:

    - ``"scar"`` — requires *c*; calls :func:`make_scar_labels`.
    - ``"case_control"`` — requires *n_labeled*; calls
      :func:`make_case_control_labels`.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels in ``{0, 1}``.
    mechanism : {"scar", "case_control"}, default "scar"
        The labeling mechanism.
    c : float or None, optional
        Labeling propensity for SCAR.  Required when ``mechanism="scar"``.
    n_labeled : int or None, optional
        Number of positives to label for case-control.  Required when
        ``mechanism="case_control"``.
    random_state : int or np.random.RandomState or None, optional
        Random seed or RandomState for reproducibility.

    Returns
    -------
    y_pu : np.ndarray of shape (n_samples,) and dtype int
        PU labels in canonical ``{+1, 0}`` form.

    Raises
    ------
    ValueError
        If *mechanism* is unrecognised or the required parameter is missing.
    """
    if mechanism == "scar":
        if c is None:
            raise ValueError("mechanism='scar' requires the 'c' parameter (labeling propensity).")
        return make_scar_labels(y_true, c=c, random_state=random_state)

    if mechanism == "case_control":
        if n_labeled is None:
            raise ValueError("mechanism='case_control' requires the 'n_labeled' parameter.")
        return make_case_control_labels(
            y_true,
            n_labeled=n_labeled,
            random_state=random_state,
        )

    raise ValueError(f"Unknown mechanism {mechanism!r}. Expected 'scar' or 'case_control'.")


# ═════════════════════════════════════════════════════════════════════
# PNU labeling
# ═════════════════════════════════════════════════════════════════════


def make_pnu_labels(
    y_true: np.ndarray,
    n_negatives: int,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Create PNU (Positive / Negative / Unlabeled) labels from true labels.

    All true positives receive ``+1`` (P).  A random subset of
    *n_negatives* true negatives receive ``-1`` (N).  The remaining
    negatives become unlabeled ``0`` (U).

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels in ``{0, 1}``.
    n_negatives : int
        Number of true negatives to label as ``-1``.  Must be ``>= 1``
        and ``<=`` the total number of true negatives.
    random_state : int or np.random.RandomState or None, optional
        Random seed or RandomState for reproducibility.

    Returns
    -------
    y_pnu : np.ndarray of shape (n_samples,) and dtype int
        PNU labels in canonical ``{+1, -1, 0}`` form.

    Raises
    ------
    ValueError
        If *n_negatives* is invalid or true labels fail validation.
    """
    _validate_binary_labels(y_true)

    if n_negatives < 1:
        raise ValueError(f"n_negatives must be >= 1; got {n_negatives}.")

    y_true_arr = np.asarray(y_true, dtype=int)
    pos_mask = y_true_arr == 1
    neg_mask = y_true_arr == 0
    neg_idx = np.where(neg_mask)[0]

    if n_negatives > len(neg_idx):
        raise ValueError(
            f"n_negatives ({n_negatives}) exceeds the number of true negatives ({len(neg_idx)})."
        )

    rng = check_random_state(random_state)
    chosen_neg = rng.choice(neg_idx, size=n_negatives, replace=False)

    y_pnu = np.full(len(y_true_arr), UNLABELED_LABEL, dtype=int)
    y_pnu[pos_mask] = POSITIVE_LABEL
    y_pnu[chosen_neg] = NEGATIVE_LABEL

    return y_pnu


# ═════════════════════════════════════════════════════════════════════
# Synthetic dataset generators
# ═════════════════════════════════════════════════════════════════════


def make_gaussian_pu_data(
    n_p: int = 50,
    n_u: int = 100,
    n_features: int = 5,
    separation: float = 2.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate a 2-class Gaussian PU dataset (case-control style).

    Positive class drawn from ``N(+separation/2, 1)``, negative class from
    ``N(-separation/2, 1)``.  All positive samples are labeled ``+1``;
    all negative samples are unlabeled ``0``.

    Parameters
    ----------
    n_p : int, default 50
        Number of positive samples.
    n_u : int, default 100
        Number of unlabeled (negative) samples.
    n_features : int, default 5
        Dimensionality of the feature space.
    separation : float, default 2.0
        Distance between the two class centres (``2 * delta``).
    random_state : int or np.random.RandomState or None, optional
        Random seed or RandomState for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_p + n_u, n_features)
        Feature matrix.
    y_pu : np.ndarray of shape (n_p + n_u,) and dtype int
        PU labels in canonical ``{+1, 0}``.
    class_prior : float
        True class prior π = n_p / (n_p + n_u).
    """
    rng = check_random_state(random_state)
    delta = separation / 2.0

    X_p = rng.randn(n_p, n_features) + delta
    X_n = rng.randn(n_u, n_features) - delta
    X = np.vstack([X_p, X_n])

    y_pu = np.concatenate(
        [np.full(n_p, POSITIVE_LABEL, dtype=int), np.full(n_u, UNLABELED_LABEL, dtype=int)],
    )
    class_prior = n_p / (n_p + n_u)

    return X, y_pu, class_prior


def make_scar_dataset(
    n: int = 100,
    c: float = 0.5,
    n_features: int = 5,
    separation: float = 4.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a 2-class Gaussian dataset with SCAR-based PU labels.

    Creates a balanced binary dataset with positive class centred at
    ``+separation/2`` and negative at ``-separation/2``, then applies
    SCAR labeling (each true positive labeled with probability *c*).

    Parameters
    ----------
    n : int, default 100
        Number of samples **per class** (total = 2n).
    c : float, default 0.5
        SCAR labeling propensity.  Must satisfy ``0 < c <= 1``.
    n_features : int, default 5
        Dimensionality of the feature space.
    separation : float, default 4.0
        Distance between the two class centres (``2 * delta``).
    random_state : int or np.random.RandomState or None, optional
        Random seed or RandomState for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (2n, n_features)
        Feature matrix.
    y_pu : np.ndarray of shape (2n,) and dtype int
        PU labels in canonical ``{+1, 0}``.
    y_true : np.ndarray of shape (2n,) and dtype int
        True binary labels ``{0, 1}`` (for evaluation).
    """
    rng = check_random_state(random_state)
    delta = separation / 2.0

    X_pos = rng.randn(n, n_features) + delta
    X_neg = rng.randn(n, n_features) - delta
    X = np.vstack([X_pos, X_neg])

    y_true = np.hstack([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    y_pu = make_scar_labels(y_true, c=c, random_state=rng)

    return X, y_pu, y_true
