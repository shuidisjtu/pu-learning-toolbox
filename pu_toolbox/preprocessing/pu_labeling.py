# ruff: noqa: N802, N803, N806

"""PNU labels and Gaussian sample designs other than same-pool OS labeling."""

from __future__ import annotations

import numpy as np

from pu_toolbox.core.config import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    UNLABELED_LABEL,
)
from pu_toolbox.core.random import check_random_state
from pu_toolbox.core.validation import validate_true_binary_labels

__all__ = [
    "make_gaussian_case_control_data",
    "make_gaussian_pu_data",
    "make_pnu_labels",
]

# ═════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════


def _validate_binary_labels(y_true: np.ndarray) -> None:
    """Validate that *y_true* is a 1-D binary label vector with both classes.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.

    Raises
    ------
    ValueError
        If *y_true* fails validation.
    """
    y = np.asarray(y_true, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"y_true must be 1-D; got ndim={y.ndim}.")
    validate_true_binary_labels(y, estimator_name="y_true")
    if len(np.unique(y)) < 2:
        raise ValueError("y_true must contain both 0 (negative) and 1 (positive) samples.")


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
    """Generate the legacy Gaussian demonstration with fully visible positives.

    Positive class drawn from ``N(+separation/2, 1)``, negative class from
    ``N(-separation/2, 1)``.  All positive samples are labeled ``+1``;
    all negative samples are unlabeled ``0``.
    Use :func:`make_gaussian_case_control_data` for an independent population
    mixture U with a specified target prior.

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
        Empirical positive fraction in the concatenated table, n_p / (n_p + n_u).
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


def make_gaussian_case_control_data(
    n_p: int = 50,
    n_u: int = 100,
    *,
    class_prior: float = 0.5,
    n_features: int = 5,
    separation: float = 2.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Draw independent P and population-mixture U for a Gaussian TS design.

    The positive conditional is N(+separation/2, I), the negative conditional
    N(-separation/2, I). P is sampled from the positive conditional; U is
    independently sampled from the population with ``P(Y=1)=class_prior``.

    Parameters
    ----------
    n_p, n_u : int, default 50, 100
        Positive-role and unlabeled-role counts, each at least one.
    class_prior : float, default 0.5
        Population positive probability in [0, 1].
    n_features : int, default 5
        Positive feature dimension.
    separation : float, default 2.0
        Finite nonnegative distance between class centres along each feature.
    random_state : int, np.random.RandomState or None, optional
        Random source for independent sequential draws.

    Returns
    -------
    X, y_pu, y_true, metadata : tuple
        Features and aligned PU/true labels, ordered P then U. Metadata holds
        the target population prior, empirical U/table priors, source sample
        identities, class parameters and sampling policy. True labels support
        synthetic-data auditing and evaluation.
    """
    for name, count in (("n_p", n_p), ("n_u", n_u), ("n_features", n_features)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")
        if count < 1:
            raise ValueError(f"{name} must be >= 1.")
    for name, value in (("class_prior", class_prior), ("separation", separation)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a real number.")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if not 0 <= class_prior <= 1:
        raise ValueError("class_prior must be in [0, 1].")
    if separation < 0:
        raise ValueError("separation must be nonnegative.")

    rng = check_random_state(random_state)
    delta = float(separation) / 2
    positive = rng.normal(size=(n_p, n_features)) + delta
    unlabeled_truth = (rng.uniform(size=n_u) < class_prior).astype(int)
    unlabeled = rng.normal(size=(n_u, n_features)) + (2 * unlabeled_truth[:, None] - 1) * delta
    X = np.vstack((positive, unlabeled))
    y_pu = np.r_[np.ones(n_p, dtype=int), np.zeros(n_u, dtype=int)]
    y_true = np.r_[np.ones(n_p, dtype=int), unlabeled_truth]
    metadata = {
        "input_sampling": "ts",
        "sampling_policy": "independent_generative",
        "population_class_prior": float(class_prior),
        "unlabeled_class_prior": float(unlabeled_truth.mean()),
        "training_class_prior": float(y_true.mean()),
        "n_positive": int(n_p),
        "n_unlabeled": int(n_u),
        "source_distribution": {
            "family": "gaussian",
            "positive_mean": delta,
            "negative_mean": -delta,
            "covariance": "identity",
            "n_features": int(n_features),
        },
        "sample_ids": [f"positive:{i}" for i in range(n_p)]
        + [f"population:{i}" for i in range(n_u)],
        "random_state": int(random_state) if isinstance(random_state, (int, np.integer)) else None,
    }
    return X, y_pu, y_true, metadata
