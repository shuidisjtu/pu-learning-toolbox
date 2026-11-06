# ruff: noqa: N802, N803, N806

"""PU / PNU label generation and synthetic data construction.

This module provides reusable utilities to convert fully-labeled binary
datasets into PU or PNU labels under SCAR and fixed-count policies,
and to generate synthetic Gaussian OS and independent case-control datasets for
benchmarks and examples.

All functions return labels in canonical form (``{+1, 0}`` for PU,
``{+1, -1, 0}`` for PNU), compatible with :mod:`pu_toolbox.core.labels`.
"""

from __future__ import annotations

import warnings

import numpy as np

from pu_toolbox.core.config import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    UNLABELED_LABEL,
)
from pu_toolbox.core.random import check_random_state
from pu_toolbox.core.validation import validate_true_binary_labels

from ._sampling import fixed_positive_count, sample_positive_labels

__all__ = [
    "make_case_control_labels",
    "make_fixed_count_labels",
    "make_gaussian_case_control_data",
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
    validate_true_binary_labels(y, estimator_name="y_true")
    if require_both and len(np.unique(y)) < 2:
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

    Under SCAR (Selected Completely At Random), a **fixed count** of
    positives is labeled: ``n_L = max(1, round(c * n_pos))`` positives are
    sampled uniformly without replacement.  This matches the fixed-count
    policy of the PU survey protocol (no per-sample Bernoulli labeling).

    All true negatives (``y_true == 0``) become unlabeled (0).

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels in ``{0, 1}``.
    c : float, default 0.5
        Labeling ratio (label frequency).  Must satisfy ``0 < c <= 1``.
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

    y_true = np.asarray(y_true, dtype=int)
    n_labeled = fixed_positive_count(int(np.sum(y_true == POSITIVE_LABEL)), c)
    return sample_positive_labels(y_true, n_labeled, random_state)


# ═════════════════════════════════════════════════════════════════════
# Fixed-count labeling
# ═════════════════════════════════════════════════════════════════════


def make_fixed_count_labels(
    y_true: np.ndarray,
    n_labeled: int,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Label a fixed number of positives uniformly within one source pool.

    Exactly ``n_labeled`` true positives are selected without replacement.
    All source rows retain their positions; the remaining rows receive 0.
    This constructs an OS PU view of the supplied clean pool.

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
    _validate_binary_labels(y_true, require_both=False)

    if isinstance(n_labeled, (bool, np.bool_)) or not isinstance(n_labeled, (int, np.integer)):
        raise TypeError("n_labeled must be an integer.")
    if n_labeled < 1:
        raise ValueError(f"n_labeled must be >= 1; got {n_labeled}.")

    y_true_arr = np.asarray(y_true, dtype=int)
    pos_idx = np.where(y_true_arr == 1)[0]

    if n_labeled > len(pos_idx):
        raise ValueError(
            f"n_labeled ({n_labeled}) exceeds the number of true positives ({len(pos_idx)})."
        )

    return sample_positive_labels(y_true_arr, n_labeled, random_state)


def make_case_control_labels(
    y_true: np.ndarray,
    n_labeled: int,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Compatibility alias for same-pool fixed-count positive labeling.

    Deprecated for removal in 2.0. Use :func:`make_fixed_count_labels` for
    this OS construction, or :func:`make_gaussian_case_control_data` for
    independently generated positive and population samples.
    The alias preserves the historical two-class input requirement and seed.
    """
    warnings.warn(
        "make_case_control_labels is deprecated; use make_fixed_count_labels for "
        "same-pool labeling. The alias is scheduled for removal in 2.0.",
        FutureWarning,
        stacklevel=2,
    )
    _validate_binary_labels(y_true)
    return make_fixed_count_labels(y_true, n_labeled, random_state=random_state)


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
    - ``"fixed_count"`` — requires *n_labeled*; calls
      :func:`make_fixed_count_labels`.
    - ``"case_control"`` — deprecated alias with the same fixed-count policy.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels in ``{0, 1}``.
    mechanism : {"scar", "fixed_count", "case_control"}, default "scar"
        The labeling mechanism.
    c : float or None, optional
        Labeling propensity for SCAR.  Required when ``mechanism="scar"``.
    n_labeled : int or None, optional
        Number of positives to label for ``"fixed_count"`` or its legacy alias.
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

    if mechanism in {"fixed_count", "case_control"}:
        if n_labeled is None:
            raise ValueError(f"mechanism={mechanism!r} requires the 'n_labeled' parameter.")
        labeler = (
            make_fixed_count_labels if mechanism == "fixed_count" else make_case_control_labels
        )
        return labeler(
            y_true,
            n_labeled=n_labeled,
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown mechanism {mechanism!r}. Expected 'scar', 'fixed_count', or 'case_control'."
    )


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
    SCAR labeling (fixed-count policy, see :func:`make_scar_labels`).

    Parameters
    ----------
    n : int, default 100
        Number of samples **per class** (total = 2n).
    c : float, default 0.5
        SCAR labeling ratio (label frequency).  Must satisfy ``0 < c <= 1``.
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
