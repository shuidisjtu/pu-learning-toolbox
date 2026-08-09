# ruff: noqa: N803, N806

"""Synthetic SCAR/SAR labeling mechanisms for PU experiments.

The helpers in this module keep three concerns separate:

* :func:`make_sar_propensity` constructs a deterministic labeling
  propensity with a calibrated positive-class mean.
* :func:`make_sar_labels` samples observable PU labels from that propensity.
* :func:`make_sar_dataset` creates Gaussian data and retains hidden truth for
  benchmark evaluation.

Returned propensity values represent ``P(S=1 | Y, X)``. They are therefore
zero for true negatives, which can never receive a positive label.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import expit

from pu_toolbox.core.config import POSITIVE_LABEL, UNLABELED_LABEL
from pu_toolbox.core.random import check_random_state
from pu_toolbox.core.validation import check_scalar_in_range

SARMechanism = Literal["scar", "linear", "nonlinear"]
SAR_MECHANISMS: tuple[SARMechanism, ...] = ("scar", "linear", "nonlinear")

__all__ = [
    "SAR_MECHANISMS",
    "SARMechanism",
    "make_sar_dataset",
    "make_sar_labels",
    "make_sar_propensity",
]


def _validate_x_y(X: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y_true)
    if X_array.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={X_array.ndim}.")
    if len(X_array) == 0:
        raise ValueError("X must contain at least one sample.")
    if not np.isfinite(X_array).all():
        raise ValueError("X must contain only finite values.")
    if y_array.ndim != 1:
        raise ValueError(f"y_true must be 1-D; got ndim={y_array.ndim}.")
    if len(X_array) != len(y_array):
        raise ValueError(
            f"X and y_true have inconsistent lengths: {len(X_array)} != {len(y_array)}."
        )
    unique = set(np.unique(y_array))
    if not unique <= {0, 1}:
        raise ValueError(f"y_true must contain only {{0, 1}} values; got {sorted(unique)}.")
    return X_array, y_array.astype(int, copy=False)


def _normalize_mechanism(mechanism: str) -> SARMechanism:
    if not isinstance(mechanism, str):
        raise TypeError(f"mechanism must be a string; got {type(mechanism).__name__}.")
    normalized = mechanism.lower().replace("-", "").replace("_", "")
    aliases: dict[str, SARMechanism] = {
        "scar": "scar",
        "linear": "linear",
        "nonlinear": "nonlinear",
    }
    if normalized not in aliases:
        expected = ", ".join(repr(item) for item in SAR_MECHANISMS)
        raise ValueError(f"Unknown mechanism {mechanism!r}. Expected one of: {expected}.")
    return aliases[normalized]


def _validate_rate_and_strength(label_frequency: float, strength: float) -> None:
    if isinstance(label_frequency, bool) or not np.isscalar(label_frequency):
        raise TypeError("label_frequency must be a real scalar.")
    if not np.isfinite(label_frequency) or not 0.0 < float(label_frequency) <= 1.0:
        raise ValueError(f"label_frequency must be in (0, 1]; got {label_frequency}.")
    if isinstance(strength, bool) or not np.isscalar(strength):
        raise TypeError("strength must be a real scalar.")
    if not np.isfinite(strength) or float(strength) < 0:
        raise ValueError(f"strength must be finite and >= 0; got {strength}.")


def _standardized_projection(
    X: np.ndarray,
    positive_mask: np.ndarray,
    feature_weights: np.ndarray | None,
) -> np.ndarray:
    positive = X[positive_mask]
    mean = positive.mean(axis=0)
    scale = positive.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (X - mean) / scale

    if feature_weights is None:
        weights = np.ones(X.shape[1], dtype=float)
    else:
        weights = np.asarray(feature_weights, dtype=float)
        if weights.ndim != 1 or len(weights) != X.shape[1]:
            raise ValueError(
                "feature_weights must be one-dimensional with length "
                f"n_features={X.shape[1]}; got shape {weights.shape}."
            )
        if not np.isfinite(weights).all():
            raise ValueError("feature_weights must contain only finite values.")
    norm = np.linalg.norm(weights)
    if norm <= 1e-12:
        raise ValueError("feature_weights must contain at least one non-zero value.")
    projection = standardized @ (weights / norm)
    return projection - projection[positive_mask].mean()


def _calibrated_intercept(scores: np.ndarray, target: float) -> float:
    """Find an intercept such that ``mean(sigmoid(scores + b)) == target``."""
    lower, upper = -80.0, 80.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if expit(scores + midpoint).mean() < target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def make_sar_propensity(
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    mechanism: str = "linear",
    label_frequency: float = 0.5,
    strength: float = 1.0,
    feature_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Construct a calibrated SCAR or SAR labeling propensity.

    The positive-class propensity is calibrated so that
    ``propensity[y_true == 1].mean()`` equals ``label_frequency`` up to
    floating-point precision. True negatives receive propensity zero.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite feature matrix.
    y_true : array-like of shape (n_samples,)
        Hidden binary labels in ``{0, 1}``.
    mechanism : {"scar", "linear", "nonlinear"}, default="linear"
        ``scar`` uses a constant propensity. ``linear`` applies a sigmoid to
        a standardized feature projection. ``nonlinear`` adds a centered
        quadratic term to that projection.
    label_frequency : float, default=0.5
        Target mean propensity among true positives. Must be in ``(0, 1]``.
    strength : float, default=1.0
        Strength of feature dependence. Zero collapses linear/nonlinear SAR
        to a constant propensity.
    feature_weights : array-like of shape (n_features,), optional
        Projection direction. The default gives all features equal weight.

    Returns
    -------
    propensity : np.ndarray of shape (n_samples,)
        ``P(S=1 | Y, X)``. Values are finite and in ``[0, 1]``; entries for
        true negatives are zero.
    """
    X_array, y_array = _validate_x_y(X, y_true)
    normalized_mechanism = _normalize_mechanism(mechanism)
    _validate_rate_and_strength(label_frequency, strength)
    positive_mask = y_array == 1
    propensity = np.zeros(len(y_array), dtype=float)
    if not positive_mask.any():
        return propensity

    target = float(label_frequency)
    if normalized_mechanism == "scar" or float(strength) == 0.0:
        propensity[positive_mask] = target
        return propensity

    projection = _standardized_projection(X_array, positive_mask, feature_weights)
    if normalized_mechanism == "linear":
        raw_scores = float(strength) * projection
    else:
        quadratic = projection**2
        quadratic -= quadratic[positive_mask].mean()
        raw_scores = float(strength) * (projection + 0.5 * quadratic)

    if target == 1.0:
        propensity[positive_mask] = 1.0
        return propensity
    intercept = _calibrated_intercept(raw_scores[positive_mask], target)
    propensity[positive_mask] = expit(raw_scores[positive_mask] + intercept)
    return propensity


def make_sar_labels(
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    mechanism: str = "linear",
    label_frequency: float = 0.5,
    strength: float = 1.0,
    feature_weights: np.ndarray | None = None,
    ensure_labeled: bool = True,
    random_state: int | np.random.RandomState | None = None,
    return_propensity: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Sample canonical PU labels from a SCAR/SAR propensity.

    Sampling follows ``S_i ~ Bernoulli(propensity_i)``. When
    ``ensure_labeled=True`` and Bernoulli sampling selects no positives, the
    highest-propensity true positive is labeled. This practical guard keeps
    downstream PU estimators usable and is most relevant for tiny datasets or
    very small ``label_frequency``.

    Parameters are shared with :func:`make_sar_propensity`.

    Parameters
    ----------
    ensure_labeled : bool, default=True
        Guarantee at least one observed positive when true positives exist.
    random_state : int, np.random.RandomState or None, optional
        Random state controlling Bernoulli draws.
    return_propensity : bool, default=False
        Return ``(y_pu, propensity)`` instead of only ``y_pu``.

    Returns
    -------
    y_pu : np.ndarray of shape (n_samples,)
        Canonical labels with ``1`` for observed positives and ``0`` for U.
    propensity : np.ndarray of shape (n_samples,), optional
        Returned only when ``return_propensity=True``.
    """
    X_array, y_array = _validate_x_y(X, y_true)
    if not isinstance(ensure_labeled, bool | np.bool_):
        raise TypeError("ensure_labeled must be a boolean.")
    if not isinstance(return_propensity, bool | np.bool_):
        raise TypeError("return_propensity must be a boolean.")
    propensity = make_sar_propensity(
        X_array,
        y_array,
        mechanism=mechanism,
        label_frequency=label_frequency,
        strength=strength,
        feature_weights=feature_weights,
    )
    rng = check_random_state(random_state)
    y_pu = np.full(len(y_array), UNLABELED_LABEL, dtype=int)
    selected = rng.uniform(size=len(y_array)) < propensity
    y_pu[selected] = POSITIVE_LABEL

    positive_indices = np.flatnonzero(y_array == 1)
    if ensure_labeled and len(positive_indices) and not np.any(y_pu == POSITIVE_LABEL):
        best = positive_indices[np.argmax(propensity[positive_indices])]
        y_pu[best] = POSITIVE_LABEL
    if return_propensity:
        return y_pu, propensity
    return y_pu


def make_sar_dataset(
    n_samples: int = 1000,
    *,
    n_features: int = 5,
    class_prior: float = 0.3,
    separation: float = 2.0,
    mechanism: str = "linear",
    label_frequency: float = 0.5,
    strength: float = 1.0,
    feature_weights: np.ndarray | None = None,
    ensure_labeled: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a Gaussian binary dataset with observable SCAR/SAR labels.

    The class count is deterministic: ``round(n_samples * class_prior)``
    positives are generated and the rows are shuffled. Positive and negative
    features have means ``+separation/2`` and ``-separation/2`` respectively
    in every feature, with identity covariance.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Generated feature matrix.
    y_pu : np.ndarray of shape (n_samples,)
        Observable canonical PU labels.
    y_true : np.ndarray of shape (n_samples,)
        Hidden binary labels retained for benchmark evaluation.
    propensity : np.ndarray of shape (n_samples,)
        Ground-truth ``P(S=1 | Y, X)`` retained for diagnostics.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int | np.integer):
        raise TypeError("n_samples must be an integer.")
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2; got {n_samples}.")
    if isinstance(n_features, bool) or not isinstance(n_features, int | np.integer):
        raise TypeError("n_features must be an integer.")
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1; got {n_features}.")
    if isinstance(class_prior, bool) or not np.isscalar(class_prior):
        raise TypeError("class_prior must be a real scalar.")
    check_scalar_in_range(float(class_prior), 0.0, 1.0, "class_prior", inclusive=False)
    if isinstance(separation, bool) or not np.isscalar(separation):
        raise TypeError("separation must be a real scalar.")
    if not np.isfinite(separation) or float(separation) < 0:
        raise ValueError(f"separation must be finite and >= 0; got {separation}.")

    rng = check_random_state(random_state)
    n_positive = int(np.clip(round(n_samples * float(class_prior)), 1, n_samples - 1))
    y_true = np.r_[
        np.ones(n_positive, dtype=int),
        np.zeros(n_samples - n_positive, dtype=int),
    ]
    rng.shuffle(y_true)
    location = np.where(y_true[:, None] == 1, separation / 2.0, -separation / 2.0)
    X = rng.normal(loc=location, scale=1.0, size=(n_samples, n_features))
    y_pu, propensity = make_sar_labels(
        X,
        y_true,
        mechanism=mechanism,
        label_frequency=label_frequency,
        strength=strength,
        feature_weights=feature_weights,
        ensure_labeled=ensure_labeled,
        random_state=rng,
        return_propensity=True,
    )
    return X, y_pu, y_true, propensity
