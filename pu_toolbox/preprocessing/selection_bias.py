# ruff: noqa: N803, N806

"""Calibrated labeling propensities for OS simulations.

These probabilities represent ``P(S=1 | Y, X)`` and are zero for true negatives.
The OS constructors turn them into observations and separately retain audit truth.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import expit

from pu_toolbox.core.validation import validate_true_binary_labels

SARMechanism = Literal["scar", "linear", "nonlinear"]
SAR_MECHANISMS: tuple[SARMechanism, ...] = ("scar", "linear", "nonlinear")

__all__ = [
    "SAR_MECHANISMS",
    "SARMechanism",
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
    validate_true_binary_labels(y_array, estimator_name="y_true")
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
