# ruff: noqa: N802, N803, N806

"""Lightweight data profiling for PU / PNU datasets.

These functions provide summary statistics that feed the Advisor
(algoirthm recommender) and help users diagnose their data before
selecting an estimator.  Only basic counts and ratios are computed —
heavy feature-level analysis is deferred to the full Data Profiler
planned for Phase 4.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from pu_toolbox.core.config import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    UNLABELED_LABEL,
)
from pu_toolbox.core.labels import normalize_pnu_labels, normalize_pu_labels

__all__ = [
    "pnu_data_summary",
    "pu_data_summary",
    "scar_diagnostic",
]

# ═════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════


def _validate_same_length(
    X: np.ndarray | sparse.spmatrix,
    y: np.ndarray,
    label: str = "y",
) -> None:
    """Raise ``ValueError`` if *X* and *y* have different row counts."""
    n_x = X.shape[0]
    n_y = len(y)
    if n_x != n_y:
        raise ValueError(f"X has {n_x} samples but {label} has {n_y}.")


def _is_sparse(x: np.ndarray | sparse.spmatrix) -> bool:
    """Return ``True`` if *x* is a scipy sparse matrix."""
    return sparse.issparse(x)


# ═════════════════════════════════════════════════════════════════════
# PU data summary
# ═════════════════════════════════════════════════════════════════════


def pu_data_summary(
    X: np.ndarray | sparse.spmatrix,
    y_pu: np.ndarray,
) -> dict:
    """Compute summary statistics for a PU dataset.

    Parameters
    ----------
    X : np.ndarray or sparse matrix of shape (n_samples, n_features)
        Feature matrix.
    y_pu : np.ndarray of shape (n_samples,)
        PU labels (any format accepted by :func:`~pu_toolbox.core.labels.normalize_pu_labels`).

    Returns
    -------
    dict
        Summary with the following keys:

        - ``"n_samples"`` (int): total number of samples.
        - ``"n_features"`` (int): number of features.
        - ``"n_positives"`` (int): labeled positive samples.
        - ``"n_unlabeled"`` (int): unlabeled samples.
        - ``"pu_ratio"`` (float): unlabeled-to-positive ratio.
          ``inf`` when there are no positives.
        - ``"positive_fraction"`` (float): ``n_positives / n_samples``.
        - ``"is_sparse"`` (bool): whether *X* is a scipy sparse matrix.
        - ``"has_nan"`` (bool): whether *X* contains any NaN values.
        - ``"has_inf"`` (bool): whether *X* contains any infinite values.
        - ``"n_features_out"`` (int): ``X.shape[1]`` (alias for
          ``n_features``; both keys are present for discoverability).

    Raises
    ------
    ValueError
        If *X* and *y_pu* have different row counts.
    """
    y = normalize_pu_labels(np.asarray(y_pu))
    _validate_same_length(X, y, label="y_pu")

    n_pos = int(np.sum(y == POSITIVE_LABEL))
    n_unl = int(np.sum(y == UNLABELED_LABEL))
    n_samples = len(y)

    pu_ratio = n_unl / n_pos if n_pos > 0 else float("inf")

    if _is_sparse(X):
        n_features = X.shape[1]
        has_nan = False  # scipy sparse doesn't store explicit NaN
        has_inf = not np.isfinite(X.data).all() if X.nnz > 0 else False
    else:
        n_features = X.shape[1]
        has_nan = bool(np.any(np.isnan(X)))
        has_inf = bool(np.any(np.isinf(X)))

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_features_out": n_features,
        "n_positives": n_pos,
        "n_unlabeled": n_unl,
        "pu_ratio": pu_ratio,
        "positive_fraction": n_pos / n_samples if n_samples > 0 else 0.0,
        "is_sparse": _is_sparse(X),
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


# ═════════════════════════════════════════════════════════════════════
# PNU data summary
# ═════════════════════════════════════════════════════════════════════


def pnu_data_summary(
    X: np.ndarray | sparse.spmatrix,
    y_pnu: np.ndarray,
) -> dict:
    """Compute summary statistics for a PNU dataset.

    Parameters
    ----------
    X : np.ndarray or sparse matrix of shape (n_samples, n_features)
        Feature matrix.
    y_pnu : np.ndarray of shape (n_samples,)
        PNU labels in ``{+1, -1, 0}`` or ``{1, -1, 0}`` format.

    Returns
    -------
    dict
        Summary with all the keys from :func:`pu_data_summary` plus:

        - ``"n_negatives"`` (int): labeled negative samples.
        - ``"nu_ratio"`` (float): unlabeled-to-negative ratio.
        - ``"pn_ratio"`` (float): positive-to-negative ratio.

    Raises
    ------
    ValueError
        If *X* and *y_pnu* have different row counts.
    """
    y = normalize_pnu_labels(np.asarray(y_pnu))
    _validate_same_length(X, y, label="y_pnu")

    n_pos = int(np.sum(y == POSITIVE_LABEL))
    n_neg = int(np.sum(y == NEGATIVE_LABEL))
    n_unl = int(np.sum(y == UNLABELED_LABEL))
    n_samples = len(y)

    pu_ratio = n_unl / n_pos if n_pos > 0 else float("inf")
    nu_ratio = n_unl / n_neg if n_neg > 0 else float("inf")
    pn_ratio = n_pos / n_neg if n_neg > 0 else float("inf")

    if _is_sparse(X):
        n_features = X.shape[1]
        has_nan = False
        has_inf = not np.isfinite(X.data).all() if X.nnz > 0 else False
    else:
        n_features = X.shape[1]
        has_nan = bool(np.any(np.isnan(X)))
        has_inf = bool(np.any(np.isinf(X)))

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_features_out": n_features,
        "n_positives": n_pos,
        "n_unlabeled": n_unl,
        "n_negatives": n_neg,
        "pu_ratio": pu_ratio,
        "nu_ratio": nu_ratio,
        "pn_ratio": pn_ratio,
        "positive_fraction": n_pos / n_samples if n_samples > 0 else 0.0,
        "is_sparse": _is_sparse(X),
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


# ═════════════════════════════════════════════════════════════════════
# SCAR assumption diagnostic
# ═════════════════════════════════════════════════════════════════════

_SCAR_AUC_THRESHOLD: float = 0.65


def scar_diagnostic(
    X: np.ndarray | sparse.spmatrix,
    y_pu: np.ndarray,
) -> dict:
    """Quick diagnostic for the SCAR (Selected Completely At Random) assumption.

    Trains a lightweight logistic regression to separate labeled positives
    (``y_pu == 1``) from unlabeled samples (``y_pu == 0``) using 3-fold
    cross-validation.  If these two groups are easily separable
    (AUC >> 0.5), the labeling mechanism likely depends on features,
    violating the SCAR assumption.

    Parameters
    ----------
    X : np.ndarray or sparse matrix of shape (n_samples, n_features)
        Feature matrix.
    y_pu : np.ndarray of shape (n_samples,)
        PU labels (any format accepted by
        :func:`~pu_toolbox.core.labels.normalize_pu_labels`).

    Returns
    -------
    dict
        Diagnostic result with the following keys:

        - ``"separability_auc"`` (float): mean ROC AUC from 3-fold CV.
        - ``"is_scar_plausible"`` (bool): ``True`` when AUC ≤ threshold.
        - ``"message"`` (str): human-readable interpretation.

    Notes
    -----
    The default threshold is 0.65.  An AUC above this value suggests
    that the labeling mechanism is feature-dependent and SCAR may not
    hold.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    y = normalize_pu_labels(np.asarray(y_pu))
    _validate_same_length(X, y, label="y_pu")

    # Binary target: labeled-P = 1, U = 0 (already canonical)
    clf = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
    auc_scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
    mean_auc = float(np.mean(auc_scores))

    is_plausible = mean_auc <= _SCAR_AUC_THRESHOLD
    if is_plausible:
        message = (
            f"SCAR assumption is plausible (AUC = {mean_auc:.3f} "
            f"<= {_SCAR_AUC_THRESHOLD})."
        )
    else:
        message = (
            f"SCAR assumption may NOT hold (AUC = {mean_auc:.3f} "
            f"> {_SCAR_AUC_THRESHOLD}). The labeling mechanism appears "
            f"feature-dependent."
        )

    return {
        "separability_auc": mean_auc,
        "is_scar_plausible": is_plausible,
        "message": message,
    }
