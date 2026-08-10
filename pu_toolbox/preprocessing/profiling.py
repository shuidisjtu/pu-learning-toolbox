# ruff: noqa: N802, N803, N806

"""Lightweight data profiling for PU / PNU datasets.

These functions provide summary statistics and a lightweight diagnostic for
the labeling mechanism.  The structured Phase 4 profiler lives in
``data_profiler.py`` and builds on these backward-compatible helpers.
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
from pu_toolbox.core.validation import validate_true_binary_labels

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
        has_nan = bool(np.any(np.isnan(X.data))) if X.nnz > 0 else False
        has_inf = not np.isfinite(X.data).all() if X.nnz > 0 else False
    else:
        n_features = X.shape[1]
        has_nan = bool(np.any(np.isnan(X)))
        has_inf = bool(np.any(np.isinf(X)))

    return {
        "n_samples": n_samples,
        "n_features": n_features,
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
        has_nan = bool(np.any(np.isnan(X.data))) if X.nnz > 0 else False
        has_inf = not np.isfinite(X.data).all() if X.nnz > 0 else False
    else:
        n_features = X.shape[1]
        has_nan = bool(np.any(np.isnan(X)))
        has_inf = bool(np.any(np.isinf(X)))

    return {
        "n_samples": n_samples,
        "n_features": n_features,
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
    *,
    y_true: np.ndarray | None = None,
    cv: int = 3,
    threshold: float = _SCAR_AUC_THRESHOLD,
    random_state: int | None = 42,
) -> dict:
    """Diagnose feature dependence in the positive-labeling mechanism.

    When audited ``y_true`` is supplied, the classifier is evaluated only
    among true positives.  It then directly tests whether selection ``S`` is
    predictable from ``X`` conditional on ``Y=1``.  Without ``y_true``, the
    function retains the historical labeled-vs-unlabeled heuristic, but marks
    it as non-identifying: unlabeled data contain negatives, so high AUC alone
    cannot distinguish SAR from ordinary class separation under SCAR.

    Parameters
    ----------
    X : np.ndarray or sparse matrix of shape (n_samples, n_features)
        Feature matrix.
    y_pu : np.ndarray of shape (n_samples,)
        PU labels (any format accepted by
        :func:`~pu_toolbox.core.labels.normalize_pu_labels`).
    y_true : np.ndarray of shape (n_samples,), optional
        Audited binary class labels in ``{0, 1}``.  Labeled positives must be
        a subset of the audited positives.  Supplying this enables the direct
        positive-only diagnostic.
    cv : int, default=3
        Maximum number of stratified cross-validation folds.  It is reduced
        automatically to the size of the smaller selection class.
    threshold : float, default=0.65
        AUC above which feature dependence is flagged.
    random_state : int or None, default=42
        Controls shuffled stratified folds and logistic regression.

    Returns
    -------
    dict
        Diagnostic result with the following keys:

        - ``"separability_auc"`` (float): mean ROC AUC from 3-fold CV.
        - ``"is_observed_dependence_absent"`` (bool or None): ``True`` when
          AUC is at or below the threshold. This is a screening signal only —
          it does not establish SCAR (see ``"is_identifying"``); ``None``
          when the check is inconclusive.
        - ``"status"``: ``"plausible"``, ``"at_risk"``, or
          ``"inconclusive"``.
        - ``"evidence"``: ``"audited_positives"`` or
          ``"observed_mixture"``.
        - ``"is_identifying"``: whether the evidence directly conditions on
          true positives.
        - ``"message"`` (str): human-readable interpretation.

    Notes
    -----
    SCAR is generally not identifiable from ``(X, S)`` alone.  Treat the
    observed-mixture mode as a screening signal, not a hypothesis test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = normalize_pu_labels(np.asarray(y_pu))
    _validate_same_length(X, y, label="y_pu")
    if isinstance(cv, bool) or not isinstance(cv, int | np.integer) or cv < 2:
        raise ValueError(f"cv must be an integer >= 2; got {cv!r}.")
    if not np.isfinite(threshold) or not 0.5 < float(threshold) < 1.0:
        raise ValueError(f"threshold must be in (0.5, 1.0); got {threshold!r}.")

    evidence = "observed_mixture"
    is_identifying = False
    X_diagnostic = X
    target = y
    if y_true is not None:
        true = np.asarray(y_true)
        if true.ndim != 1:
            raise ValueError(f"y_true must be 1-D; got ndim={true.ndim}.")
        _validate_same_length(X, true, label="y_true")
        validate_true_binary_labels(true, estimator_name="y_true")
        if np.any((y == POSITIVE_LABEL) & (true != POSITIVE_LABEL)):
            raise ValueError("Every labeled positive in y_pu must be positive in y_true.")
        positive_mask = true == POSITIVE_LABEL
        X_diagnostic = X[positive_mask]
        target = y[positive_mask]
        evidence = "audited_positives"
        is_identifying = True

    counts = np.bincount(target.astype(int), minlength=2)
    n_splits = min(int(cv), int(counts.min()))
    common = {
        "threshold": float(threshold),
        "evidence": evidence,
        "is_identifying": is_identifying,
        "n_samples_evaluated": int(len(target)),
        "n_splits": max(n_splits, 0),
    }
    if n_splits < 2:
        message = (
            "SCAR diagnostic is inconclusive: at least two labeled and two "
            "unlabeled examples are required in the evaluated population."
        )
        return {
            "separability_auc": float("nan"),
            "is_observed_dependence_absent": None,
            "status": "inconclusive",
            "message": message,
            **common,
        }

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            random_state=random_state,
        ),
    )
    auc_scores = cross_val_score(
        clf,
        X_diagnostic,
        target,
        cv=splitter,
        scoring="roc_auc",
    )
    mean_auc = float(np.mean(auc_scores))

    is_plausible = mean_auc <= float(threshold)
    if is_plausible:
        message = (
            f"No strong feature dependence was detected (AUC = {mean_auc:.3f} "
            f"<= {float(threshold):.3f})."
        )
    else:
        message = (
            f"Feature dependence was detected (AUC = {mean_auc:.3f} > {float(threshold):.3f})."
        )
    if not is_identifying:
        message += (
            " This labeled-vs-unlabeled signal is non-identifying because "
            "unlabeled data mix positives and negatives; it cannot establish SAR."
        )

    return {
        "separability_auc": mean_auc,
        "is_observed_dependence_absent": is_plausible,
        "status": "plausible" if is_plausible else "at_risk",
        "message": message,
        **common,
    }
