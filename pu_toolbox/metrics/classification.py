"""PU classification metrics.

Two categories:
- PU-only: require only PU labels (usable during training/model selection).
- Supervised: require true labels (usable when ground truth is available).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from pu_toolbox.core.config import POSITIVE_LABEL, UNLABELED_LABEL
from pu_toolbox.core.labels import normalize_pu_labels

__all__ = [
    "pu_zero_one_risk",
    "pu_recall",
    "pu_estimated_precision",
    "pu_negative_rate",
    "pu_accuracy",
    "pu_f1",
    "pu_auc_roc",
]


def pu_zero_one_risk(
    y_pu: np.ndarray,
    scores: np.ndarray,
    class_prior: float,
) -> float:
    """PU zero-one validation risk (du Plessis et al., 2015, Eq. 2).

    R = 2π · FNR_P + FPR_U − π

    where FNR_P = fraction of positives with score ≤ 0,
    FPR_U = fraction of unlabeled with score > 0.

    Parameters
    ----------
    y_pu : array-like of shape (n_samples,)
        PU labels in {+1, 0}.
    scores : array-like of shape (n_samples,)
        Decision function values (higher = more positive).
    class_prior : float
        True class prior π = P(y=1), in (0, 1).

    Returns
    -------
    float
        The estimated zero-one risk. Can be negative in finite samples.
    """
    y_pu = normalize_pu_labels(np.asarray(y_pu))
    scores = np.asarray(scores, dtype=float)
    if len(y_pu) != len(scores):
        raise ValueError(
            f"y_pu and scores must have the same length, "
            f"got {len(y_pu)} and {len(scores)}"
        )
    if not 0.0 < class_prior < 1.0:
        raise ValueError(f"class_prior must be in (0, 1), got {class_prior}")

    mask_p = y_pu == POSITIVE_LABEL
    mask_u = y_pu == UNLABELED_LABEL
    scores_p = scores[mask_p]
    scores_u = scores[mask_u]

    if len(scores_p) == 0 or len(scores_u) == 0:
        return np.inf

    fnr_p = float(np.mean(scores_p <= 0.0))
    fpr_u = float(np.mean(scores_u > 0.0))
    return 2.0 * class_prior * fnr_p + fpr_u - class_prior


def pu_recall(y_pu: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall estimated from labeled positives only.

    Computes the fraction of labeled positive samples that were correctly
    predicted as positive.  This quantity is directly observable in PU
    settings (no ground-truth labels needed).

    Parameters
    ----------
    y_pu : array-like of shape (n_samples,)
        PU labels in {+1, 0}.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (positive vs. non-positive).

    Returns
    -------
    float
        Recall on the labeled positives, in [0, 1].
    """
    y_pu = normalize_pu_labels(np.asarray(y_pu))
    y_pred = np.asarray(y_pred)
    if len(y_pu) != len(y_pred):
        raise ValueError(
            f"y_pu and y_pred must have the same length, "
            f"got {len(y_pu)} and {len(y_pred)}"
        )

    mask_p = y_pu == POSITIVE_LABEL
    if mask_p.sum() == 0:
        return 0.0

    return float(np.mean(y_pred[mask_p] == POSITIVE_LABEL))


def pu_estimated_precision(
    y_pu: np.ndarray,
    y_pred: np.ndarray,
    class_prior: float,
) -> float:
    """Estimate precision using the class prior.

    Uses the identity ``precision ≈ (π * recall) / predicted_positive_rate``
    where *recall* is estimated from labeled positives and
    *predicted_positive_rate* is the fraction of samples predicted positive.

    Parameters
    ----------
    y_pu : array-like of shape (n_samples,)
        PU labels in {+1, 0}.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (positive vs. non-positive).
    class_prior : float
        True class prior π = P(y=1), in (0, 1).

    Returns
    -------
    float
        Estimated precision, in [0, 1].  Returns 0.0 when no sample is
        predicted positive.
    """
    y_pu = normalize_pu_labels(np.asarray(y_pu))
    y_pred = np.asarray(y_pred)
    if len(y_pu) != len(y_pred):
        raise ValueError(
            f"y_pu and y_pred must have the same length, "
            f"got {len(y_pu)} and {len(y_pred)}"
        )
    if not 0.0 < class_prior < 1.0:
        raise ValueError(f"class_prior must be in (0, 1), got {class_prior}")

    recall = pu_recall(y_pu, y_pred)
    predicted_positive_rate = float(np.mean(y_pred == POSITIVE_LABEL))
    if predicted_positive_rate == 0.0:
        return 0.0

    return float(
        min(class_prior * recall / predicted_positive_rate, 1.0)
    )


def pu_negative_rate(y_pu: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of unlabeled samples predicted as negative.

    Useful for PU model diagnostics: this rate should be higher than
    ``1 - class_prior`` when the model is working properly.

    Parameters
    ----------
    y_pu : array-like of shape (n_samples,)
        PU labels in {+1, 0}.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (positive vs. non-positive).

    Returns
    -------
    float
        Negative prediction rate among unlabeled samples, in [0, 1].
        Returns 0.0 when there are no unlabeled samples.
    """
    y_pu = normalize_pu_labels(np.asarray(y_pu))
    y_pred = np.asarray(y_pred)
    if len(y_pu) != len(y_pred):
        raise ValueError(
            f"y_pu and y_pred must have the same length, "
            f"got {len(y_pu)} and {len(y_pred)}"
        )

    mask_u = y_pu == UNLABELED_LABEL
    if mask_u.sum() == 0:
        return 0.0

    return float(np.mean(y_pred[mask_u] != POSITIVE_LABEL))


def pu_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy given true binary labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels {0, 1}.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels {0, 1}.
    """
    return float(accuracy_score(y_true, y_pred))


def pu_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score for the positive class given true binary labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels {0, 1}.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels {0, 1}.
    """
    return float(f1_score(y_true, y_pred))


def pu_auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area Under the ROC Curve given true binary labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels {0, 1}.
    scores : array-like of shape (n_samples,)
        Decision function values or probabilities (higher = more positive).
    """
    return float(roc_auc_score(y_true, scores))
