# ruff: noqa: N803

"""Metric resolution and PU-aware cross-validation execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..core.base import BasePUClassifier
from ..metrics.classification import (
    average_precision,
    balanced_accuracy,
    brier_score,
    expected_calibration_error,
    pu_accuracy,
    pu_auc_roc,
    pu_estimated_precision,
    pu_f1,
    pu_negative_rate,
    pu_recall,
    pu_zero_one_risk,
)
from ..progress import CancellationToken, ProgressCallback, emit_progress
from .report import CVMetric

DEFAULT_METRICS: tuple[str, ...] = (
    "pu_zero_one_risk",
    "pu_recall",
    "pu_estimated_precision",
    "pu_auc_roc",
)

_METRIC_ALIASES = {
    "pu_risk": "pu_zero_one_risk",
    "risk": "pu_zero_one_risk",
    "auc": "pu_auc_roc",
    "roc_auc": "pu_auc_roc",
    "recall": "pu_recall",
    "precision": "pu_estimated_precision",
    "accuracy": "pu_accuracy",
    "f1": "pu_f1",
    "negative_rate": "pu_negative_rate",
    "ap": "average_precision",
    "bacc": "balanced_accuracy",
    "brier": "brier_score",
    "ece": "expected_calibration_error",
}

# Spec tuples: (needs_scores, needs_prior, needs_y_true, needs_proba, basis)
_METRIC_SPECS = {
    "pu_zero_one_risk": (False, True, False, False, "class_prior_dependent"),
    "pu_recall": (False, False, False, False, "pu_observed"),
    "pu_estimated_precision": (False, True, False, False, "class_prior_dependent"),
    "pu_auc_roc": (True, False, True, False, "supervised_oracle"),
    "pu_accuracy": (False, False, True, False, "supervised_oracle"),
    "pu_f1": (False, False, True, False, "supervised_oracle"),
    "pu_negative_rate": (False, False, False, False, "pu_observed"),
    "average_precision": (True, False, True, False, "supervised_oracle"),
    "balanced_accuracy": (False, False, True, False, "supervised_oracle"),
    "brier_score": (False, False, True, True, "probability_calibration"),
    "expected_calibration_error": (False, False, True, True, "probability_calibration"),
}


def resolve_metric_names(metrics: Sequence[str] | None) -> list[str]:
    if metrics is None:
        return list(DEFAULT_METRICS)
    resolved: list[str] = []
    for name in metrics:
        canonical = _METRIC_ALIASES.get(name, name)
        if canonical not in _METRIC_SPECS:
            raise ValueError(
                f"Unknown metric {name!r}. Available: "
                + ", ".join(sorted(_METRIC_SPECS))
                + f" (aliases: {sorted(_METRIC_ALIASES)})"
            )
        if canonical not in resolved:
            resolved.append(canonical)
    if not resolved:
        raise ValueError("metrics must contain at least one metric name.")
    return resolved


def run_cross_validation(
    *,
    X: Any,
    y_pu: np.ndarray,
    splitter: Any,
    n_splits: int,
    metrics: list[str],
    evaluate_fold: Callable[[np.ndarray, np.ndarray], dict[str, tuple[float | None, str | None]]],
    total_steps: int,
    progress_callback: ProgressCallback | None,
    cancellation_token: CancellationToken | None,
) -> dict[str, CVMetric]:
    """Execute all folds and aggregate stable metric records."""
    per_fold: dict[str, list[float | None]] = {name: [] for name in metrics}
    fold_reasons: dict[str, list[str]] = {name: [] for name in metrics}
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, y_pu), start=1):
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        outcomes = evaluate_fold(train_idx, test_idx)
        for name in metrics:
            value, reason = outcomes[name]
            per_fold[name].append(value)
            if reason is not None:
                fold_reasons[name].append(reason)
        emit_progress(
            progress_callback,
            stage="cross_validation",
            completed=2 + fold_index,
            total=total_steps,
            message=f"交叉验证 {fold_index}/{n_splits}",
        )
    return {
        name: CVMetric(
            name=name,
            per_fold=tuple(per_fold[name]),
            basis=_METRIC_SPECS[name][4],
            reason=_aggregate_reason(
                fold_reasons[name], all(value is None for value in per_fold[name])
            ),
        )
        for name in metrics
    }


def extract_scores(clf: BasePUClassifier, X: Any) -> np.ndarray | None:
    if hasattr(clf, "decision_function"):
        try:
            return np.asarray(clf.decision_function(X), dtype=float)
        except Exception:  # noqa: BLE001 - scores are best-effort
            pass
    if hasattr(clf, "predict_proba"):
        try:
            proba = np.asarray(clf.predict_proba(X), dtype=float)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
        except Exception:  # noqa: BLE001 - scores are best-effort
            pass
    return None


def extract_proba(clf: BasePUClassifier, X: Any) -> np.ndarray | None:
    """Positive-class probability from a genuine ``predict_proba``.

    Never falls back to decision_function scores: calibration metrics
    require true probabilities (contract §3).
    """
    proba_fn = getattr(clf, "predict_proba", None)
    if proba_fn is None:
        return None
    try:
        proba = np.asarray(proba_fn(X), dtype=float)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
    except Exception:  # noqa: BLE001 - probability is best-effort
        return None
    return None


def compute_metric(
    name: str,
    y_pu_fold: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray | None,
    y_true_fold: np.ndarray | None,
    prior: float | None,
    proba: np.ndarray | None = None,
) -> tuple[float | None, str | None]:
    needs_scores, needs_prior, needs_y_true, needs_proba, _ = _METRIC_SPECS[name]
    if needs_scores and scores is None:
        return None, "score-based metric requires a decision function"
    if needs_prior and prior is None:
        return None, "class-prior-dependent metric requires a class prior"
    if needs_y_true and y_true_fold is None:
        return None, "supervised-oracle metric requires y_true"
    if needs_proba and proba is None:
        return None, "probabilistic metric requires predict_proba"
    try:
        if name == "pu_zero_one_risk":
            # Zero-one risk evaluates the classifier's actual binary decision.
            # Raw decision scores are not guaranteed to use zero as their
            # prediction threshold (Elkan-Noto returns probability-scale scores).
            return pu_zero_one_risk(y_pu_fold, pred, prior), None
        if name == "pu_recall":
            return pu_recall(y_pu_fold, pred), None
        if name == "pu_estimated_precision":
            return pu_estimated_precision(y_pu_fold, pred, prior), None
        if name == "pu_auc_roc":
            return pu_auc_roc(y_true_fold, scores), None
        if name == "pu_accuracy":
            return pu_accuracy(y_true_fold, pred), None
        if name == "pu_f1":
            return pu_f1(y_true_fold, pred), None
        if name == "pu_negative_rate":
            return pu_negative_rate(y_pu_fold, pred), None
        if name == "average_precision":
            return average_precision(y_true_fold, scores), None
        if name == "balanced_accuracy":
            return balanced_accuracy(y_true_fold, pred), None
        if name == "brier_score":
            return brier_score(y_true_fold, proba), None
        if name == "expected_calibration_error":
            return expected_calibration_error(y_true_fold, proba), None
    except ValueError as exc:
        return None, f"fold metric failed: {exc}"
    raise AssertionError(f"Unreachable metric {name!r}.")


def _aggregate_reason(reasons: list[str], all_skipped: bool) -> str | None:
    return reasons[0] if all_skipped and reasons else None
