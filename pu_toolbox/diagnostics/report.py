# ruff: noqa: N803, N806

"""Composable diagnostic reports for PU data and model outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse
from sklearn.exceptions import NotFittedError

from pu_toolbox.core.labels import normalize_pu_labels
from pu_toolbox.metrics import (
    pu_accuracy,
    pu_auc_roc,
    pu_estimated_precision,
    pu_f1,
    pu_negative_rate,
    pu_recall,
    pu_zero_one_risk,
)
from pu_toolbox.preprocessing import ProfileIssue, PUDataProfile, profile_pu_data

MetricBasis = Literal[
    "pu_observed",
    "class_prior_dependent",
    "supervised_oracle",
    "unavailable",
]

__all__ = [
    "DiagnosticMetric",
    "MetricBasis",
    "PUDiagnosticReport",
    "build_diagnostic_report",
]


@dataclass(frozen=True)
class DiagnosticMetric:
    """One metric together with its evidence basis and availability."""

    value: float | None
    basis: MetricBasis
    reason: str | None = None

    @property
    def available(self) -> bool:
        """Whether the metric has a finite value."""
        return self.value is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible representation."""
        return {
            "value": self.value,
            "basis": self.basis,
            "available": self.available,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PUDiagnosticReport:
    """Data, model-output, and metric diagnostics in one report."""

    data_profile: PUDataProfile
    model: dict[str, Any]
    metrics: dict[str, DiagnosticMetric]
    prediction_statistics: dict[str, Any]
    issues: tuple[ProfileIssue, ...]
    provenance: dict[str, Any]

    @property
    def has_errors(self) -> bool:
        """Whether any issue blocks reliable interpretation."""
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Whether any issue merits review."""
        return any(issue.severity == "warning" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-serializable nested dictionary."""
        payload = {
            "schema_version": "1.0",
            "data_profile": self.data_profile.to_dict(),
            "model": self.model,
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
            "prediction_statistics": self.prediction_statistics,
            "issues": [issue.to_dict() for issue in self.issues],
            "provenance": self.provenance,
        }
        return _json_safe(payload)

    def to_json(self, *, indent: int = 2) -> str:
        """Render strict JSON without non-standard NaN/Infinity values."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    def to_markdown(self) -> str:
        """Render a self-contained Markdown diagnostic report."""
        summary = self.data_profile.summary
        selection = self.data_profile.selection_diagnostic
        lines = [
            "# PU Diagnostic Report",
            "",
            "## Data",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| Samples | {summary['n_samples']} |",
            f"| Features | {summary['n_features']} |",
            f"| Labeled positives | {summary['n_positives']} |",
            f"| Unlabeled | {summary['n_unlabeled']} |",
            f"| Positive fraction | {summary['positive_fraction']:.6f} |",
            "",
            "## Selection Evidence",
            "",
            f"- Status: `{selection['status']}`",
            f"- Evidence: `{selection['evidence']}`",
            f"- Identifying: `{selection['is_identifying']}`",
            f"- AUC: {_format_value(selection['separability_auc'])}",
            f"- Interpretation: {_escape_markdown(selection['message'])}",
            "",
            "## Model",
            "",
            f"- Input mode: `{self.model['input_mode']}`",
            f"- Estimator: `{self.model['estimator_class'] or 'not supplied'}`",
            f"- Score source: `{self.model['score_source'] or 'not supplied'}`",
            "",
            "## Metrics",
            "",
            "| Metric | Value | Basis | Availability note |",
            "|---|---:|---|---|",
        ]
        for name, metric in self.metrics.items():
            lines.append(
                f"| `{name}` | {_format_value(metric.value)} | `{metric.basis}` | "
                f"{_escape_markdown(metric.reason or '')} |"
            )
        lines.extend(["", "## Issues", ""])
        if self.issues:
            lines.extend(
                f"- **{issue.severity.upper()} `{issue.code}`**: "
                f"{_escape_markdown(issue.message)} "
                f"Action: {_escape_markdown(issue.action)}"
                for issue in self.issues
            )
        else:
            lines.append("No issues detected by the configured checks.")
        lines.extend(["", "## Interpretation Boundaries", ""])
        lines.extend(f"- {_escape_markdown(hint)}" for hint in self.data_profile.assumption_hints)
        return "\n".join(lines) + "\n"

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["json", "markdown"] | None = None,
    ) -> Path:
        """Write JSON or Markdown, inferring the format from the suffix."""
        destination = Path(path)
        output_format = format or _format_from_suffix(destination)
        if output_format == "json":
            content = self.to_json() + "\n"
        elif output_format == "markdown":
            content = self.to_markdown()
        else:
            raise ValueError("format must be 'json' or 'markdown'.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return destination


def _metric(
    value: float | None,
    basis: MetricBasis,
    reason: str | None = None,
) -> DiagnosticMetric:
    if value is not None and not np.isfinite(value):
        return DiagnosticMetric(None, "unavailable", "The computed value is non-finite.")
    return DiagnosticMetric(None if value is None else float(value), basis, reason)


def _unavailable(reason: str) -> DiagnosticMetric:
    return DiagnosticMetric(None, "unavailable", reason)


def _validate_vector(
    values: Any,
    *,
    name: str,
    n_samples: int,
    dtype: type | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got ndim={array.ndim}.")
    if len(array) != n_samples:
        raise ValueError(f"{name} must have {n_samples} samples; got {len(array)}.")
    return array


def _extract_estimator_outputs(
    estimator: Any,
    X: Any,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray | None, str | None, dict[str, Any]]:
    if not hasattr(estimator, "predict"):
        raise TypeError("estimator must provide predict(X).")
    try:
        predictions = _validate_vector(
            estimator.predict(X),
            name="estimator predictions",
            n_samples=n_samples,
        )
        scores: np.ndarray | None = None
        score_source: str | None = None
        if hasattr(estimator, "decision_function"):
            scores = _validate_vector(
                estimator.decision_function(X),
                name="estimator decision scores",
                n_samples=n_samples,
                dtype=float,
            )
            score_source = "decision_function"
        elif hasattr(estimator, "predict_proba"):
            probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
            if probabilities.shape != (n_samples, 2):
                raise ValueError(
                    "estimator predict_proba(X) must have shape "
                    f"({n_samples}, 2); got {probabilities.shape}."
                )
            scores = probabilities[:, 1]
            score_source = "predict_proba"
    except NotFittedError as exc:
        raise ValueError("estimator must be fitted before building a report.") from exc

    metadata: dict[str, Any] = {}
    get_metadata = getattr(estimator, "get_pu_metadata", None)
    if callable(get_metadata):
        metadata = dict(get_metadata())
    return predictions, scores, score_source, metadata


def _prediction_statistics(
    predictions: np.ndarray | None,
    scores: np.ndarray | None,
) -> dict[str, Any]:
    statistics: dict[str, Any] = {
        "predicted_positive_count": None,
        "predicted_positive_fraction": None,
        "score_finite_count": None,
        "score_nonfinite_count": None,
        "score_min": None,
        "score_median": None,
        "score_max": None,
    }
    if predictions is not None:
        n_positive = int(np.sum(predictions == 1))
        statistics["predicted_positive_count"] = n_positive
        statistics["predicted_positive_fraction"] = n_positive / len(predictions)
    if scores is not None:
        finite_mask = np.isfinite(scores)
        finite_scores = scores[finite_mask]
        statistics["score_finite_count"] = int(finite_mask.sum())
        statistics["score_nonfinite_count"] = int((~finite_mask).sum())
        if len(finite_scores):
            statistics["score_min"] = float(np.min(finite_scores))
            statistics["score_median"] = float(np.median(finite_scores))
            statistics["score_max"] = float(np.max(finite_scores))
    return statistics


def _build_metrics(
    y_pu: np.ndarray,
    predictions: np.ndarray | None,
    scores: np.ndarray | None,
    y_true: np.ndarray | None,
    class_prior: float | None,
) -> dict[str, DiagnosticMetric]:
    no_predictions = "Predictions were not supplied."
    no_truth = "Audited true labels were not supplied."
    no_prior = "A class prior was not supplied."
    finite_scores = scores is not None and np.isfinite(scores).all()
    has_labeled_positives = bool(np.any(y_pu == 1))
    has_unlabeled = bool(np.any(y_pu == 0))

    if predictions is None:
        recall = negative_rate = predicted_rate = _unavailable(no_predictions)
    else:
        recall = (
            _metric(pu_recall(y_pu, predictions), "pu_observed")
            if has_labeled_positives
            else _unavailable("No labeled positives are available.")
        )
        negative_rate = (
            _metric(pu_negative_rate(y_pu, predictions), "pu_observed")
            if has_unlabeled
            else _unavailable("No unlabeled samples are available.")
        )
        predicted_rate = _metric(np.mean(predictions == 1), "pu_observed")

    if predictions is None:
        precision = risk = _unavailable(no_predictions)
    elif class_prior is None:
        precision = risk = _unavailable(no_prior)
    elif not has_labeled_positives or not has_unlabeled:
        reason = "Both labeled-positive and unlabeled groups are required."
        precision = risk = _unavailable(reason)
    else:
        precision = _metric(
            pu_estimated_precision(y_pu, predictions, class_prior),
            "class_prior_dependent",
        )
        signed_predictions = np.where(predictions == 1, 1.0, -1.0)
        risk = _metric(
            pu_zero_one_risk(y_pu, signed_predictions, class_prior),
            "class_prior_dependent",
        )

    if y_true is None:
        accuracy = f1 = auc = _unavailable(no_truth)
    else:
        accuracy = (
            _metric(pu_accuracy(y_true, predictions), "supervised_oracle")
            if predictions is not None
            else _unavailable(no_predictions)
        )
        if predictions is None:
            f1 = _unavailable(no_predictions)
        elif not np.any(y_true == 1):
            f1 = _unavailable("F1 requires at least one true positive sample.")
        else:
            f1 = _metric(pu_f1(y_true, predictions), "supervised_oracle")
        if scores is None:
            auc = _unavailable("Scores were not supplied.")
        elif not finite_scores:
            auc = _unavailable("Scores contain non-finite values.")
        elif len(np.unique(y_true)) < 2:
            auc = _unavailable("ROC AUC requires both true classes.")
        else:
            auc = _metric(pu_auc_roc(y_true, scores), "supervised_oracle")

    return {
        "labeled_positive_recall": recall,
        "unlabeled_negative_rate": negative_rate,
        "predicted_positive_rate": predicted_rate,
        "pu_estimated_precision": precision,
        "pu_zero_one_risk": risk,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": auc,
    }


def build_diagnostic_report(
    X: Any,
    y_pu: np.ndarray,
    *,
    estimator: Any | None = None,
    y_pred: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    class_prior: float | None = None,
    random_state: int | None = 42,
    **profile_kwargs: Any,
) -> PUDiagnosticReport:
    """Build a diagnostic report without fitting or mutating an estimator.

    ``estimator`` mode and explicit ``y_pred``/``scores`` mode are mutually
    exclusive. A data-only report is valid. Metrics that need unavailable
    inputs remain present with ``basis='unavailable'`` and an explicit reason.
    ``y_true`` is used only for audited selection evidence and oracle metrics.
    """
    if estimator is not None and (y_pred is not None or scores is not None):
        raise ValueError("estimator cannot be combined with explicit y_pred or scores.")
    if sparse.issparse(X):
        n_samples = X.shape[0]
    else:
        X_array = np.asarray(X)
        n_samples = X_array.shape[0] if X_array.ndim >= 1 else 0

    profile = profile_pu_data(
        X,
        y_pu,
        y_true=y_true,
        class_prior=class_prior,
        random_state=random_state,
        **profile_kwargs,
    )
    canonical_y = normalize_pu_labels(np.asarray(y_pu))
    audited_truth = None
    if y_true is not None:
        audited_truth = _validate_vector(
            y_true,
            name="y_true",
            n_samples=n_samples,
        ).astype(int, copy=False)

    metadata: dict[str, Any] = {}
    score_source: str | None = None
    if estimator is not None:
        predictions, score_values, score_source, metadata = _extract_estimator_outputs(
            estimator, X, n_samples
        )
        input_mode = "estimator"
        estimator_class = type(estimator).__name__
    else:
        predictions = (
            _validate_vector(y_pred, name="y_pred", n_samples=n_samples)
            if y_pred is not None
            else None
        )
        score_values = (
            _validate_vector(
                scores,
                name="scores",
                n_samples=n_samples,
                dtype=float,
            )
            if scores is not None
            else None
        )
        input_mode = "explicit_outputs" if y_pred is not None or scores is not None else "data_only"
        estimator_class = None
        score_source = "explicit" if scores is not None else None

    if predictions is not None:
        unique_predictions = set(np.unique(predictions))
        if not unique_predictions <= {0, 1}:
            raise ValueError(
                f"Predictions must contain only {{0, 1}}; got {sorted(unique_predictions)}."
            )
        predictions = predictions.astype(int, copy=False)

    issues = list(profile.issues)
    statistics = _prediction_statistics(predictions, score_values)
    if predictions is not None and len(np.unique(predictions)) == 1:
        issues.append(
            ProfileIssue(
                "constant_predictions",
                "warning",
                "The model predicts only one class on the evaluated data.",
                "Check thresholding, class prior, convergence, and score distribution.",
            )
        )
    if score_values is not None:
        nonfinite = int((~np.isfinite(score_values)).sum())
        if nonfinite:
            issues.append(
                ProfileIssue(
                    "nonfinite_scores",
                    "error",
                    f"Model scores contain {nonfinite} non-finite values.",
                    "Inspect preprocessing and model numerical stability.",
                )
            )
        elif len(np.unique(score_values)) == 1:
            issues.append(
                ProfileIssue(
                    "constant_scores",
                    "warning",
                    "All model scores are identical on the evaluated data.",
                    "Check whether fitting converged and whether features vary.",
                )
            )

    metrics = _build_metrics(
        canonical_y,
        predictions,
        score_values,
        audited_truth,
        class_prior,
    )
    model = {
        "input_mode": input_mode,
        "estimator_class": estimator_class,
        "score_source": score_source,
        "metadata": metadata,
    }
    provenance = {
        "n_samples": n_samples,
        "class_prior_supplied": class_prior is not None,
        "audited_truth_supplied": y_true is not None,
        "random_state": random_state,
        "profile_parameters": dict(profile_kwargs),
    }
    return PUDiagnosticReport(
        data_profile=profile,
        model=model,
        metrics=metrics,
        prediction_statistics=statistics,
        issues=tuple(issues),
        provenance=provenance,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _format_value(value: Any) -> str:
    if value is None:
        return "unavailable"
    try:
        if not np.isfinite(value):
            return "unavailable"
    except TypeError:
        return _escape_markdown(str(value))
    return f"{float(value):.6f}"


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _format_from_suffix(path: Path) -> Literal["json", "markdown"]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    raise ValueError("Cannot infer report format. Use a .json/.md suffix or pass format=.")
