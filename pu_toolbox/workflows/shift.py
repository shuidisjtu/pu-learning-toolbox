# ruff: noqa: N803, N806

"""Covariate-shift-aware orchestration built on :class:`PUPipeline`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pu_toolbox.core.exceptions import PipelineError
from pu_toolbox.core.labels import normalize_pu_labels
from pu_toolbox.core.validation import validate_true_binary_labels
from pu_toolbox.diagnostics.shift import PUShiftReport, analyze_pu_shift
from pu_toolbox.utils.serialization import escape_markdown, format_from_suffix, format_value

from ._evaluation import compute_metric, extract_scores
from .pipeline import PUPipeline
from .report import CVMetric, PipelineReport

__all__ = ["ShiftAwarePUPipeline", "ShiftAwarePipelineReport", "ShiftComparisonReport"]


_LOWER_IS_BETTER = {"pu_zero_one_risk"}


@dataclass(frozen=True)
class ShiftComparisonReport:
    """Paired unweighted/weighted evaluation on the same target domain."""

    shift: PUShiftReport
    baseline: ShiftAwarePipelineReport
    weighted: ShiftAwarePipelineReport | None
    metric_deltas: dict[str, dict[str, Any]]
    primary_metric: str | None
    recommendation: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "analysis_type": "shift_adaptation_comparison",
            "shift": self.shift.to_dict(),
            "baseline": self.baseline.to_dict(),
            "weighted": self.weighted.to_dict() if self.weighted is not None else None,
            "metric_deltas": self.metric_deltas,
            "primary_metric": self.primary_metric,
            "recommendation": self.recommendation,
            "provenance": self.provenance,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)

    def to_markdown(self) -> str:
        lines = [
            "# Shift Adaptation Comparison",
            "",
            f"- Recommendation: `{self.recommendation}`",
            f"- Primary metric: `{self.primary_metric or 'unavailable'}`",
            f"- Shift severity: `{self.shift.severity}`",
            f"- Weighted arm executed: `{self.weighted is not None}`",
            "",
            "## Paired Target Metrics",
            "",
            "| Metric | Baseline | Weighted | Raw delta | Improvement | Basis |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for name, values in self.metric_deltas.items():
            lines.append(
                f"| `{name}` | {format_value(values['baseline'])} | "
                f"{format_value(values['weighted'])} | {format_value(values['delta'])} | "
                f"{format_value(values['improvement'])} | `{values['basis']}` |"
            )
        lines.extend(
            [
                "",
                "## Decision Rule",
                "",
                "- Positive `improvement` always means the weighted arm is better; risk "
                "metrics have their sign reversed for this column.",
                "- Automatic selection is made only from target truth or a target class prior; "
                "PU-observed metrics alone do not authorize model selection.",
            ]
        )
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path, *, format: Literal["json", "markdown"] | None = None) -> Path:
        destination = Path(path)
        output_format = format or format_from_suffix(destination)
        if output_format == "csv":
            raise ValueError("ShiftComparisonReport supports JSON or Markdown, not CSV.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_json() + "\n" if output_format == "json" else self.to_markdown()
        destination.write_text(content, encoding="utf-8")
        return destination


@dataclass(frozen=True)
class ShiftAwarePipelineReport:
    """Combined drift audit, weighted source workflow, and target evaluation."""

    shift: PUShiftReport
    source_pipeline: PipelineReport
    target_metrics: dict[str, CVMetric]
    adaptation_applied: bool
    target_predictions: np.ndarray | None
    provenance: dict[str, Any]

    @property
    def final_model(self) -> Any:
        """Return the source pipeline's final fitted model."""
        return self.source_pipeline.final_model

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible combined report."""
        return {
            "schema_version": "1.0",
            "analysis_type": "covariate_shift_aware_pu_pipeline",
            "adaptation_applied": self.adaptation_applied,
            "shift": self.shift.to_dict(),
            "source_pipeline": self.source_pipeline.to_dict(),
            "target_metrics": {
                name: metric.to_dict() for name, metric in self.target_metrics.items()
            },
            "target_predictions": {
                "available": self.target_predictions is not None,
                "n_samples": (
                    len(self.target_predictions) if self.target_predictions is not None else 0
                ),
            },
            "provenance": self.provenance,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Render strict JSON without NaN or Infinity."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)

    def to_markdown(self) -> str:
        """Render the combined workflow report as Markdown."""
        lines = [
            "# Shift-aware PU Pipeline Report",
            "",
            f"- Adaptation applied: `{self.adaptation_applied}`",
            f"- Adaptation guarantee: `{self.provenance['guarantee']}`",
            f"- Domain OOF AUC: `{self.shift.domain_auc:.6f}`",
            f"- Shift severity: `{self.shift.severity}`",
            f"- Classifier: `{self.source_pipeline.provenance.get('classifier')}`",
            "",
            "## Source Cross-validation Metrics",
            "",
            "| Metric | Mean | Std | Basis |",
            "|---|---:|---:|---|",
        ]
        for name, metric in self.source_pipeline.cv_metrics.items():
            lines.append(
                f"| `{name}` | {format_value(metric.mean)} | {format_value(metric.std)} | "
                f"`{metric.basis}` |"
            )
        lines.extend(
            [
                "",
                "## Target Metrics",
                "",
                "| Metric | Value | Basis | Availability |",
                "|---|---:|---|---|",
            ]
        )
        for name, metric in self.target_metrics.items():
            note = (
                "available" if metric.available else escape_markdown(metric.reason or "unavailable")
            )
            lines.append(f"| `{name}` | {format_value(metric.mean)} | `{metric.basis}` | {note} |")
        lines.extend(
            [
                "",
                "## Interpretation Boundary",
                "",
                "- Adaptation uses marginal `p_target(x) / p_source(x)` weights and assumes "
                "covariate shift.",
                "- It does not implement joint `p_target(x, y) / p_source(x, y)` adaptation.",
                "- Source CV metrics and target metrics are reported separately.",
            ]
        )
        return "\n".join(lines) + "\n"

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["json", "markdown"] | None = None,
    ) -> Path:
        """Save the combined report as JSON or Markdown."""
        destination = Path(path)
        output_format = format or format_from_suffix(destination)
        if output_format == "csv":
            raise ValueError("ShiftAwarePipelineReport supports JSON or Markdown, not CSV.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_json() + "\n" if output_format == "json" else self.to_markdown()
        destination.write_text(content, encoding="utf-8")
        return destination


class ShiftAwarePUPipeline:
    """Audit drift and optionally train a covariate-weighted PU pipeline.

    Parameters
    ----------
    pipeline : PUPipeline, optional
        Configured base workflow. When omitted, one is created from
        ``classifier`` and ``pipeline_params``.
    classifier : str or BasePUClassifier, default ``"elkan_noto"``
        Classifier used only when ``pipeline`` is omitted. The default has
        effective sample-weight support.
    alpha : float, default 0.1
        Relative density-ratio parameter. Source weights are bounded before
        mean normalization by ``1 / alpha``.
    shift_cv : int, default 5
        Number of stratified folds used for out-of-fold domain AUC.
    allow_unstable : bool, default False
        Permit adaptation despite low effective sample size or boundary-heavy
        weights. Missing target PU labels can never be overridden.
    pipeline_params : keyword arguments
        Forwarded to :class:`PUPipeline` when ``pipeline`` is omitted.
    """

    def __init__(
        self,
        *,
        pipeline: PUPipeline | None = None,
        classifier: Any = "elkan_noto",
        alpha: float = 0.1,
        shift_cv: int = 5,
        allow_unstable: bool = False,
        **pipeline_params: Any,
    ) -> None:
        if pipeline is not None and (classifier != "elkan_noto" or pipeline_params):
            raise ValueError(
                "pipeline cannot be combined with classifier or pipeline constructor parameters."
            )
        if pipeline is not None and not isinstance(pipeline, PUPipeline):
            raise TypeError("pipeline must be a PUPipeline instance or None.")
        if not isinstance(allow_unstable, bool):
            raise TypeError("allow_unstable must be a bool.")
        self.pipeline = pipeline or PUPipeline(classifier=classifier, **pipeline_params)
        self.alpha = alpha
        self.shift_cv = shift_cv
        self.allow_unstable = allow_unstable

    def fit_evaluate(
        self,
        X_source: Any,
        y_source_pu: Any,
        X_target: Any,
        *,
        y_target_pu: Any | None = None,
        y_true_source: Any | None = None,
        y_true_target: Any | None = None,
        class_prior: float | None = None,
        target_class_prior: float | None = None,
        adapt: bool = True,
        refit: bool = True,
        progress_callback: Any | None = None,
        cancellation_token: Any | None = None,
    ) -> ShiftAwarePipelineReport:
        """Run drift audit, source training, and optional target evaluation."""
        if not isinstance(adapt, bool):
            raise TypeError("adapt must be a bool.")
        shift = analyze_pu_shift(
            X_source,
            y_source_pu,
            X_target,
            y_target_pu=y_target_pu,
            alpha=self.alpha,
            cv=self.shift_cv,
            random_state=self.pipeline.random_state,
        )
        if adapt and y_target_pu is None:
            raise PipelineError(
                "Covariate adaptation requires target-domain PU labels. "
                "Pass y_target_pu or call fit_evaluate(..., adapt=False) for audit plus baseline."
            )
        if adapt and not shift.adaptation_ready and not self.allow_unstable:
            raise PipelineError(
                "The shift audit found unstable importance weights (low ESS or boundary mass). "
                "Collect better-covered target data, or set allow_unstable=True to proceed "
                "with the warning preserved in the report."
            )
        sample_weight = shift.source_importance_weights if adapt else None
        source_report = self.pipeline.fit_evaluate(
            X_source,
            y_source_pu,
            y_true=y_true_source,
            class_prior=class_prior,
            sample_weight=sample_weight,
            refit=refit,
            progress_callback=progress_callback,
            cancellation_token=cancellation_token,
        )
        target_metrics, target_predictions = _target_evaluation(
            source_report,
            X_target,
            y_target_pu=y_target_pu,
            y_true_target=y_true_target,
            target_class_prior=target_class_prior,
        )
        return ShiftAwarePipelineReport(
            shift=shift,
            source_pipeline=source_report,
            target_metrics=target_metrics,
            adaptation_applied=adapt,
            target_predictions=target_predictions,
            provenance={
                "guarantee": "covariate_shift_only",
                "target_pu_used_for_density_ratio": False,
                "target_pu_required_as_adaptation_gate": adapt,
                "target_class_prior": target_class_prior,
                "allow_unstable": self.allow_unstable,
            },
        )

    def compare(
        self,
        X_source: Any,
        y_source_pu: Any,
        X_target: Any,
        *,
        y_target_pu: Any,
        y_true_source: Any | None = None,
        y_true_target: Any | None = None,
        class_prior: float | None = None,
        target_class_prior: float | None = None,
        primary_metric: str | None = None,
        min_improvement: float = 0.0,
    ) -> ShiftComparisonReport:
        """Compare unweighted and covariate-weighted arms without silent selection.

        Both arms use the same pipeline configuration and target evaluation set.
        An unstable audit skips the weighted arm unless ``allow_unstable=True``.
        """
        if not np.isfinite(min_improvement) or min_improvement < 0:
            raise ValueError("min_improvement must be a finite non-negative number.")
        shift = analyze_pu_shift(
            X_source,
            y_source_pu,
            X_target,
            y_target_pu=y_target_pu,
            alpha=self.alpha,
            cv=self.shift_cv,
            random_state=self.pipeline.random_state,
        )
        baseline = self._fit_from_shift(
            shift,
            X_source,
            y_source_pu,
            X_target,
            y_target_pu=y_target_pu,
            y_true_source=y_true_source,
            y_true_target=y_true_target,
            class_prior=class_prior,
            target_class_prior=target_class_prior,
            adapt=False,
        )
        weighted = None
        if shift.adaptation_ready or self.allow_unstable:
            weighted = self._fit_from_shift(
                shift,
                X_source,
                y_source_pu,
                X_target,
                y_target_pu=y_target_pu,
                y_true_source=y_true_source,
                y_true_target=y_true_target,
                class_prior=class_prior,
                target_class_prior=target_class_prior,
                adapt=True,
            )
        metric_deltas = _compare_target_metrics(baseline, weighted)
        chosen = _select_primary_metric(
            metric_deltas,
            requested=primary_metric,
            has_target_truth=y_true_target is not None,
            has_target_prior=target_class_prior is not None,
        )
        if weighted is None:
            recommendation = "collect_target_data"
        elif chosen is None:
            recommendation = "audit_only"
        elif metric_deltas[chosen]["improvement"] > min_improvement:
            recommendation = "reweight_recommended"
        else:
            recommendation = "no_clear_benefit"
        return ShiftComparisonReport(
            shift=shift,
            baseline=baseline,
            weighted=weighted,
            metric_deltas=metric_deltas,
            primary_metric=chosen,
            recommendation=recommendation,
            provenance={
                "comparison": "paired_same_target",
                "selection_uses_pu_observed_only": False,
                "min_improvement": float(min_improvement),
                "guarantee": "covariate_shift_only",
            },
        )

    def _fit_from_shift(
        self,
        shift: PUShiftReport,
        X_source: Any,
        y_source_pu: Any,
        X_target: Any,
        *,
        y_target_pu: Any,
        y_true_source: Any | None,
        y_true_target: Any | None,
        class_prior: float | None,
        target_class_prior: float | None,
        adapt: bool,
    ) -> ShiftAwarePipelineReport:
        source_report = self.pipeline.fit_evaluate(
            X_source,
            y_source_pu,
            y_true=y_true_source,
            class_prior=class_prior,
            sample_weight=shift.source_importance_weights if adapt else None,
        )
        target_metrics, target_predictions = _target_evaluation(
            source_report,
            X_target,
            y_target_pu=y_target_pu,
            y_true_target=y_true_target,
            target_class_prior=target_class_prior,
        )
        return ShiftAwarePipelineReport(
            shift=shift,
            source_pipeline=source_report,
            target_metrics=target_metrics,
            adaptation_applied=adapt,
            target_predictions=target_predictions,
            provenance={
                "guarantee": "covariate_shift_only",
                "target_pu_used_for_density_ratio": False,
                "target_pu_required_as_adaptation_gate": adapt,
                "target_class_prior": target_class_prior,
                "allow_unstable": self.allow_unstable,
            },
        )


def _compare_target_metrics(
    baseline: ShiftAwarePipelineReport, weighted: ShiftAwarePipelineReport | None
) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    for name, base_metric in baseline.target_metrics.items():
        weighted_metric = weighted.target_metrics.get(name) if weighted is not None else None
        base_value = base_metric.mean
        weighted_value = weighted_metric.mean if weighted_metric is not None else None
        delta = (
            float(weighted_value - base_value)
            if base_value is not None and weighted_value is not None
            else None
        )
        improvement = -delta if delta is not None and name in _LOWER_IS_BETTER else delta
        comparison[name] = {
            "baseline": base_value,
            "weighted": weighted_value,
            "delta": delta,
            "improvement": improvement,
            "basis": base_metric.basis,
            "available": improvement is not None,
        }
    return comparison


def _select_primary_metric(
    metrics: dict[str, dict[str, Any]],
    *,
    requested: str | None,
    has_target_truth: bool,
    has_target_prior: bool,
) -> str | None:
    if requested is not None:
        if requested not in metrics:
            raise ValueError(f"primary_metric {requested!r} was not requested from the pipeline.")
        if metrics[requested]["basis"] == "pu_observed":
            raise ValueError("primary_metric cannot use PU-observed evidence alone.")
        if not metrics[requested]["available"]:
            raise ValueError(f"primary_metric {requested!r} is unavailable on the target domain.")
        return requested
    if has_target_truth:
        for name in ("pu_auc_roc", "pu_f1", "pu_accuracy"):
            if name in metrics and metrics[name]["available"]:
                return name
    if has_target_prior:
        for name in ("pu_zero_one_risk", "pu_estimated_precision"):
            if name in metrics and metrics[name]["available"]:
                return name
    return None


def _target_evaluation(
    source_report: PipelineReport,
    X_target: Any,
    *,
    y_target_pu: Any | None,
    y_true_target: Any | None,
    target_class_prior: float | None,
) -> tuple[dict[str, CVMetric], np.ndarray | None]:
    results: dict[str, CVMetric] = {}
    model = source_report.final_model
    if model is None:
        for name, source_metric in source_report.cv_metrics.items():
            results[name] = CVMetric(
                name=name,
                per_fold=(None,),
                basis=source_metric.basis,
                reason="target evaluation requires refit=True",
            )
        return results, None
    predictions = np.asarray(model.predict(X_target), dtype=int)
    scores = extract_scores(model, X_target)
    if y_target_pu is None:
        for name, source_metric in source_report.cv_metrics.items():
            results[name] = CVMetric(
                name=name,
                per_fold=(None,),
                basis=source_metric.basis,
                reason="target evaluation requires y_target_pu",
            )
        return results, predictions
    labels = normalize_pu_labels(y_target_pu)
    if len(labels) != len(predictions):
        raise ValueError(f"y_target_pu has {len(labels)} rows but X_target has {len(predictions)}.")
    true_labels = _validated_target_truth(y_true_target, len(predictions))
    for name, source_metric in source_report.cv_metrics.items():
        value, reason = compute_metric(
            name,
            labels,
            predictions,
            scores,
            true_labels,
            target_class_prior,
        )
        results[name] = CVMetric(
            name=name,
            per_fold=(value,),
            basis=source_metric.basis,
            reason=reason,
        )
    return results, predictions


def _validated_target_truth(y_true: Any | None, n_samples: int) -> np.ndarray | None:
    if y_true is None:
        return None
    labels = np.asarray(y_true)
    if labels.shape != (n_samples,):
        raise ValueError(f"y_true_target must have shape ({n_samples},); got {labels.shape}.")
    validate_true_binary_labels(labels, estimator_name="y_true_target")
    return labels.astype(int, copy=False)
