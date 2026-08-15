"""Final pipeline report assembly and provenance normalization."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..core.base import BasePUClassifier
from ..diagnostics.report import PUDiagnosticReport
from ..preprocessing.data_profiler import ProfileIssue, PUDataProfile
from ..registry import RecommendationResult
from ._inputs import cv_provenance
from .report import CVMetric, PipelineReport, PriorInfo


def build_pipeline_report(
    *,
    profile: PUDataProfile,
    prior_info: PriorInfo,
    recommendation: RecommendationResult | None,
    cv_metrics: dict[str, CVMetric],
    classifier_name: str,
    auto_mode: bool,
    classifier_cls: type[BasePUClassifier] | None,
    skipped_candidates: list[dict[str, str]],
    y_true: np.ndarray | None,
    splitter: Any,
    n_splits: int,
    final_model: BasePUClassifier | None,
    diagnostic: PUDiagnosticReport | None,
    random_state: int | None,
    classifier_params: dict[str, Any],
) -> PipelineReport:
    """Assemble issues, provenance, CV metadata, and fitted artifacts."""
    issues: list[ProfileIssue] = list(profile.issues)
    if prior_info.degraded:
        issues.append(
            ProfileIssue(
                "prior_estimation_failed",
                "warning",
                prior_info.degraded,
                "Auto mode degraded to a no-prior run: methods that "
                "require a class prior were excluded from the candidates.",
            )
        )
    if recommendation is not None:
        issues.extend(
            ProfileIssue(
                "recommender_warning",
                "warning",
                message,
                "Review the method choice or pass an explicit classifier=.",
            )
            for message in recommendation.global_warnings
        )
    issues.extend(
        ProfileIssue(
            f"metric_{name}_unavailable",
            "info",
            metric.reason,
            "Supply the missing input or remove the metric from metrics=.",
        )
        for name, metric in cv_metrics.items()
        if not metric.available and metric.reason
    )
    prior_audit_flagged = prior_info.source == "estimated" and any(
        issue.code == "inconsistent_class_prior" for issue in profile.issues
    )
    provenance = {
        "classifier": classifier_name,
        "classifier_mode": (
            "auto" if auto_mode else "name" if classifier_cls is not None else "instance"
        ),
        "prior_source": prior_info.source,
        "prior_audit_flagged": prior_audit_flagged,
        "random_state": random_state,
        "classifier_params": parameter_provenance(classifier_params),
        "y_true_supplied": y_true is not None,
        "skipped_candidates": skipped_candidates,
    }
    return PipelineReport(
        profile=profile,
        recommendation=recommendation,
        prior=prior_info,
        cv_metrics=cv_metrics,
        cv_provenance=cv_provenance(splitter, n_splits),
        final_model=final_model,
        diagnostic=diagnostic,
        issues=tuple(issues),
        provenance=provenance,
    )


def parameter_provenance(params: dict[str, Any]) -> dict[str, Any]:
    """Represent classifier parameters without making reports unserializable."""
    recorded: dict[str, Any] = {}
    for key, value in params.items():
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            recorded[key] = repr(value)
        else:
            recorded[key] = value
    return recorded
