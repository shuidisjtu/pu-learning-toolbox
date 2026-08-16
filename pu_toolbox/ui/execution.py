# ruff: noqa: N803

"""UI-independent execution dispatch for one configured analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from pu_toolbox.model_selection.comparison import (
    ModelComparisonResult,
    PUModelComparator,
)
from pu_toolbox.model_selection.tuning import PUTuner, TuningResult
from pu_toolbox.progress import CancellationToken, ProgressCallback
from pu_toolbox.ui.history import append as history_append
from pu_toolbox.workflows import PUPipeline
from pu_toolbox.workflows.report import PipelineReport


@dataclass(frozen=True)
class AnalysisResult:
    """Objects required by the UI result renderer."""

    report: PipelineReport
    tuning: TuningResult | None = None
    comparison: ModelComparisonResult | None = None


def execute_analysis(
    *,
    X: np.ndarray,
    y_pu: np.ndarray,
    y_true: np.ndarray | None,
    class_prior: float | None,
    classifier: str,
    classifier_params: dict[str, Any],
    tuning_grid: dict[str, Any],
    comparison_classifiers: list[str],
    scoring: str,
    pipeline_params: dict[str, Any],
    cancellation_token: CancellationToken,
    progress_callback: ProgressCallback,
) -> AnalysisResult:
    """Execute normal, tuning, or comparison mode without touching Streamlit."""
    if comparison_classifiers:
        comparison = PUModelComparator(
            classifiers=comparison_classifiers,
            scoring=scoring,
            **pipeline_params,
        ).fit(
            X,
            y_pu,
            y_true=y_true,
            class_prior=class_prior,
            progress_callback=progress_callback,
            cancellation_token=cancellation_token,
        )
        return AnalysisResult(comparison.best_report, comparison=comparison)
    if tuning_grid:
        tuning = PUTuner(
            classifier=classifier,
            param_grid=tuning_grid,
            scoring=scoring,
            **pipeline_params,
        ).fit(
            X,
            y_pu,
            y_true=y_true,
            class_prior=class_prior,
            progress_callback=progress_callback,
            cancellation_token=cancellation_token,
        )
        return AnalysisResult(tuning.best_report, tuning=tuning)
    report = PUPipeline(
        classifier=classifier,
        classifier_params=classifier_params,
        **pipeline_params,
    ).fit_evaluate(
        X,
        y_pu,
        y_true=y_true,
        class_prior=class_prior,
        progress_callback=progress_callback,
        cancellation_token=cancellation_token,
    )
    return AnalysisResult(report)


def finalize_run(active_run, mode: str) -> tuple[dict, Any | None, str | None]:
    """Build the history entry for a finished background run and append it.

    Returns ``(entry, analysis, error_message)``: exactly one of
    ``analysis`` / ``error_message`` is non-None (cancelled counts as an
    error message). Kept here, not in app.py, so all three terminal states
    share one history write and stay unit-testable.
    """
    snapshot = active_run.snapshot()
    entry = {
        "开始时间": snapshot.started_at,
        "结束时间": datetime.now(timezone.utc).isoformat(),
        "模式": mode,
    }
    analysis: Any | None = None
    error_message: str | None = None
    try:
        analysis = active_run.future.result()
    except Exception as exc:  # noqa: BLE001 - UI error boundary
        status = "cancelled" if active_run.token.is_cancelled else "failed"
        error_message = str(exc) or "run cancelled by user"
        entry["状态"] = status
        entry["结果"] = error_message
    else:
        entry["状态"] = "completed"
        entry["结果"] = analysis.report.provenance.get("classifier", "unknown")
    history_append(entry)
    return entry, analysis, error_message
