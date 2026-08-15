# ruff: noqa: N803

"""Hyperparameter search built on the PU-aware workflow and metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import ParameterGrid

from ..workflows.pipeline import DEFAULT_METRICS, PipelineError, PUPipeline
from ..workflows.report import PipelineReport

__all__ = ["PUTuner", "TuningResult", "TuningTrial"]


@dataclass(frozen=True)
class TuningTrial:
    """One evaluated parameter combination."""

    index: int
    params: dict[str, Any]
    score: float | None
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly trial summary."""
        return {
            "index": self.index,
            "params": self.params,
            "score": self.score,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class TuningResult:
    """Best candidate, all trial summaries, and the fitted best report."""

    classifier: str
    scoring: str
    higher_is_better: bool
    best_params: dict[str, Any]
    best_score: float
    trials: tuple[TuningTrial, ...]
    best_report: PipelineReport

    def to_dict(self) -> dict[str, Any]:
        """Return tuning metadata without serializing the fitted estimator."""
        return {
            "classifier": self.classifier,
            "scoring": self.scoring,
            "higher_is_better": self.higher_is_better,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "trials": [trial.to_dict() for trial in self.trials],
        }


class PUTuner:
    """Exhaustive PU-aware parameter search.

    Every candidate is evaluated with :class:`PUPipeline`, so it uses
    PU-stratified folds, class-prior handling, and the same metric semantics as
    a normal run. Failed candidates are recorded and skipped; an error is
    raised only when no candidate produces the requested score.
    """

    def __init__(
        self,
        *,
        classifier: str,
        param_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Sequence[Any]]],
        scoring: str = "pu_zero_one_risk",
        higher_is_better: bool | None = None,
        metrics: Sequence[str] | None = None,
        **pipeline_params: Any,
    ) -> None:
        if not isinstance(classifier, str) or classifier == "auto":
            raise ValueError("PUTuner requires an explicit registered classifier name.")
        forbidden = {"classifier", "classifier_params", "metrics"}.intersection(pipeline_params)
        if forbidden:
            raise ValueError(f"pipeline_params cannot override {sorted(forbidden)}.")
        try:
            candidates = list(ParameterGrid(param_grid))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid param_grid: {exc}") from exc
        if not candidates:
            raise ValueError("param_grid must produce at least one candidate.")

        metric_names = list(metrics or DEFAULT_METRICS)
        if scoring not in metric_names:
            metric_names.append(scoring)
        validator = PUPipeline(
            classifier=classifier,
            classifier_params=candidates[0],
            metrics=[scoring],
            **pipeline_params,
        )
        self.classifier = classifier
        self.param_grid = candidates
        self.scoring = validator.metrics[0]
        self.metrics = metric_names
        self.higher_is_better = (
            self.scoring != "pu_zero_one_risk"
            if higher_is_better is None
            else bool(higher_is_better)
        )
        self.pipeline_params = dict(pipeline_params)

    def fit(
        self,
        X: Any,
        y_pu: np.ndarray,
        *,
        y_true: np.ndarray | None = None,
        class_prior: float | None = None,
    ) -> TuningResult:
        """Evaluate the parameter grid and return its best valid trial."""
        trials: list[TuningTrial] = []
        successful: list[tuple[float, dict[str, Any]]] = []
        for index, params in enumerate(self.param_grid):
            try:
                report = PUPipeline(
                    classifier=self.classifier,
                    classifier_params=params,
                    metrics=self.metrics,
                    **self.pipeline_params,
                ).fit_evaluate(
                    X,
                    y_pu,
                    y_true=y_true,
                    class_prior=class_prior,
                    refit=False,
                )
                metric = report.cv_metrics[self.scoring]
                if not metric.available or metric.mean is None or not np.isfinite(metric.mean):
                    reason = metric.reason or "metric produced no finite fold values"
                    trials.append(TuningTrial(index, dict(params), None, "unavailable", reason))
                    continue
                score = float(metric.mean)
                trials.append(TuningTrial(index, dict(params), score, "ok"))
                successful.append((score, dict(params)))
            # A single model/optimizer/backend failure must not discard the
            # rest of the search.  Exception deliberately excludes
            # KeyboardInterrupt and SystemExit, so users can still cancel.
            except Exception as exc:  # noqa: BLE001 - trial isolation is the API contract
                trials.append(TuningTrial(index, dict(params), None, "failed", str(exc)))

        if not successful:
            details = "; ".join(
                f"trial {trial.index}: {trial.error}" for trial in trials if trial.error
            )
            raise PipelineError(
                f"No tuning candidate produced scoring metric {self.scoring!r}. {details}"
            )
        best_score, best_params = (
            max(successful, key=lambda item: item[0])
            if self.higher_is_better
            else min(successful, key=lambda item: item[0])
        )
        # Only the selected configuration pays for a full-data fit and model
        # diagnostics. Search trials above perform CV only.
        best_report = PUPipeline(
            classifier=self.classifier,
            classifier_params=best_params,
            metrics=self.metrics,
            **self.pipeline_params,
        ).fit_evaluate(X, y_pu, y_true=y_true, class_prior=class_prior)
        return TuningResult(
            classifier=self.classifier,
            scoring=self.scoring,
            higher_is_better=self.higher_is_better,
            best_params=best_params,
            best_score=best_score,
            trials=tuple(trials),
            best_report=best_report,
        )
