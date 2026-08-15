# ruff: noqa: N803

"""PU-aware comparison of multiple registered classifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..workflows.pipeline import DEFAULT_METRICS, PipelineError, PUPipeline
from ..workflows.report import PipelineReport

__all__ = ["ModelComparisonResult", "ModelComparisonTrial", "PUModelComparator"]


@dataclass(frozen=True)
class ModelComparisonTrial:
    """CV result for one classifier in a comparison."""

    classifier: str
    score: float | None
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier": self.classifier,
            "score": self.score,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class ModelComparisonResult:
    """Best classifier, all CV summaries, and the fitted best report."""

    scoring: str
    higher_is_better: bool
    best_classifier: str
    best_score: float
    trials: tuple[ModelComparisonTrial, ...]
    best_report: PipelineReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "scoring": self.scoring,
            "higher_is_better": self.higher_is_better,
            "best_classifier": self.best_classifier,
            "best_score": self.best_score,
            "trials": [trial.to_dict() for trial in self.trials],
        }


class PUModelComparator:
    """Compare registered classifiers under identical PU-aware CV settings."""

    def __init__(
        self,
        *,
        classifiers: Sequence[str],
        classifier_params: Mapping[str, Mapping[str, Any]] | None = None,
        scoring: str = "pu_zero_one_risk",
        higher_is_better: bool | None = None,
        metrics: Sequence[str] | None = None,
        **pipeline_params: Any,
    ) -> None:
        names = list(classifiers)
        if len(names) < 2:
            raise ValueError("PUModelComparator requires at least two classifiers.")
        if any(not isinstance(name, str) or not name or name == "auto" for name in names):
            raise ValueError("comparison classifiers must be explicit registered names.")
        if len(set(names)) != len(names):
            raise ValueError("comparison classifiers must be unique.")
        forbidden = {"classifier", "classifier_params", "metrics"}.intersection(pipeline_params)
        if forbidden:
            raise ValueError(f"pipeline_params cannot override {sorted(forbidden)}.")
        params = {name: dict(values) for name, values in (classifier_params or {}).items()}
        unknown = sorted(set(params) - set(names))
        if unknown:
            raise ValueError(f"classifier_params contains unselected classifiers: {unknown}.")

        metric_names = list(metrics or DEFAULT_METRICS)
        if scoring not in metric_names:
            metric_names.append(scoring)
        # Construct every pipeline now so unknown methods or invalid parameters
        # fail before a potentially expensive comparison starts.
        for name in names:
            PUPipeline(
                classifier=name,
                classifier_params=params.get(name, {}),
                metrics=[scoring],
                **pipeline_params,
            )
        validator = PUPipeline(
            classifier=names[0],
            classifier_params=params.get(names[0], {}),
            metrics=[scoring],
            **pipeline_params,
        )
        self.classifiers = tuple(names)
        self.classifier_params = params
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
    ) -> ModelComparisonResult:
        """Evaluate all classifiers and fully refit only the best one."""
        trials: list[ModelComparisonTrial] = []
        successful: list[tuple[float, str]] = []
        for name in self.classifiers:
            try:
                report = self._pipeline(name).fit_evaluate(
                    X, y_pu, y_true=y_true, class_prior=class_prior, refit=False
                )
                metric = report.cv_metrics[self.scoring]
                if not metric.available or metric.mean is None or not np.isfinite(metric.mean):
                    reason = metric.reason or "metric produced no finite fold values"
                    trials.append(ModelComparisonTrial(name, None, "unavailable", reason))
                    continue
                score = float(metric.mean)
                trials.append(ModelComparisonTrial(name, score, "ok"))
                successful.append((score, name))
            except Exception as exc:  # noqa: BLE001 - isolate one unavailable backend/model
                trials.append(ModelComparisonTrial(name, None, "failed", str(exc)))
        if not successful:
            details = "; ".join(
                f"{trial.classifier}: {trial.error}" for trial in trials if trial.error
            )
            raise PipelineError(
                f"No compared classifier produced scoring metric {self.scoring!r}. {details}"
            )
        best_score, best_classifier = (
            max(successful, key=lambda item: item[0])
            if self.higher_is_better
            else min(successful, key=lambda item: item[0])
        )
        best_report = self._pipeline(best_classifier).fit_evaluate(
            X, y_pu, y_true=y_true, class_prior=class_prior
        )
        return ModelComparisonResult(
            scoring=self.scoring,
            higher_is_better=self.higher_is_better,
            best_classifier=best_classifier,
            best_score=best_score,
            trials=tuple(trials),
            best_report=best_report,
        )

    def _pipeline(self, classifier: str) -> PUPipeline:
        return PUPipeline(
            classifier=classifier,
            classifier_params=self.classifier_params.get(classifier, {}),
            metrics=self.metrics,
            **self.pipeline_params,
        )
