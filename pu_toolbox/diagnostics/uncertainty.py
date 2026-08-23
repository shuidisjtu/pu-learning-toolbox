# ruff: noqa: N803, N806

"""Prediction uncertainty, abstention, and active-review utilities for PU models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cluster import KMeans

from pu_toolbox.core.labels import normalize_pu_labels
from pu_toolbox.core.validation import validate_true_binary_labels
from pu_toolbox.utils.serialization import format_from_suffix, json_safe

QueryStrategy = Literal["uncertainty", "shift_weighted", "diverse_uncertainty"]

__all__ = ["PUUncertaintyReport", "analyze_pu_uncertainty"]


@dataclass(frozen=True)
class PUUncertaintyReport:
    """Row-level uncertainty plus deployment and review summaries."""

    positive_probability: np.ndarray
    uncertainty: np.ndarray
    selective_predictions: np.ndarray
    query_indices: np.ndarray
    summary: dict[str, Any]
    provenance: dict[str, Any]

    def to_frame(self) -> pd.DataFrame:
        queried = np.zeros(len(self.uncertainty), dtype=bool)
        queried[self.query_indices] = True
        return pd.DataFrame(
            {
                "row": np.arange(len(self.uncertainty), dtype=int),
                "positive_probability": self.positive_probability,
                "uncertainty": self.uncertainty,
                "selective_prediction": self.selective_predictions,
                "selected_for_review": queried,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "schema_version": "1.0",
                "analysis_type": "pu_prediction_uncertainty",
                "summary": self.summary,
                "query_indices": self.query_indices.tolist(),
                "artifacts": {"row_scores": "uncertainty_rows.csv"},
                "provenance": self.provenance,
            }
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)

    def to_markdown(self) -> str:
        lines = [
            "# PU Prediction Uncertainty Report",
            "",
            f"- Coverage after abstention: `{self.summary['coverage']:.6f}`",
            f"- Abstained rows: `{self.summary['n_abstained']}`",
            f"- Review queries: `{self.summary['n_queries']}`",
            f"- Query strategy: `{self.provenance['query_strategy']}`",
        ]
        if self.summary.get("selective_accuracy") is not None:
            lines.append(
                f"- Selective accuracy (oracle): `{self.summary['selective_accuracy']:.6f}`"
            )
        lines.extend(
            [
                "",
                "## Interpretation Boundary",
                "",
                "- Margin uncertainty measures ambiguity of this model; it is not a calibrated "
                "epistemic uncertainty interval.",
                "- Abstention changes coverage and must be evaluated with downstream review cost.",
                "- Active-review rankings propose rows; they do not create labels automatically.",
            ]
        )
        return "\n".join(lines) + "\n"

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["json", "markdown", "csv"] | None = None,
    ) -> Path:
        destination = Path(path)
        output_format = format or format_from_suffix(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "json":
            destination.write_text(self.to_json() + "\n", encoding="utf-8")
        elif output_format == "markdown":
            destination.write_text(self.to_markdown(), encoding="utf-8")
        elif output_format == "csv":
            self.to_frame().to_csv(destination, index=False)
        else:
            raise ValueError("format must be 'json', 'markdown', or 'csv'.")
        return destination


def analyze_pu_uncertainty(
    estimator: Any,
    X: Any,
    *,
    y_pu: Any | None = None,
    y_true: Any | None = None,
    min_confidence: float = 0.5,
    query_budget: int = 0,
    query_strategy: QueryStrategy = "uncertainty",
    importance_weight: Any | None = None,
    random_state: int | None = 42,
) -> PUUncertaintyReport:
    """Build a selective prediction and active human-review plan.

    Labeled positives are excluded from queries when ``y_pu`` is supplied.
    ``shift_weighted`` prioritizes uncertain target-relevant rows, while
    ``diverse_uncertainty`` spreads the review budget over feature clusters.
    """
    if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must lie in [0, 1].")
    if isinstance(query_budget, bool) or not isinstance(query_budget, int) or query_budget < 0:
        raise ValueError("query_budget must be an integer >= 0.")
    if query_strategy not in {"uncertainty", "shift_weighted", "diverse_uncertainty"}:
        raise ValueError("Unknown query_strategy.")
    probability, score_source = _positive_probability(estimator, X)
    n_samples = len(probability)
    confidence = 2.0 * np.abs(probability - 0.5)
    uncertainty = 1.0 - confidence
    predictions = (probability >= 0.5).astype(int)
    selective = predictions.copy()
    selective[confidence < min_confidence] = -1
    labels = None
    if y_pu is not None:
        labels = normalize_pu_labels(y_pu)
        if labels.shape != (n_samples,):
            raise ValueError(f"y_pu must have shape ({n_samples},); got {labels.shape}.")
    candidates = np.arange(n_samples) if labels is None else np.flatnonzero(labels == 0)
    budget = min(query_budget, len(candidates))
    weights = _validated_importance_weight(importance_weight, n_samples)
    if query_strategy == "shift_weighted" and weights is None:
        raise ValueError("shift_weighted queries require importance_weight.")
    if query_strategy == "diverse_uncertainty":
        query_indices = _diverse_queries(X, candidates, uncertainty, budget, random_state)
    else:
        priority = uncertainty.copy()
        if query_strategy == "shift_weighted":
            priority *= weights / np.mean(weights)
        order = np.lexsort((candidates, -priority[candidates]))
        query_indices = candidates[order[:budget]]
    covered = selective != -1
    summary: dict[str, Any] = {
        "n_samples": n_samples,
        "n_abstained": int(np.sum(~covered)),
        "coverage": float(np.mean(covered)),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "n_query_candidates": len(candidates),
        "n_queries": len(query_indices),
        "selective_accuracy": None,
    }
    if y_true is not None:
        truth = np.asarray(y_true)
        if truth.shape != (n_samples,):
            raise ValueError(f"y_true must have shape ({n_samples},); got {truth.shape}.")
        validate_true_binary_labels(truth, estimator_name="y_true")
        summary["selective_accuracy"] = (
            float(np.mean(selective[covered] == truth[covered])) if np.any(covered) else None
        )
    return PUUncertaintyReport(
        positive_probability=probability,
        uncertainty=uncertainty,
        selective_predictions=selective,
        query_indices=np.asarray(query_indices, dtype=int),
        summary=summary,
        provenance={
            "uncertainty_method": "binary_probability_margin",
            "score_source": score_source,
            "min_confidence": float(min_confidence),
            "query_strategy": query_strategy,
            "labeled_positives_excluded_from_queries": labels is not None,
            "epistemic_uncertainty_guarantee": False,
        },
    )


def _positive_probability(estimator: Any, X: Any) -> tuple[np.ndarray, str]:
    if hasattr(estimator, "predict_proba"):
        try:
            values = np.asarray(estimator.predict_proba(X), dtype=float)
            if values.ndim == 2 and values.shape[1] == 2:
                probability = values[:, 1]
                if np.isfinite(probability).all():
                    return np.clip(probability, 0.0, 1.0), "predict_proba"
        except (AttributeError, NotImplementedError):
            pass
    if not hasattr(estimator, "decision_function"):
        raise ValueError("estimator must implement predict_proba or decision_function.")
    scores = np.asarray(estimator.decision_function(X), dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("estimator decision scores must be a finite 1-D array.")
    if np.all((scores >= 0.0) & (scores <= 1.0)):
        return scores, "decision_function_probability"
    return expit(scores), "decision_function_logit"


def _validated_importance_weight(values: Any | None, n_samples: int) -> np.ndarray | None:
    if values is None:
        return None
    weights = np.asarray(values, dtype=float)
    if weights.shape != (n_samples,):
        raise ValueError(f"importance_weight must have shape ({n_samples},); got {weights.shape}.")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("importance_weight must be finite, non-negative, and not all zero.")
    return weights


def _diverse_queries(
    X: Any,
    candidates: np.ndarray,
    uncertainty: np.ndarray,
    budget: int,
    random_state: int | None,
) -> np.ndarray:
    if budget == 0:
        return np.array([], dtype=int)
    features = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=float)
    if features.ndim == 4:
        features = features.reshape(features.shape[0], -1)
    if features.ndim != 2 or features.shape[0] != len(uncertainty):
        raise ValueError("X must be a row-aligned 2-D matrix for diverse queries.")
    pool_size = min(len(candidates), max(budget * 5, budget))
    order = np.lexsort((candidates, -uncertainty[candidates]))
    pool = candidates[order[:pool_size]]
    if budget >= len(pool):
        return pool
    clusters = KMeans(n_clusters=budget, n_init=10, random_state=random_state).fit_predict(
        features[pool]
    )
    selected = [
        int(cluster_rows[np.argmax(uncertainty[cluster_rows])])
        for cluster in range(budget)
        if len(cluster_rows := pool[clusters == cluster])
    ]
    return np.asarray(sorted(selected, key=lambda row: (-uncertainty[row], row)), dtype=int)
