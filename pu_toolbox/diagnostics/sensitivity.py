# ruff: noqa: N803

"""Assumption sensitivity analysis for PU model outputs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from pu_toolbox.core.labels import normalize_pu_labels
from pu_toolbox.metrics import pu_estimated_precision, pu_zero_one_risk

SensitivityAxis = Literal["class_prior", "label_propensity"]

__all__ = [
    "PUSensitivityAnalysis",
    "SensitivityAxis",
    "SensitivityPoint",
    "analyze_pu_sensitivity",
]


@dataclass(frozen=True)
class SensitivityPoint:
    """Metrics and algebraic consistency for one assumed parameter value."""

    axis: SensitivityAxis
    value: float
    class_prior: float
    label_propensity: float
    is_consistent: bool
    consistency_reason: str | None
    pu_estimated_precision: float | None
    pu_zero_one_risk: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible row."""
        return {
            "axis": self.axis,
            "value": self.value,
            "class_prior": self.class_prior,
            "label_propensity": self.label_propensity,
            "is_consistent": self.is_consistent,
            "consistency_reason": self.consistency_reason,
            "pu_estimated_precision": self.pu_estimated_precision,
            "pu_zero_one_risk": self.pu_zero_one_risk,
        }


@dataclass(frozen=True)
class PUSensitivityAnalysis:
    """Structured result of class-prior and labeling-propensity sweeps."""

    observed_label_rate: float
    points: tuple[SensitivityPoint, ...]
    metric_ranges: dict[str, dict[str, dict[str, float | int | None]]]
    provenance: dict[str, Any]

    @property
    def has_inconsistent_assumptions(self) -> bool:
        """Whether any grid point contradicts the observed labeling rate."""
        return any(not point.is_consistent for point in self.points)

    def to_frame(self, *, axis: SensitivityAxis | None = None) -> pd.DataFrame:
        """Return all rows, or one axis, as an independent DataFrame."""
        if axis not in {None, "class_prior", "label_propensity"}:
            raise ValueError("axis must be 'class_prior', 'label_propensity', or None.")
        rows = [point.to_dict() for point in self.points if axis is None or point.axis == axis]
        return pd.DataFrame(rows, columns=_POINT_COLUMNS)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-serializable representation."""
        return {
            "schema_version": "1.0",
            "analysis_type": "fixed_output_assumption_sensitivity",
            "observed_label_rate": self.observed_label_rate,
            "feasible_region": {
                "class_prior_lower_bound": self.observed_label_rate,
                "class_prior_upper_bound_exclusive": 1.0,
                "mean_label_propensity_lower_bound": self.observed_label_rate,
                "mean_label_propensity_upper_bound": 1.0,
            },
            "points": [point.to_dict() for point in self.points],
            "metric_ranges": self.metric_ranges,
            "provenance": self.provenance,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Render strict JSON without NaN or Infinity."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    def to_markdown(self) -> str:
        """Render a self-contained Markdown sensitivity report."""
        lines = [
            "# PU Assumption Sensitivity Analysis",
            "",
            f"- Observed labeled-positive rate: `{self.observed_label_rate:.6f}`",
            "- Analysis mode: `fixed_output_assumption_sensitivity`",
            "- Identity audited: `P(S=1) = class_prior * mean_label_propensity`",
            "",
            "| Axis | Assumed value | Class prior | Mean propensity | Consistent | "
            "Estimated precision | PU zero-one risk | Note |",
            "|---|---:|---:|---:|---|---:|---:|---|",
        ]
        for point in self.points:
            lines.append(
                f"| `{point.axis}` | {point.value:.6f} | "
                f"{point.class_prior:.6f} | {point.label_propensity:.6f} | "
                f"`{point.is_consistent}` | "
                f"{_format_value(point.pu_estimated_precision)} | "
                f"{_format_value(point.pu_zero_one_risk)} | "
                f"{point.consistency_reason or ''} |"
            )
        lines.extend(
            [
                "",
                "## Interpretation Boundary",
                "",
                "- This report perturbs assumptions while keeping model outputs fixed.",
                "- It does not estimate an unidentified propensity or retrain the model.",
                "- Refit each model under each assumption for training sensitivity claims.",
            ]
        )
        return "\n".join(lines) + "\n"

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["json", "markdown", "csv"] | None = None,
    ) -> Path:
        """Save JSON, Markdown, or CSV, inferring format from the suffix."""
        destination = Path(path)
        output_format = format or _format_from_suffix(destination)
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


_POINT_COLUMNS = [
    "axis",
    "value",
    "class_prior",
    "label_propensity",
    "is_consistent",
    "consistency_reason",
    "pu_estimated_precision",
    "pu_zero_one_risk",
]


def _validate_grid(
    values: Iterable[float] | None,
    *,
    name: str,
    include_one: bool,
) -> tuple[float, ...]:
    if values is None:
        return ()
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of numeric values.") from exc
    if not raw_values:
        raise ValueError(f"{name} cannot be empty when supplied.")
    try:
        array = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values.") from exc
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 1-D grid.")
    upper_valid = array <= 1.0 if include_one else array < 1.0
    interval = "(0, 1]" if include_one else "(0, 1)"
    if np.any(array <= 0.0) or not np.all(upper_valid):
        raise ValueError(f"{name} values must lie in {interval}.")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{name} cannot contain duplicate values.")
    return tuple(float(value) for value in array)


def _metrics_for_prior(
    y_pu: np.ndarray,
    y_pred: np.ndarray,
    risk_scores: np.ndarray,
    class_prior: float,
) -> tuple[float | None, float | None]:
    if not 0.0 < class_prior < 1.0:
        return None, None
    return (
        pu_estimated_precision(y_pu, y_pred, class_prior),
        pu_zero_one_risk(y_pu, risk_scores, class_prior),
    )


def _build_metric_ranges(
    points: tuple[SensitivityPoint, ...],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for axis in ("class_prior", "label_propensity"):
        axis_points = [point for point in points if point.axis == axis and point.is_consistent]
        result[axis] = {}
        for metric_name in ("pu_estimated_precision", "pu_zero_one_risk"):
            values = [
                value for point in axis_points if (value := getattr(point, metric_name)) is not None
            ]
            result[axis][metric_name] = {
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
                "span": max(values) - min(values) if values else None,
                "n_available": len(values),
            }
    return result


def analyze_pu_sensitivity(
    y_pu: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_priors: Iterable[float] | None = None,
    label_propensities: Iterable[float] | None = None,
    scores: np.ndarray | None = None,
) -> PUSensitivityAnalysis:
    """Audit fixed model outputs over class-prior and propensity assumptions.

    The analysis uses ``P(S=1) = pi * c_bar``, where ``c_bar`` is the mean
    positive-labeling propensity. It does not identify ``pi`` or ``c_bar``
    separately and does not refit the model.
    """
    priors = _validate_grid(class_priors, name="class_priors", include_one=False)
    propensities = _validate_grid(
        label_propensities,
        name="label_propensities",
        include_one=True,
    )
    if not priors and not propensities:
        raise ValueError("Supply class_priors, label_propensities, or both.")

    canonical_y = normalize_pu_labels(np.asarray(y_pu))
    predictions = np.asarray(y_pred)
    if len(canonical_y) == 0:
        raise ValueError("y_pu and y_pred cannot be empty.")
    if predictions.ndim != 1 or len(predictions) != len(canonical_y):
        raise ValueError("y_pred must be 1-D and have the same length as y_pu.")
    unique_predictions = set(np.unique(predictions))
    if not unique_predictions <= {0, 1}:
        raise ValueError(f"y_pred must contain only {{0, 1}}; got {sorted(unique_predictions)}.")
    if not np.any(canonical_y == 1) or not np.any(canonical_y == 0):
        raise ValueError(
            "Sensitivity analysis requires both labeled-positive and unlabeled samples."
        )
    predictions = predictions.astype(int, copy=False)

    if scores is None:
        risk_scores = np.where(predictions == 1, 1.0, -1.0)
        score_source = "signed_predictions"
    else:
        risk_scores = np.asarray(scores, dtype=float)
        if risk_scores.ndim != 1 or len(risk_scores) != len(canonical_y):
            raise ValueError("scores must be 1-D and have the same length as y_pu.")
        if not np.isfinite(risk_scores).all():
            raise ValueError("scores must contain only finite values.")
        score_source = "explicit_scores"

    observed_rate = float(np.mean(canonical_y == 1))
    tolerance = np.finfo(float).eps * 8
    points: list[SensitivityPoint] = []
    for class_prior in priors:
        implied_propensity = observed_rate / class_prior
        consistent = bool(implied_propensity <= 1.0 + tolerance)
        reason = None if consistent else "Implied mean propensity exceeds 1."
        precision, risk = _metrics_for_prior(canonical_y, predictions, risk_scores, class_prior)
        points.append(
            SensitivityPoint(
                axis="class_prior",
                value=class_prior,
                class_prior=class_prior,
                label_propensity=implied_propensity,
                is_consistent=consistent,
                consistency_reason=reason,
                pu_estimated_precision=precision,
                pu_zero_one_risk=risk,
            )
        )

    for propensity in propensities:
        implied_prior = observed_rate / propensity
        consistent = bool(implied_prior <= 1.0 + tolerance)
        reason = None if consistent else "Implied class prior exceeds 1."
        precision, risk = _metrics_for_prior(canonical_y, predictions, risk_scores, implied_prior)
        points.append(
            SensitivityPoint(
                axis="label_propensity",
                value=propensity,
                class_prior=implied_prior,
                label_propensity=propensity,
                is_consistent=consistent,
                consistency_reason=reason,
                pu_estimated_precision=precision,
                pu_zero_one_risk=risk,
            )
        )

    frozen_points = tuple(points)
    provenance = {
        "n_samples": len(canonical_y),
        "n_labeled_positive": int(np.sum(canonical_y == 1)),
        "n_unlabeled": int(np.sum(canonical_y == 0)),
        "score_source": score_source,
        "class_prior_grid_size": len(priors),
        "label_propensity_grid_size": len(propensities),
        "model_refit": False,
    }
    return PUSensitivityAnalysis(
        observed_label_rate=observed_rate,
        points=frozen_points,
        metric_ranges=_build_metric_ranges(frozen_points),
        provenance=provenance,
    )


def _format_value(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6f}"


def _format_from_suffix(path: Path) -> Literal["json", "markdown", "csv"]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Cannot infer output format from suffix {suffix!r}; use .json, .md, or .csv.")
