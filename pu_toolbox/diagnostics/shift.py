# ruff: noqa: N803, N806

"""Distribution-shift auditing for positive-unlabeled datasets.

This module deliberately estimates the observable marginal density ratio
``p_target(x) / p_source(x)``.  It is a covariate-shift baseline, not an
implementation of class-conditional or joint ``p_target(x, y)`` weighting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pu_toolbox.core.labels import normalize_pu_labels
from pu_toolbox.preprocessing.data_profiler import ProfileIssue
from pu_toolbox.utils.serialization import escape_markdown, format_from_suffix, json_safe

ShiftSeverity = Literal["low", "moderate", "high"]

__all__ = [
    "PUShiftReport",
    "ShiftSeverity",
    "analyze_pu_shift",
]


@dataclass(frozen=True)
class PUShiftReport:
    """Structured source-to-target distribution-shift audit."""

    domain_auc: float
    severity: ShiftSeverity
    sample_summary: dict[str, Any]
    weight_summary: dict[str, float]
    issues: tuple[ProfileIssue, ...]
    adaptation_ready: bool
    source_importance_weights: np.ndarray
    provenance: dict[str, Any]

    @property
    def has_warnings(self) -> bool:
        """Whether any audit issue merits review."""
        return any(issue.severity == "warning" for issue in self.issues)

    def to_weights_frame(self) -> pd.DataFrame:
        """Return source-row indices and normalized importance weights."""
        return pd.DataFrame(
            {
                "source_row": np.arange(len(self.source_importance_weights), dtype=int),
                "importance_weight": self.source_importance_weights.copy(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible summary (weights are a separate artifact)."""
        return json_safe(
            {
                "schema_version": "1.0",
                "analysis_type": "marginal_distribution_shift_audit",
                "domain_auc": self.domain_auc,
                "severity": self.severity,
                "sample_summary": self.sample_summary,
                "weight_summary": self.weight_summary,
                "adaptation_ready": self.adaptation_ready,
                "issues": [issue.to_dict() for issue in self.issues],
                "artifacts": {
                    "source_importance_weights": "source_importance_weights.csv",
                },
                "provenance": self.provenance,
            }
        )

    def to_json(self, *, indent: int = 2) -> str:
        """Render strict JSON without NaN or Infinity."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)

    def to_markdown(self) -> str:
        """Render a self-contained Markdown shift report."""
        summary = self.sample_summary
        weights = self.weight_summary
        lines = [
            "# PU Distribution Shift Audit",
            "",
            f"- Domain OOF AUC: `{self.domain_auc:.6f}`",
            f"- Shift severity: `{self.severity}`",
            f"- Source samples: `{summary['n_source']}`",
            f"- Target samples: `{summary['n_target']}`",
            f"- Data ready for covariate adaptation: `{self.adaptation_ready}`",
            "",
            "## Importance-weight Stability",
            "",
            f"- Effective sample size: `{weights['effective_sample_size']:.6f}` "
            f"(`{weights['effective_sample_fraction']:.6f}` of source)",
            f"- Weight range: `[{weights['minimum']:.6f}, {weights['maximum']:.6f}]`",
            f"- Probability clipping fraction: `{weights['probability_clip_fraction']:.6f}`",
            f"- Relative-boundary fraction: `{weights['relative_boundary_fraction']:.6f}`",
            "",
            "## Issues",
            "",
        ]
        if self.issues:
            lines.extend(
                f"- **{issue.severity.upper()} `{issue.code}`**: "
                f"{escape_markdown(issue.message)} Action: {escape_markdown(issue.action)}"
                for issue in self.issues
            )
        else:
            lines.append("No configured issue threshold was triggered.")
        lines.extend(
            [
                "",
                "## Interpretation Boundary",
                "",
                "- Domain AUC detects observable domain separability; it does not identify "
                "the shift type or prove that distributions are equal.",
                "- Exported weights estimate the marginal ratio `p_target(x) / p_source(x)` "
                "with a bounded relative-ratio transform.",
                "- Weighting is justified only under a covariate-shift assumption; it does "
                "not automatically correct concept or joint distribution shift.",
                "- Source-domain cross-validation is not evidence of target-domain performance.",
            ]
        )
        return "\n".join(lines) + "\n"

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["json", "markdown", "csv"] | None = None,
    ) -> Path:
        """Save the report as JSON/Markdown, or source weights as CSV."""
        destination = Path(path)
        output_format = format or format_from_suffix(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "json":
            destination.write_text(self.to_json() + "\n", encoding="utf-8")
        elif output_format == "markdown":
            destination.write_text(self.to_markdown(), encoding="utf-8")
        elif output_format == "csv":
            self.to_weights_frame().to_csv(destination, index=False)
        else:  # pragma: no cover - Literal plus format_from_suffix make this defensive
            raise ValueError("format must be 'json', 'markdown', or 'csv'.")
        return destination


def _as_feature_matrix(X: Any, *, name: str) -> np.ndarray | sparse.csr_matrix:
    if sparse.issparse(X):
        matrix = sparse.csr_matrix(X, dtype=float)
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one sample and one feature.")
        if not np.isfinite(matrix.data).all():
            raise ValueError(f"{name} must contain only finite numeric values.")
        return matrix
    try:
        array = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric feature values.") from exc
    if array.ndim == 4:
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D or 4-D NCHW; got ndim={array.ndim}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one feature.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values.")
    return array


def _validated_pu_labels(y_pu: Any, n_samples: int, *, name: str) -> np.ndarray:
    labels = normalize_pu_labels(y_pu)
    if len(labels) != n_samples:
        raise ValueError(f"{name} has {len(labels)} rows but its feature matrix has {n_samples}.")
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError(f"{name} must contain both labeled-positive and unlabeled samples.")
    return labels


def _domain_model(*, sparse_input: bool, random_state: int | None) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=not sparse_input)),
            (
                "domain_classifier",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )


def _severity(domain_auc: float, moderate_auc: float, high_auc: float) -> ShiftSeverity:
    if domain_auc >= high_auc:
        return "high"
    if domain_auc >= moderate_auc:
        return "moderate"
    return "low"


def _weight_statistics(
    weights: np.ndarray,
    raw_relative_weights: np.ndarray,
    domain_probabilities: np.ndarray,
    *,
    alpha: float,
    probability_clip: float,
) -> dict[str, float]:
    total = float(weights.sum())
    ess = total * total / float(np.square(weights).sum())
    quantiles = np.quantile(weights, [0.05, 0.25, 0.5, 0.75, 0.95])
    boundary = 1.0 / alpha
    return {
        "minimum": float(weights.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "maximum": float(weights.max()),
        "mean": float(weights.mean()),
        "effective_sample_size": ess,
        "effective_sample_fraction": ess / len(weights),
        "probability_clip_fraction": float(
            np.mean(
                (domain_probabilities <= probability_clip)
                | (domain_probabilities >= 1.0 - probability_clip)
            )
        ),
        "relative_boundary_fraction": float(np.mean(raw_relative_weights >= 0.99 * boundary)),
        "raw_relative_upper_bound": boundary,
    }


def _audit_issues(
    *,
    severity: ShiftSeverity,
    weight_summary: dict[str, float],
    target_labels_available: bool,
    source_label_rate: float,
    target_label_rate: float | None,
    min_effective_sample_fraction: float,
    max_boundary_fraction: float,
) -> tuple[ProfileIssue, ...]:
    issues: list[ProfileIssue] = []
    if severity == "high":
        issues.append(
            ProfileIssue(
                code="domain_shift_high",
                severity="warning",
                message="Source and target samples are strongly distinguishable.",
                action="Inspect changed features and validate performance on target-domain data.",
            )
        )
    elif severity == "moderate":
        issues.append(
            ProfileIssue(
                code="domain_shift_moderate",
                severity="info",
                message="The audit found observable source-to-target differences.",
                action="Review feature, time, and population splits before deployment.",
            )
        )
    if weight_summary["effective_sample_fraction"] < min_effective_sample_fraction:
        issues.append(
            ProfileIssue(
                code="low_effective_sample_size",
                severity="warning",
                message="Importance weighting leaves too little effective source data.",
                action="Collect target-domain PU data or improve source-domain coverage.",
            )
        )
    if weight_summary["relative_boundary_fraction"] > max_boundary_fraction:
        issues.append(
            ProfileIssue(
                code="relative_weight_boundary",
                severity="warning",
                message="Many source weights reach the relative density-ratio boundary.",
                action="Treat weighted estimates as high variance and inspect support overlap.",
            )
        )
    if not target_labels_available:
        issues.append(
            ProfileIssue(
                code="target_pu_missing",
                severity="info",
                message=(
                    "Target-domain PU labels were not supplied; only drift auditing is available."
                ),
                action="Provide a small target PU sample before attempting adaptation.",
            )
        )
    elif target_label_rate is not None and abs(target_label_rate - source_label_rate) >= 0.1:
        issues.append(
            ProfileIssue(
                code="observed_label_rate_shift",
                severity="warning",
                message="Observed labeled-positive rates differ materially between domains.",
                action=(
                    "Audit class priors and labeling propensities separately; label rate is not π."
                ),
            )
        )
    return tuple(issues)


def analyze_pu_shift(
    X_source: Any,
    y_source_pu: Any,
    X_target: Any,
    *,
    y_target_pu: Any | None = None,
    alpha: float = 0.1,
    probability_clip: float = 1e-6,
    cv: int = 5,
    random_state: int | None = 42,
    moderate_auc: float = 0.60,
    high_auc: float = 0.75,
    min_effective_sample_fraction: float = 0.50,
    max_boundary_fraction: float = 0.05,
) -> PUShiftReport:
    """Audit source-to-target drift and estimate bounded marginal weights.

    The returned weights estimate a relative transform of
    ``p_target(x) / p_source(x)`` and are normalized to mean one on source
    rows.  They are justified as adaptation weights only under covariate shift.
    """
    source = _as_feature_matrix(X_source, name="X_source")
    target = _as_feature_matrix(X_target, name="X_target")
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            "X_source and X_target must have the same number of flattened features; "
            f"got {source.shape[1]} and {target.shape[1]}."
        )
    source_labels = _validated_pu_labels(y_source_pu, source.shape[0], name="y_source_pu")
    target_labels = (
        _validated_pu_labels(y_target_pu, target.shape[0], name="y_target_pu")
        if y_target_pu is not None
        else None
    )
    if isinstance(cv, bool) or not isinstance(cv, int) or cv < 2:
        raise ValueError("cv must be an integer >= 2.")
    numeric = {
        "alpha": alpha,
        "probability_clip": probability_clip,
        "moderate_auc": moderate_auc,
        "high_auc": high_auc,
        "min_effective_sample_fraction": min_effective_sample_fraction,
        "max_boundary_fraction": max_boundary_fraction,
    }
    if any(isinstance(value, bool) or not np.isscalar(value) for value in numeric.values()):
        raise TypeError("shift-audit numeric parameters must be real scalars.")
    alpha = float(alpha)
    probability_clip = float(probability_clip)
    moderate_auc = float(moderate_auc)
    high_auc = float(high_auc)
    min_effective_sample_fraction = float(min_effective_sample_fraction)
    max_boundary_fraction = float(max_boundary_fraction)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1].")
    if not 0.0 < probability_clip < 0.5:
        raise ValueError("probability_clip must lie in (0, 0.5).")
    if not 0.5 <= moderate_auc < high_auc <= 1.0:
        raise ValueError("AUC thresholds must satisfy 0.5 <= moderate_auc < high_auc <= 1.")
    if not 0.0 < min_effective_sample_fraction <= 1.0:
        raise ValueError("min_effective_sample_fraction must lie in (0, 1].")
    if not 0.0 <= max_boundary_fraction <= 1.0:
        raise ValueError("max_boundary_fraction must lie in [0, 1].")

    n_source, n_target = source.shape[0], target.shape[0]
    n_splits = min(cv, n_source, n_target)
    if n_splits < 2:
        raise ValueError("Each domain must contain at least two samples for OOF auditing.")
    sparse_input = sparse.issparse(source) or sparse.issparse(target)
    if sparse_input:
        combined = sparse.vstack(
            [sparse.csr_matrix(source), sparse.csr_matrix(target)], format="csr"
        )
    else:
        combined = np.vstack([source, target])
    domains = np.concatenate([np.zeros(n_source, dtype=int), np.ones(n_target, dtype=int)])
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = _domain_model(sparse_input=sparse_input, random_state=random_state)
    oof_probabilities = cross_val_predict(
        model,
        combined,
        domains,
        cv=splitter,
        method="predict_proba",
    )[:, 1]
    raw_auc = float(roc_auc_score(domains, oof_probabilities))
    domain_auc = max(raw_auc, 1.0 - raw_auc)

    model.fit(combined, domains)
    source_domain_probabilities = model.predict_proba(source)[:, 1]
    clipped_probabilities = np.clip(
        source_domain_probabilities, probability_clip, 1.0 - probability_clip
    )
    marginal_ratio = clipped_probabilities / (1.0 - clipped_probabilities) * (n_source / n_target)
    raw_relative_weights = marginal_ratio / (alpha * marginal_ratio + (1.0 - alpha))
    mean_weight = float(raw_relative_weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0.0:
        raise ValueError("Estimated source importance weights are not finite and positive.")
    weights = raw_relative_weights / mean_weight
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("Normalized source importance weights are not finite and positive.")

    weight_summary = _weight_statistics(
        weights,
        raw_relative_weights,
        source_domain_probabilities,
        alpha=alpha,
        probability_clip=probability_clip,
    )
    shift_severity = _severity(domain_auc, moderate_auc, high_auc)
    source_label_rate = float(np.mean(source_labels == 1))
    target_label_rate = float(np.mean(target_labels == 1)) if target_labels is not None else None
    issues = _audit_issues(
        severity=shift_severity,
        weight_summary=weight_summary,
        target_labels_available=target_labels is not None,
        source_label_rate=source_label_rate,
        target_label_rate=target_label_rate,
        min_effective_sample_fraction=min_effective_sample_fraction,
        max_boundary_fraction=max_boundary_fraction,
    )
    coverage_ok = (
        weight_summary["effective_sample_fraction"] >= min_effective_sample_fraction
        and weight_summary["relative_boundary_fraction"] <= max_boundary_fraction
    )
    return PUShiftReport(
        domain_auc=domain_auc,
        severity=shift_severity,
        sample_summary={
            "n_source": n_source,
            "n_target": n_target,
            "n_features": source.shape[1],
            "source_labeled_positive_rate": source_label_rate,
            "target_labeled_positive_rate": target_label_rate,
            "target_pu_labels_available": target_labels is not None,
        },
        weight_summary=weight_summary,
        issues=issues,
        adaptation_ready=target_labels is not None and coverage_ok,
        source_importance_weights=weights,
        provenance={
            "density_ratio_scope": "marginal_covariate",
            "domain_classifier": "StandardScaler+LogisticRegression",
            "domain_auc_evaluation": "stratified_out_of_fold_orientation_invariant",
            "n_splits": n_splits,
            "requested_cv": cv,
            "alpha": alpha,
            "probability_clip": probability_clip,
            "random_state": random_state,
            "moderate_auc": moderate_auc,
            "high_auc": high_auc,
            "min_effective_sample_fraction": min_effective_sample_fraction,
            "max_boundary_fraction": max_boundary_fraction,
            "guarantee": "covariate_shift_only",
        },
    )
