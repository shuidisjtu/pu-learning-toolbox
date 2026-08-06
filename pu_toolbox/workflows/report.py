# ruff: noqa: N803, N806

"""Report data classes for the end-to-end PU workflow pipeline.

``PipelineReport`` aggregates everything produced by a
:class:`~pu_toolbox.workflows.PUPipeline` run: the data profile, the
recommendation (auto mode only), the resolved class prior and its
source, per-fold CV metrics, the final refitted model, and the final
diagnostic report.  All report types follow the toolbox convention of
``to_dict()`` / ``to_json()`` / ``to_markdown()`` / ``save()``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..diagnostics.report import MetricBasis, PUDiagnosticReport
from ..preprocessing.data_profiler import ProfileIssue, PUDataProfile
from ..registry import RecommendationResult
from ..utils.serialization import escape_markdown, format_from_suffix, format_value, json_safe

PriorSource = Literal["user", "constructor", "estimated", "none"]

__all__ = ["CVMetric", "PipelineReport", "PriorInfo", "PriorSource"]


@dataclass(frozen=True)
class PriorInfo:
    """The resolved class prior and where it came from.

    ``degraded`` is set when automatic estimation failed and the
    pipeline degraded to a no-prior run (auto mode only).
    """

    value: float | None
    source: PriorSource
    method_requires_prior: bool
    estimator: str | None = None
    auto_selected: bool = False
    degraded: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible representation."""
        return {
            "value": self.value,
            "source": self.source,
            "method_requires_prior": self.method_requires_prior,
            "estimator": self.estimator,
            "auto_selected": self.auto_selected,
            "degraded": self.degraded,
        }


@dataclass(frozen=True)
class CVMetric:
    """One metric aggregated over CV folds, with per-fold availability.

    A per-fold ``None`` means that fold was skipped (e.g. a fold with a
    single class for AUC, missing scores, or a missing class prior).
    ``reason`` is only set when every fold was skipped.
    """

    name: str
    per_fold: tuple[float | None, ...]
    basis: MetricBasis
    reason: str | None = None

    @property
    def n_computed(self) -> int:
        """Number of folds with a computed value."""
        return sum(value is not None for value in self.per_fold)

    @property
    def mean(self) -> float | None:
        """Mean over computed folds; ``None`` when none were computed."""
        computed = [value for value in self.per_fold if value is not None]
        if not computed:
            return None
        return float(np.mean(computed))

    @property
    def std(self) -> float | None:
        """Std over computed folds; ``None`` when fewer than two computed."""
        computed = [value for value in self.per_fold if value is not None]
        if len(computed) < 2:
            return None
        return float(np.std(computed))

    @property
    def available(self) -> bool:
        """Whether at least one fold produced a value."""
        return self.mean is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible representation."""
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "basis": self.basis,
            "available": self.available,
            "reason": self.reason,
            "n_computed": self.n_computed,
            "per_fold": list(self.per_fold),
        }


@dataclass(frozen=True)
class PipelineReport:
    """Everything produced by one :meth:`PUPipeline.fit_evaluate` run."""

    profile: PUDataProfile
    recommendation: RecommendationResult | None
    prior: PriorInfo
    cv_metrics: dict[str, CVMetric]
    cv_provenance: dict[str, Any]
    final_model: Any
    diagnostic: PUDiagnosticReport | None
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

    def summary(self) -> str:
        """Render a compact human-readable summary (metrics + issues)."""
        prior_src = self.prior.source
        lines = [
            "PU Pipeline Report",
            "==================",
            "",
            f"Classifier: {self.provenance.get('classifier', 'unknown')}",
            f"Class prior: {format_value(self.prior.value)} (source: {prior_src})",
            f"CV folds: {self.cv_provenance.get('n_splits', 'unknown')}",
            "",
            "## CV Metrics (mean +/- std)",
            "",
            "| Metric | Mean | Std | Basis |",
            "|---|---:|---:|---|",
        ]
        for name, metric in self.cv_metrics.items():
            lines.append(
                f"| `{name}` | {format_value(metric.mean)} | "
                f"{format_value(metric.std)} | `{metric.basis}` |"
            )
        lines.extend(["", "## Issues", ""])
        if self.issues:
            lines.extend(
                f"- **{issue.severity.upper()} `{issue.code}`**: {escape_markdown(issue.message)}"
                for issue in self.issues
            )
            n_errors = sum(issue.severity == "error" for issue in self.issues)
            n_warnings = sum(issue.severity == "warning" for issue in self.issues)
            lines.append("")
            lines.append(f"Summary: {n_errors} error(s), {n_warnings} warning(s).")
        else:
            lines.append("No issues detected.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-serializable nested dictionary.

        The fitted model itself cannot be serialized; it is represented
        by its class name and PU metadata instead.
        """
        model_info: dict[str, Any] = {"class": type(self.final_model).__name__}
        if self.final_model is not None:
            model_info["is_fitted"] = bool(getattr(self.final_model, "_is_fitted", False))
            try:
                model_info["metadata"] = self.final_model.get_pu_metadata()
            except Exception:  # noqa: BLE001 - metadata is best-effort
                model_info["metadata"] = None
        payload = {
            "schema_version": "1.0",
            "classifier": self.provenance.get("classifier"),
            "profile": self.profile.to_dict(),
            "recommendation": (
                self.recommendation.to_dict() if self.recommendation is not None else None
            ),
            "prior": self.prior.to_dict(),
            "cv_metrics": {name: metric.to_dict() for name, metric in self.cv_metrics.items()},
            "cv_provenance": self.cv_provenance,
            "final_model": model_info,
            "diagnostic": self.diagnostic.to_dict() if self.diagnostic is not None else None,
            "issues": [issue.to_dict() for issue in self.issues],
            "provenance": self.provenance,
        }
        return json_safe(payload)

    def to_json(self, *, indent: int = 2) -> str:
        """Render strict JSON without non-standard NaN/Infinity values."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)

    def to_markdown(self) -> str:
        """Render a self-contained Markdown pipeline report."""
        lines = ["# PU Pipeline Report", "", "## Workflow", ""]
        lines.append(f"- Classifier: `{self.provenance.get('classifier')}`")
        lines.append(
            f"- Class prior: {format_value(self.prior.value)} (source: `{self.prior.source}`)"
        )
        lines.append(f"- Prior estimator: `{self.prior.estimator or 'not used'}`")
        lines.append(f"- CV folds: {self.cv_provenance.get('n_splits')}")
        lines.extend(
            [
                "",
                "## CV Metrics",
                "",
                "| Metric | Mean | Std | Basis | Available |",
                "|---|---:|---:|---|---|",
            ]
        )
        for name, metric in self.cv_metrics.items():
            lines.append(
                f"| `{name}` | {format_value(metric.mean)} | {format_value(metric.std)} | "
                f"`{metric.basis}` | {metric.available} |"
            )
        lines.extend(["", "## Issues", ""])
        if self.issues:
            lines.extend(
                f"- **{issue.severity.upper()} `{issue.code}`**: "
                f"{escape_markdown(issue.message)} Action: {escape_markdown(issue.action)}"
                for issue in self.issues
            )
        else:
            lines.append("No issues detected by the configured checks.")
        return "\n".join(lines)

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["json", "markdown"] | None = None,
    ) -> Path:
        """Write the report to ``path``, inferring the format from the suffix.

        An unknown ``format`` raises ``ValueError`` instead of silently
        writing a markdown report (matching ``PUDiagnosticReport.save``
        and ``RecommendationResult.save``).
        """
        target = Path(path)
        fmt = format
        if fmt is None:
            fmt = format_from_suffix(target)
        if fmt not in {"json", "markdown"}:
            raise ValueError(f"Unknown format {fmt!r}; expected 'json' or 'markdown'.")
        text = self.to_json() if fmt == "json" else self.to_markdown()
        target.write_text(text, encoding="utf-8")
        return target
