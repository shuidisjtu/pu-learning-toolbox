# ruff: noqa: N803

"""Algorithm recommender — match data profiles to registered PU methods."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    Maturity,
    Scenario,
    SourceStatus,
)
from ..preprocessing import PUDataProfile, profile_pu_data
from .metadata import AlgorithmMetadata
from .registry import _REGISTRY, list_algorithms

__all__ = [
    "MethodCandidate",
    "RecommendationResult",
    "recommend_from_profile",
    "recommend_methods",
]

_MATURITY_SCORE: dict[Maturity, float] = {
    Maturity.STABLE: 20.0,
    Maturity.RESEARCH: 12.0,
    Maturity.EXPERIMENTAL: 5.0,
    Maturity.DEPRECATED: 0.0,
}

_SOURCE_SCORE: dict[SourceStatus, float] = {
    SourceStatus.OFFICIAL_EXACT: 15.0,
    SourceStatus.OFFICIAL_BUNDLE: 12.0,
    SourceStatus.OFFICIAL_RELATED: 9.0,
    SourceStatus.THIRD_PARTY_ONLY: 6.0,
    SourceStatus.NOT_FOUND: 3.0,
    SourceStatus.UNKNOWN: 1.0,
}

_MAX_RAW_SCORE = 30.0 + 20.0 + 15.0 + 20.0 + 5.0 + 10.0  # 100


@dataclass(frozen=True)
class MethodCandidate:
    """A single recommended method with score and rationale."""

    name: str
    score: float
    rank: int
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: AlgorithmMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 1),
            "rank": self.rank,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "family": str(self.metadata.family.value),
            "backend": str(self.metadata.backend.value),
            "maturity": str(self.metadata.maturity.value),
        }


@dataclass(frozen=True)
class RecommendationResult:
    """Ranked algorithm recommendations with filters and warnings."""

    candidates: tuple[MethodCandidate, ...]
    filters_applied: dict[str, Any]
    global_warnings: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "candidates": [c.to_dict() for c in self.candidates],
            "filters_applied": self.filters_applied,
            "global_warnings": list(self.global_warnings),
            "provenance": self.provenance,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    def to_markdown(self) -> str:
        lines = [
            "# PU Method Recommendations",
            "",
        ]
        if self.global_warnings:
            lines.append("## Warnings")
            lines.append("")
            for w in self.global_warnings:
                lines.append(f"- {w}")
            lines.append("")
        lines.extend(
            [
                "## Recommended Methods",
                "",
                "| Rank | Method | Score | Family | Backend | Maturity |",
                "|---:|---|---:|---|---|---|",
            ]
        )
        for c in self.candidates:
            lines.append(
                f"| {c.rank} | {c.name} | {c.score:.1f} "
                f"| {c.metadata.family.value} | {c.metadata.backend.value} "
                f"| {c.metadata.maturity.value} |"
            )
        lines.append("")
        for c in self.candidates:
            lines.append(f"### {c.rank}. {c.name}")
            lines.append("")
            if c.reasons:
                for r in c.reasons:
                    lines.append(f"- {r}")
            if c.warnings:
                lines.append("")
                for w in c.warnings:
                    lines.append(f"- **Warning**: {w}")
            lines.append("")
        return "\n".join(lines)

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["json", "markdown"] | None = None,
    ) -> Path:
        destination = Path(path)
        fmt = format or ("markdown" if destination.suffix in {".md", ".markdown"} else "json")
        content = self.to_markdown() if fmt == "markdown" else self.to_json() + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return destination


def recommend_methods(
    X: Any,
    y_pu: np.ndarray,
    *,
    scenario: Scenario | str | None = None,
    assumption: Assumption | str | None = None,
    class_prior: float | None = None,
    has_gpu: bool = False,
    top_k: int = 5,
    random_state: int | None = 42,
) -> RecommendationResult:
    """Recommend PU methods by profiling data and matching against the registry.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y_pu : array-like of shape (n_samples,)
        PU labels (1 = labeled positive, 0 = unlabeled).
    scenario : Scenario or str, optional
        Data collection scenario constraint.
    assumption : Assumption or str, optional
        Labeling mechanism assumption constraint.
    class_prior : float, optional
        Known class prior P(Y=1). Methods requiring it are excluded when None.
    has_gpu : bool
        Whether GPU is available.
    top_k : int
        Maximum number of candidates to return.
    random_state : int, optional
        Seed for the data profiler.

    Returns
    -------
    RecommendationResult
    """
    profile = profile_pu_data(X, y_pu, class_prior=class_prior, random_state=random_state)
    return recommend_from_profile(
        profile,
        scenario=scenario,
        assumption=assumption,
        class_prior=class_prior,
        has_gpu=has_gpu,
        top_k=top_k,
    )


def recommend_from_profile(
    profile: PUDataProfile,
    *,
    scenario: Scenario | str | None = None,
    assumption: Assumption | str | None = None,
    class_prior: float | None = None,
    has_gpu: bool = False,
    top_k: int = 5,
) -> RecommendationResult:
    """Recommend PU methods from an existing data profile.

    Skips data profiling when a ``PUDataProfile`` is already available.
    See :func:`recommend_methods` for parameter descriptions.
    """
    if not _REGISTRY:
        from .builtin_methods import register_all_builtin_methods

        register_all_builtin_methods()
    scenario_enum = _resolve_enum(scenario, Scenario) if scenario else None
    assumption_enum = _resolve_enum(assumption, Assumption) if assumption else None

    all_methods = list_algorithms(trainable_only=True)

    filters_applied: dict[str, Any] = {}
    filtered = all_methods

    if scenario_enum is not None:
        filtered = [m for m in filtered if scenario_enum in m.scenario]
        filters_applied["scenario"] = scenario_enum.value

    if profile.summary.get("is_sparse", False):
        before = len(filtered)
        filtered = [m for m in filtered if m.supports_sparse]
        if len(filtered) < before:
            filters_applied["sparse_support"] = True

    if class_prior is None:
        before = len(filtered)
        filtered = [m for m in filtered if not m.requires_class_prior]
        if len(filtered) < before:
            filters_applied["class_prior_required"] = "excluded (not provided)"

    if assumption_enum is not None:
        filtered = [m for m in filtered if assumption_enum in m.assumption]
        filters_applied["assumption"] = assumption_enum.value

    scored = []
    for meta in filtered:
        score, reasons = _score_method(meta, profile, assumption_enum, has_gpu)
        warnings = _method_warnings(meta, profile, assumption_enum)
        scored.append((meta, score, reasons, warnings))

    scored.sort(key=lambda t: t[1], reverse=True)
    scored = scored[:top_k]

    candidates = tuple(
        MethodCandidate(
            name=meta.name,
            score=score,
            rank=i + 1,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            metadata=meta,
        )
        for i, (meta, score, reasons, warnings) in enumerate(scored)
    )

    global_warnings = _global_warnings(profile, class_prior)

    provenance = {
        "n_samples": profile.summary.get("n_samples"),
        "n_features": profile.summary.get("n_features"),
        "n_candidates_before_filter": len(all_methods),
        "n_candidates_after_filter": len(filtered),
        "top_k": top_k,
        "has_gpu": has_gpu,
    }

    return RecommendationResult(
        candidates=candidates,
        filters_applied=filters_applied,
        global_warnings=tuple(global_warnings),
        provenance=provenance,
    )


# ── Internal helpers ─────────────────────────────────────────────


def _resolve_enum(value: str | Any, enum_cls: type) -> Any:
    if isinstance(value, enum_cls):
        return value
    for member in enum_cls:
        if member.value.lower() == str(value).lower():
            return member
        if member.name.lower() == str(value).lower():
            return member
    raise ValueError(f"Unknown {enum_cls.__name__} value: {value!r}")


def _score_method(
    meta: AlgorithmMetadata,
    profile: PUDataProfile,
    assumption_enum: Assumption | None,
    has_gpu: bool,
) -> tuple[float, list[str]]:
    raw = 0.0
    reasons: list[str] = []

    # 1. Assumption match (max 30)
    assumption_score = _score_assumption(meta, profile, assumption_enum)
    raw += assumption_score
    if assumption_score >= 25:
        reasons.append("Strong assumption match")

    # 2. Maturity (max 20)
    mat_score = _MATURITY_SCORE.get(meta.maturity, 5.0)
    raw += mat_score
    if meta.maturity == Maturity.STABLE:
        reasons.append("Stable, well-tested implementation")

    # 3. Source status (max 15)
    src_score = _SOURCE_SCORE.get(meta.source_status, 1.0)
    raw += src_score
    if meta.source_status == SourceStatus.OFFICIAL_EXACT:
        reasons.append("Verified against official source code")

    # 4. Data scale fit (max 20)
    scale_score, scale_reason = _score_data_scale(meta, profile)
    raw += scale_score
    if scale_reason:
        reasons.append(scale_reason)

    # 5. GPU preference (max 5)
    if has_gpu and meta.supports_gpu:
        raw += 5.0
        reasons.append("GPU-accelerated")
    elif not has_gpu and not meta.supports_gpu:
        raw += 3.0
    else:
        raw += 1.0

    # 6. Labeled positive sufficiency (max 10)
    lp_score = _score_labeled_positives(meta, profile)
    raw += lp_score

    final = raw / _MAX_RAW_SCORE * 100.0
    return round(final, 1), reasons


def _score_assumption(
    meta: AlgorithmMetadata,
    profile: PUDataProfile,
    assumption_enum: Assumption | None,
) -> float:
    if assumption_enum is not None:
        return 30.0 if assumption_enum in meta.assumption else 0.0

    diag_status = profile.selection_diagnostic.get("status", "inconclusive")
    has_scar = Assumption.SCAR in meta.assumption
    has_sar = Assumption.SAR in meta.assumption

    if diag_status == "plausible":
        if has_scar:
            return 30.0
        if has_sar:
            return 15.0
    elif diag_status == "at_risk":
        if has_sar:
            return 30.0
        if has_scar:
            return 10.0
    else:
        if has_scar and has_sar:
            return 28.0
        if has_scar or has_sar:
            return 22.0

    return 15.0


def _score_data_scale(
    meta: AlgorithmMetadata,
    profile: PUDataProfile,
) -> tuple[float, str]:
    n_samples = profile.summary.get("n_samples", 0)

    if n_samples < 1000:
        if meta.backend in {Backend.NUMPY, Backend.SKLEARN}:
            return 20.0, "Well-suited for small datasets"
        return 8.0, ""
    elif n_samples > 10000:
        if meta.backend == Backend.TORCH:
            return 20.0, "Scales well for large datasets"
        return 12.0, ""
    else:
        return 16.0, ""


def _score_labeled_positives(
    meta: AlgorithmMetadata,
    profile: PUDataProfile,
) -> float:
    has_few = any(issue.code == "few_labeled_positives" for issue in profile.issues)
    if has_few and meta.family == AlgorithmFamily.DEEP_PU:
        return 2.0
    return 10.0


def _method_warnings(
    meta: AlgorithmMetadata,
    profile: PUDataProfile,
    assumption_enum: Assumption | None,
) -> list[str]:
    warnings: list[str] = []

    diag_status = profile.selection_diagnostic.get("status", "inconclusive")
    is_scar_only = Assumption.SAR not in meta.assumption
    if assumption_enum is None and diag_status == "at_risk" and is_scar_only:
        warnings.append(
            "SCAR assumption may not hold for this data; "
            "results from this SCAR-only method may be biased."
        )

    if meta.maturity == Maturity.EXPERIMENTAL:
        warnings.append("Experimental implementation — not yet thoroughly validated.")

    if meta.requires_class_prior:
        warnings.append(
            "Requires class prior (π); incorrect estimation can significantly affect results."
        )

    return warnings


def _global_warnings(
    profile: PUDataProfile,
    class_prior: float | None,
) -> list[str]:
    warnings: list[str] = []

    diag_status = profile.selection_diagnostic.get("status", "inconclusive")
    if diag_status == "at_risk":
        warnings.append(
            "SCAR assumption may not hold for this data. "
            "Consider SAR-aware methods (PUSB, LBE) or "
            "verifying with sensitivity analysis."
        )

    if class_prior is not None:
        warnings.append(
            "Class prior was user-supplied. Incorrect class prior "
            "estimation significantly affects risk estimation methods."
        )

    has_few = any(issue.code == "few_labeled_positives" for issue in profile.issues)
    if has_few:
        warnings.append(
            "Few labeled positives detected. Deep methods may "
            "underperform simpler baselines with limited labels."
        )

    if diag_status == "inconclusive":
        warnings.append(
            "SCAR/SAR diagnostic was inconclusive. Consider providing "
            "y_true for an audited assessment."
        )

    return warnings
