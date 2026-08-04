# ruff: noqa: N803

"""Scoring rules, filter logic, and risk warnings for algorithm recommendation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    Maturity,
    SourceStatus,
)
from ..preprocessing import PUDataProfile
from ..registry.metadata import AlgorithmMetadata

__all__ = [
    "DEFAULT_CONFIG",
    "ScoringConfig",
    "global_warnings",
    "method_warnings",
    "score_method",
]


@dataclass(frozen=True)
class ScoringConfig:
    """Scoring weights for the recommendation engine.

    All dimension weights and thresholds are configurable. The total raw
    score is the sum of all ``*_max`` fields; each method's final score
    is ``raw / total_max * 100``.
    """

    assumption_max: float = 30.0
    maturity_max: float = 20.0
    source_max: float = 15.0
    scale_max: float = 20.0
    gpu_max: float = 5.0
    labeled_pos_max: float = 10.0

    maturity_scores: dict[str, float] = field(
        default_factory=lambda: {
            "stable": 20.0,
            "research": 12.0,
            "experimental": 5.0,
            "deprecated": 0.0,
        }
    )
    source_scores: dict[str, float] = field(
        default_factory=lambda: {
            "official_exact": 15.0,
            "official_bundle": 12.0,
            "official_related": 9.0,
            "third_party_only": 6.0,
            "not_found": 3.0,
            "unknown": 1.0,
        }
    )

    small_data_threshold: int = 1000
    large_data_threshold: int = 10000

    @property
    def max_raw_score(self) -> float:
        return (
            self.assumption_max
            + self.maturity_max
            + self.source_max
            + self.scale_max
            + self.gpu_max
            + self.labeled_pos_max
        )


DEFAULT_CONFIG = ScoringConfig()


# ── Scoring ─────────────────────────────────────────────────────


def score_method(
    meta: AlgorithmMetadata,
    profile: PUDataProfile,
    assumption_enum: Assumption | None,
    has_gpu: bool,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> tuple[float, list[str]]:
    """Score a single method against a data profile.

    Returns ``(normalized_score, reasons)`` where score is 0–100.
    """
    raw = 0.0
    reasons: list[str] = []

    assumption_score = _score_assumption(meta, profile, assumption_enum, config)
    raw += assumption_score
    if assumption_score >= config.assumption_max * 0.83:
        reasons.append("Strong assumption match")

    mat_score = config.maturity_scores[meta.maturity.value]
    raw += mat_score
    if meta.maturity == Maturity.STABLE:
        reasons.append("Stable, well-tested implementation")

    src_score = config.source_scores[meta.source_status.value]
    raw += src_score
    if meta.source_status == SourceStatus.OFFICIAL_EXACT:
        reasons.append("Verified against official source code")

    scale_score, scale_reason = _score_data_scale(meta, profile, config)
    raw += scale_score
    if scale_reason:
        reasons.append(scale_reason)

    if has_gpu and meta.supports_gpu:
        raw += config.gpu_max
        reasons.append("GPU-accelerated")
    elif not has_gpu and not meta.supports_gpu:
        raw += config.gpu_max * 0.6
    else:
        raw += config.gpu_max * 0.2

    lp_score = _score_labeled_positives(meta, profile, config)
    raw += lp_score

    final = raw / config.max_raw_score * 100.0
    return round(final, 1), reasons


def _score_assumption(
    meta: AlgorithmMetadata,
    profile: PUDataProfile,
    assumption_enum: Assumption | None,
    config: ScoringConfig,
) -> float:
    cap = config.assumption_max
    if assumption_enum is not None:
        return cap if assumption_enum in meta.assumption else 0.0

    diag_status = profile.selection_diagnostic.get("status", "inconclusive")
    has_scar = Assumption.SCAR in meta.assumption
    has_sar = Assumption.SAR in meta.assumption

    if diag_status == "plausible":
        if has_scar:
            return cap
        if has_sar:
            return cap * 0.5
    elif diag_status == "at_risk":
        if has_sar:
            return cap
        if has_scar:
            return cap / 3.0
    else:
        if has_scar and has_sar:
            return cap * 28.0 / 30.0
        if has_scar or has_sar:
            return cap * 22.0 / 30.0

    return cap / 6.0


def _score_data_scale(
    meta: AlgorithmMetadata,
    profile: PUDataProfile,
    config: ScoringConfig,
) -> tuple[float, str]:
    n_samples = profile.summary.get("n_samples") or 0
    cap = config.scale_max

    if n_samples < config.small_data_threshold:
        if meta.backend in {Backend.NUMPY, Backend.SKLEARN}:
            return cap, "Well-suited for small datasets"
        return cap * 0.4, ""
    elif n_samples > config.large_data_threshold:
        if meta.backend == Backend.TORCH:
            return cap, "Scales well for large datasets"
        return cap * 0.6, ""
    else:
        return cap * 0.8, ""


def _score_labeled_positives(
    meta: AlgorithmMetadata,
    profile: PUDataProfile,
    config: ScoringConfig,
) -> float:
    has_few = any(issue.code == "few_labeled_positives" for issue in profile.issues)
    if has_few and meta.family == AlgorithmFamily.DEEP_PU:
        return config.labeled_pos_max * 0.2
    return config.labeled_pos_max


# ── Warnings ────────────────────────────────────────────────────


def method_warnings(
    meta: AlgorithmMetadata,
    profile: PUDataProfile,
    assumption_enum: Assumption | None,
) -> list[str]:
    """Generate per-method risk warnings."""
    warnings: list[str] = []

    diag_status = profile.selection_diagnostic.get("status", "inconclusive")
    is_scar_only = Assumption.SCAR in meta.assumption and Assumption.SAR not in meta.assumption
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


def global_warnings(
    profile: PUDataProfile,
    class_prior: float | None,
) -> list[str]:
    """Generate global risk warnings based on data profile."""
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
