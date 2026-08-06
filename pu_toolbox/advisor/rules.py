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
    TrainingCost,
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

    Each ``*_max`` field is the normalization anchor of its dimension
    (the denominator in ``raw / total_max * 100``); the ``*_scores``
    tables map discrete levels to their contribution and must not exceed
    their dimension's anchor (enforced in ``__post_init__``).
    """

    assumption_max: float = 30.0
    maturity_max: float = 20.0
    source_max: float = 15.0
    scale_max: float = 20.0
    gpu_max: float = 5.0
    labeled_pos_max: float = 10.0
    cost_max: float = 10.0

    # NOTE: no "deprecated" level here -- the recommender filters
    # DEPRECATED methods before scoring (see recommend_from_profile), so
    # a dead table entry would never be consumed.
    maturity_scores: dict[str, float] = field(
        default_factory=lambda: {
            "stable": 20.0,
            "research": 12.0,
            "experimental": 5.0,
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

    def __post_init__(self) -> None:
        """Reject score tables whose entries exceed their dimension anchor.

        Without this, a table value above its ``*_max`` silently inflates
        the raw score past ``max_raw_score`` and breaks the documented
        0-100 output contract of :func:`score_method`.
        """
        for table_name, max_field in (
            ("maturity_scores", "maturity_max"),
            ("source_scores", "source_max"),
        ):
            cap = getattr(self, max_field)
            if any(value > cap for value in getattr(self, table_name).values()):
                raise ValueError(f"{table_name} values must not exceed {max_field}={cap}")

    @property
    def max_raw_score(self) -> float:
        return (
            self.assumption_max
            + self.maturity_max
            + self.source_max
            + self.scale_max
            + self.gpu_max
            + self.labeled_pos_max
            + self.cost_max
        )


DEFAULT_CONFIG = ScoringConfig()

# Fractional constants anchoring the assumption scoring to its cap
# (kept as named constants so the threshold and the branch coefficients
# stay in sync when the assumption bands change).
_STRONG_ASSUMPTION_FRACTION = 0.83  # "Strong assumption match" reason threshold
_ASSUMPTION_AT_RISK_SCAR = 1.0 / 3.0  # SCAR-only method under an at-risk diagnostic
_ASSUMPTION_INCONCLUSIVE_BOTH = 28.0 / 30.0  # both SCAR and SAR claimed
_ASSUMPTION_INCONCLUSIVE_ONE = 22.0 / 30.0  # exactly one claim
_ASSUMPTION_NO_MATCH = 1.0 / 6.0  # no claim matches the diagnostic


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
    if assumption_score >= config.assumption_max * _STRONG_ASSUMPTION_FRACTION:
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

    cost_score, cost_reason = _score_training_cost(meta, profile, config)
    raw += cost_score
    if cost_reason:
        reasons.append(cost_reason)

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
            return cap * _ASSUMPTION_AT_RISK_SCAR
    else:
        if has_scar and has_sar:
            return cap * _ASSUMPTION_INCONCLUSIVE_BOTH
        if has_scar or has_sar:
            return cap * _ASSUMPTION_INCONCLUSIVE_ONE

    return cap * _ASSUMPTION_NO_MATCH


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


def _score_training_cost(
    meta: AlgorithmMetadata,
    profile: PUDataProfile,
    config: ScoringConfig,
) -> tuple[float, str]:
    """Score the relative training cost of a method.

    Heavy fixed-epoch iterative solvers (e.g. LLSVM's 3000-epoch SGD)
    dominate runtime on small data, so HIGH cost scores a small fraction
    of the cap there; the penalty shrinks as data grows (runtime
    perception fades) by reusing the same size bands as
    :func:`_score_data_scale`.
    """
    n_samples = profile.summary.get("n_samples") or 0
    cap = config.cost_max
    if n_samples > config.large_data_threshold:
        fraction = {
            TrainingCost.LOW: 1.0,
            TrainingCost.MEDIUM: 0.85,
            TrainingCost.HIGH: 0.7,
            TrainingCost.UNKNOWN: 0.6,
        }
    elif n_samples >= config.small_data_threshold:
        fraction = {
            TrainingCost.LOW: 1.0,
            TrainingCost.MEDIUM: 0.8,
            TrainingCost.HIGH: 0.5,
            TrainingCost.UNKNOWN: 0.6,
        }
    else:
        fraction = {
            TrainingCost.LOW: 1.0,
            TrainingCost.MEDIUM: 0.7,
            TrainingCost.HIGH: 0.2,
            TrainingCost.UNKNOWN: 0.6,
        }
    score = cap * fraction[meta.training_cost]
    reason = ""
    if n_samples < config.small_data_threshold:
        if meta.training_cost == TrainingCost.LOW:
            reason = "Fast on small datasets"
        elif meta.training_cost == TrainingCost.HIGH:
            reason = "High training cost on small datasets"
    return score, reason


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
    class_prior_source: str | None = None,
) -> list[str]:
    """Generate global risk warnings based on data profile.

    ``class_prior_source`` distinguishes an explicitly supplied prior
    ("user" / "constructor" / None → warning) from an auto-estimated one
    ("estimated" → no warning; profile checks like ``inconsistent_class_prior``
    already cover a suspect estimate).
    """
    warnings: list[str] = []

    diag_status = profile.selection_diagnostic.get("status", "inconclusive")
    if diag_status == "at_risk":
        warnings.append(
            "SCAR assumption may not hold for this data. "
            "Consider SAR-aware methods (PUSB, LBE) or "
            "verifying with sensitivity analysis."
        )

    if class_prior is not None and class_prior_source != "estimated":
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
