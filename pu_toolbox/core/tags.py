"""Typed tags for PU scenarios, assumptions, and algorithm metadata.

These lightweight value objects are used by the registry, advisor, and
validation layers to communicate PU-specific attributes without relying
on raw strings.
"""

from __future__ import annotations

from enum import Enum

# ── Scenario ───────────────────────────────────────────────────────


class Scenario(str, Enum):
    """Data collection scenario for PU learning."""

    SINGLE_TRAINING_SET = "single_training_set"
    CASE_CONTROL = "case_control"
    SELECTION_BIASED = "selection_biased"
    UNKNOWN = "unknown"


# ── Assumption ─────────────────────────────────────────────────────


class Assumption(str, Enum):
    """Labeling mechanism assumption.

    ``SAR`` is the canonical name for instance-dependent labeling
    (P(s=1|y=1, x) = c(x)).  Registry consumers that accept raw
    strings should normalise ``"instance_dependent"`` → ``"SAR"``
    before comparison.
    """

    SCAR = "SCAR"  # P(s=1|y=1, x) = constant
    SAR = "SAR"  # P(s=1|y=1, x) = c(x)
    UNKNOWN = "unknown"


# ── Implementation status ──────────────────────────────────────────


class ImplementationStatus(str, Enum):
    """How an algorithm is implemented in the toolbox."""

    API_ONLY = "api_only"  # placeholder, no training logic
    NATIVE = "native"  # clean-room implementation
    OFFICIAL_ADAPTER = "official_adapter"  # wraps author source
    OFFICIAL_ALIGNED_NATIVE = "official_aligned_native"  # native impl + alignment tests
    THIRD_PARTY_REFERENCE = "third_party_reference_only"
    EXPERIMENTAL = "experimental"


# ── Source status ──────────────────────────────────────────────────


class SourceStatus(str, Enum):
    """Availability of author/official source code."""

    OFFICIAL_EXACT = "official_exact"
    OFFICIAL_BUNDLE = "official_bundle"
    OFFICIAL_RELATED = "official_related"
    THIRD_PARTY_ONLY = "third_party_only"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"


# ── Algorithm family ───────────────────────────────────────────────


class AlgorithmFamily(str, Enum):
    """Algorithm family for grouping and recommendation."""

    CLASS_PRIOR_ESTIMATION = "class_prior_estimation"
    CLASSIC_CALIBRATION = "classic_calibration"
    RISK_ESTIMATION = "risk_estimation"
    BIAS_AWARE = "bias_aware"
    DEEP_PU = "deep_pu"
    UNKNOWN = "unknown"


# ── Backend ────────────────────────────────────────────────────────


class Backend(str, Enum):
    """Computational backend used by an algorithm."""

    NUMPY = "numpy"
    SKLEARN = "sklearn"
    TORCH = "torch"
    UNKNOWN = "unknown"


# ── Maturity ───────────────────────────────────────────────────────


class Maturity(str, Enum):
    """Maturity level of an implementation."""

    STABLE = "stable"
    RESEARCH = "research"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"
