"""Typed tags for PU scenarios, assumptions, and algorithm metadata.

These lightweight value objects are used by the registry and
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


# ── Training cost ──────────────────────────────────────────────────


class TrainingCost(str, Enum):
    """Relative training cost of an algorithm (used by the recommender).

    A registry-only field: estimators do not declare it as a class
    attribute.  HIGH is reserved for heavy fixed-epoch iterative
    solvers that dominate runtime on small data (e.g. LLSVM's 3000
    non-convex SGD epochs).
    """

    LOW = "low"  # closed-form or convex solver, effectively instant
    MEDIUM = "medium"  # bounded iterative training (<= ~1000 epochs)
    HIGH = "high"  # heavy fixed-epoch non-convex training
    UNKNOWN = "unknown"  # not classified; scored neutrally
