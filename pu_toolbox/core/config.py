"""Global configuration and constants for PU Learning Toolbox."""

from __future__ import annotations

# fmt: off
# ── PU label encoding ──────────────────────────────────────────────
# Canonical form: positive = +1, unlabeled = 0
# These are public constants — third-party estimators may reference them.
POSITIVE_LABEL: int = 1
UNLABELED_LABEL: int = 0
NEGATIVE_LABEL: int = -1

# ── Numeric defaults ───────────────────────────────────────────────
DEFAULT_RANDOM_SEED: int = 42
CLASS_PRIOR_CLIP_EPS: float = 1e-12    # floor / ceil for π ∈ (0, 1)
PROPENSITY_CLIP_EPS: float = 1e-12     # floor / ceil for c ∈ (0, 1]

# ── Validation thresholds ──────────────────────────────────────────
MIN_POSITIVE_SAMPLES: int = 2          # minimum labeled positives required
MAX_PU_RATIO: float = 1e3              # warn if |U| / |P| exceeds this
