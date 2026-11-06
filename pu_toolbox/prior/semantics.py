"""Class-prior mass identities for OS and independent TS sampling."""

from __future__ import annotations

from numbers import Real

import numpy as np

from ..core.validation import validate_input_sampling

__all__ = ["native_unlabeled_to_population", "population_to_native_unlabeled"]


def _probability(value: float, name: str, *, allow_one: bool = True) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    if not allow_one and result == 1.0:
        raise ValueError(f"{name} must be < 1 so the native unlabeled pool is nonempty.")
    return result


def native_unlabeled_to_population(
    class_prior: float, labeled_fraction: float, *, input_sampling: str = "os"
) -> float:
    """Convert a native-U positive prior to the target population prior.

    OS uses ``q + (1-q) * pi_U`` under one-sided positive labeling. TS
    uses ``pi_U`` because its U is sampled from the target population.
    ``labeled_fraction`` is P's fraction of the concatenated input, in
    ``[0, 1)``. Prior endpoints are allowed.
    """
    validate_input_sampling(input_sampling)
    prior = _probability(class_prior, "class_prior")
    q = _probability(labeled_fraction, "labeled_fraction", allow_one=False)
    return q + (1.0 - q) * prior if input_sampling == "os" else prior


def population_to_native_unlabeled(
    class_prior: float, labeled_fraction: float, *, input_sampling: str = "os"
) -> float:
    """Convert a population positive prior to the native-U distribution.

    OS uses ``(pi-q)/(1-q)`` and requires ``pi >= q`` under one-sided
    positive labeling. TS uses ``pi``. ``labeled_fraction`` is in
    ``[0, 1)`` and prior endpoints are allowed.
    """
    validate_input_sampling(input_sampling)
    prior = _probability(class_prior, "class_prior")
    q = _probability(labeled_fraction, "labeled_fraction", allow_one=False)
    if input_sampling == "ts":
        return prior
    if prior < q:
        raise ValueError(f"OS population class_prior ({prior}) must be >= labeled_fraction ({q}).")
    return (prior - q) / (1.0 - q)
