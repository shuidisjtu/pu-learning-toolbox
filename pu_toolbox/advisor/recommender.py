# ruff: noqa: N803

"""Algorithm recommender — match data profiles to registered PU methods."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from ..core.tags import Assumption, Maturity, Scenario
from ..preprocessing import PUDataProfile, profile_pu_data
from ..registry.registry import list_algorithms
from ._types import MethodCandidate, RecommendationResult
from .rules import DEFAULT_CONFIG, ScoringConfig, global_warnings, method_warnings, score_method

__all__ = [
    "recommend_from_profile",
    "recommend_methods",
]

_BUILTINS_REGISTERED = False


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
    config: ScoringConfig | None = None,
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
    config : ScoringConfig, optional
        Custom scoring weights. Uses ``DEFAULT_CONFIG`` when not provided.

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
        config=config,
    )


def recommend_from_profile(
    profile: PUDataProfile,
    *,
    scenario: Scenario | str | None = None,
    assumption: Assumption | str | None = None,
    class_prior: float | None = None,
    has_gpu: bool = False,
    top_k: int = 5,
    config: ScoringConfig | None = None,
) -> RecommendationResult:
    """Recommend PU methods from an existing data profile.

    Skips data profiling when a ``PUDataProfile`` is already available.
    See :func:`recommend_methods` for parameter descriptions.

    .. note::

       ``class_prior`` controls hard filtering (excluding methods that
       require π when it is unavailable).  It must be consistent with
       the value used to build *profile*; passing a different value
       produces contradictory results.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    cfg = config or DEFAULT_CONFIG

    global _BUILTINS_REGISTERED  # noqa: PLW0603
    if not _BUILTINS_REGISTERED:
        from ..registry.builtin_methods import register_all_builtin_methods

        register_all_builtin_methods()
        _BUILTINS_REGISTERED = True

    scenario_enum = _resolve_enum(scenario, Scenario) if scenario is not None else None
    assumption_enum = _resolve_enum(assumption, Assumption) if assumption is not None else None

    all_methods = list_algorithms(trainable_only=True)

    filters_applied: dict[str, Any] = {}
    filtered = [m for m in all_methods if m.maturity != Maturity.DEPRECATED]

    if scenario_enum is not None:
        filtered = [m for m in filtered if scenario_enum in m.scenario]
        filters_applied["scenario"] = scenario_enum.value

    if profile.summary.get("is_sparse", False):
        filtered = [m for m in filtered if m.supports_sparse]
        filters_applied["sparse_support"] = True

    if class_prior is None:
        filtered = [m for m in filtered if not m.requires_class_prior]
        filters_applied["class_prior_required"] = "excluded (not provided)"

    if assumption_enum is not None:
        filtered = [m for m in filtered if assumption_enum in m.assumption]
        filters_applied["assumption"] = assumption_enum.value

    scored = []
    for meta in filtered:
        sc, reasons = score_method(meta, profile, assumption_enum, has_gpu, cfg)
        warns = method_warnings(meta, profile, assumption_enum)
        scored.append((meta, sc, reasons, warns))

    scored.sort(key=lambda t: t[1], reverse=True)
    scored = scored[:top_k]

    candidates = tuple(
        MethodCandidate(
            name=meta.name,
            score=sc,
            rank=i + 1,
            reasons=tuple(reasons),
            warnings=tuple(warns),
            metadata=meta,
        )
        for i, (meta, sc, reasons, warns) in enumerate(scored)
    )

    gw = global_warnings(profile, class_prior)

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
        global_warnings=tuple(gw),
        provenance=provenance,
    )


def _resolve_enum(value: str | Any, enum_cls: type) -> Any:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, Enum):
        raise ValueError(f"Expected {enum_cls.__name__}, got {type(value).__name__}: {value!r}")
    for member in enum_cls:
        if member.value.lower() == str(value).lower():
            return member
        if member.name.lower() == str(value).lower():
            return member
    raise ValueError(f"Unknown {enum_cls.__name__} value: {value!r}")
