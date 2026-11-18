"""Algorithm registry — discovery, metadata, and source-aware selection."""

from .metadata import AlgorithmMetadata as AlgorithmMetadata
from .registry import clear_registry as clear_registry
from .registry import get_algorithm as get_algorithm
from .registry import get_algorithm_registry as get_algorithm_registry
from .registry import get_metadata as get_metadata
from .registry import list_algorithms as list_algorithms
from .registry import register_method as register_method
from .registry import unregister_method as unregister_method

_ADVISOR_SYMBOLS = {
    "MethodCandidate",
    "RecommendationResult",
    "ScoringConfig",
    "recommend_from_profile",
    "recommend_methods",
}


def __getattr__(name: str):
    if name in _ADVISOR_SYMBOLS:
        from .. import advisor  # noqa: F811

        return getattr(advisor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AlgorithmMetadata",
    "MethodCandidate",
    "RecommendationResult",
    "ScoringConfig",
    "clear_registry",
    "get_algorithm",
    "get_algorithm_registry",
    "get_metadata",
    "list_algorithms",
    "recommend_from_profile",
    "recommend_methods",
    "register_method",
    "unregister_method",
]
