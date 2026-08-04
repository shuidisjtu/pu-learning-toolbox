"""Algorithm registry — discovery, metadata, and source-aware selection."""

from ..advisor import MethodCandidate as MethodCandidate
from ..advisor import RecommendationResult as RecommendationResult
from ..advisor import ScoringConfig as ScoringConfig
from ..advisor import recommend_from_profile as recommend_from_profile
from ..advisor import recommend_methods as recommend_methods
from .builtin_methods import register_all_builtin_methods as register_all_builtin_methods
from .metadata import AlgorithmMetadata as AlgorithmMetadata
from .registry import bind_estimator_class as bind_estimator_class
from .registry import clear_registry as clear_registry
from .registry import get_algorithm as get_algorithm
from .registry import get_algorithm_registry as get_algorithm_registry
from .registry import get_metadata as get_metadata
from .registry import list_algorithms as list_algorithms
from .registry import register_method as register_method
from .registry import unregister_method as unregister_method

__all__ = [
    "AlgorithmMetadata",
    "MethodCandidate",
    "RecommendationResult",
    "ScoringConfig",
    "bind_estimator_class",
    "clear_registry",
    "get_algorithm",
    "get_algorithm_registry",
    "get_metadata",
    "list_algorithms",
    "recommend_from_profile",
    "recommend_methods",
    "register_all_builtin_methods",
    "register_method",
    "unregister_method",
]
