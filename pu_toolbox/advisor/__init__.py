"""Algorithm advisor — data-driven PU method recommendation."""

from ._types import MethodCandidate as MethodCandidate
from ._types import RecommendationResult as RecommendationResult
from .recommender import recommend_from_profile as recommend_from_profile
from .recommender import recommend_methods as recommend_methods
from .rules import DEFAULT_CONFIG as DEFAULT_CONFIG
from .rules import ScoringConfig as ScoringConfig

__all__ = [
    "DEFAULT_CONFIG",
    "MethodCandidate",
    "RecommendationResult",
    "ScoringConfig",
    "recommend_from_profile",
    "recommend_methods",
]
