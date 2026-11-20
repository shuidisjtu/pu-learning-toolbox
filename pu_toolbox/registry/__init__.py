"""Algorithm registry — discovery, metadata, and source-aware selection."""

from ..core.tags import PriorConsumer as PriorConsumer
from .metadata import AlgorithmMetadata as AlgorithmMetadata
from .registry import clear_registry as clear_registry
from .registry import get_algorithm as get_algorithm
from .registry import get_algorithm_registry as get_algorithm_registry
from .registry import get_metadata as get_metadata
from .registry import list_algorithms as list_algorithms
from .registry import register_method as register_method
from .registry import unregister_method as unregister_method

__all__ = [
    "AlgorithmMetadata",
    "PriorConsumer",
    "clear_registry",
    "get_algorithm",
    "get_algorithm_registry",
    "get_metadata",
    "list_algorithms",
    "register_method",
    "unregister_method",
]
