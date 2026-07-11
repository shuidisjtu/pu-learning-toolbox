"""Algorithm registry — discovery, metadata, and source-aware selection."""

from .builtin_methods import register_all_builtin_methods as register_all_builtin_methods
from .metadata import AlgorithmMetadata as AlgorithmMetadata
from .registry import clear_registry as clear_registry
from .registry import get_algorithm as get_algorithm
from .registry import get_algorithm_registry as get_algorithm_registry
from .registry import get_metadata as get_metadata
from .registry import list_algorithms as list_algorithms
from .registry import register_method as register_method
from .registry import unregister_method as unregister_method
from .source_metadata import SourceMetadata as SourceMetadata
from .source_policy import SourcePolicy as SourcePolicy

__all__ = [
    "AlgorithmMetadata",
    "SourceMetadata",
    "SourcePolicy",
    "clear_registry",
    "get_algorithm",
    "get_algorithm_registry",
    "get_metadata",
    "list_algorithms",
    "register_all_builtin_methods",
    "register_method",
    "unregister_method",
]
