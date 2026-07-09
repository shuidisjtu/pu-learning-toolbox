"""Algorithm registry — discovery, metadata, and source-aware selection."""

from .builtin_methods import register_all_builtin_methods
from .metadata import AlgorithmMetadata
from .registry import (
    clear_registry,
    get_algorithm,
    get_algorithm_registry,
    get_metadata,
    list_algorithms,
    register_method,
    unregister_method,
)
from .source_metadata import SourceMetadata
from .source_policy import SourcePolicy
