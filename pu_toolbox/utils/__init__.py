"""Shared utility functions for PU Learning Toolbox.

Modules in this package provide common building blocks (basis function
construction, kernel helpers, numerical utilities) that are reused across
estimator implementations.  They are NOT part of the public API and may
change between minor releases.

See ``docs/architecture.md`` for the package layering rationale.
"""

from pu_toolbox.utils.basis import (
    build_linear_basis,
    build_rbf_basis,
    subsample_centers,
)

__all__ = [
    "build_linear_basis",
    "build_rbf_basis",
    "subsample_centers",
]
