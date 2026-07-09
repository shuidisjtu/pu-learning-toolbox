"""Core utilities for PU Learning Toolbox.

This subpackage provides the foundational types, constants, and helper
functions that every other module depends on — base classes, label
normalisation, data validation, exceptions, random seeding, and PU tags.
"""

from .base import (
    BasePriorEstimator,
    BasePropensityEstimator,
    BasePUClassifier,
    BasePULoss,
)
from .exceptions import (
    ClassPriorError,
    NotFittedError,
    PULearningError,
    RegistryError,
    SourceAdapterError,
    ValidationError,
)
from .labels import normalize_pu_labels
from .validation import validate_pu_X_y
