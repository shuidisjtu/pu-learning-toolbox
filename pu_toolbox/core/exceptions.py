"""Custom exceptions for PU Learning Toolbox.

All exceptions inherit from :class:`PULearningError` so callers can catch
PU-specific errors with a single except clause.
"""

from sklearn.exceptions import NotFittedError as _SklearnNotFittedError


class PULearningError(Exception):
    """Base exception for all PU Learning Toolbox errors."""


class ValidationError(PULearningError):
    """Raised when input data or labels fail PU-specific validation rules.

    Typical causes: wrong label values, mismatched shapes, P/U ratio
    outside sensible ranges, or incompatible assumption/scenario combos.
    """


class NotFittedError(PULearningError, _SklearnNotFittedError):
    """Raised when ``predict`` / ``decision_function`` is called before ``fit``.

    Inherits from both :class:`PULearningError` (for toolbox-wide catching)
    and :class:`sklearn.exceptions.NotFittedError` (for compatibility with
    sklearn Pipeline, GridSearchCV, and ``check_is_fitted``).
    """


class ClassPriorError(PULearningError):
    """Raised when a class prior estimate is invalid or unavailable.

    This includes out-of-range values, negative estimates, and attempts to
    train a prior-dependent model without providing or estimating the prior.
    """


class SourceAdapterError(PULearningError):
    """Raised when an external source adapter cannot be loaded or executed.

    Typical causes: missing dependency, license restriction, broken external
    repo, or incompatible framework version.
    """


class RegistryError(PULearningError):
    """Raised when registry operations fail (duplicate name, missing alias, etc.)."""
