"""Base classes for all PU Learning estimators and losses.

Every PU classifier, prior estimator, propensity estimator, and loss
function in the toolbox inherits from one of the base classes defined
here.

API Contracts
-------------
See ``docs/architecture.md`` §5 for the full specification.
"""

# ruff: noqa: N803

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin

from .exceptions import NotFittedError
from .tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    ImplementationStatus,
    Maturity,
    Scenario,
    SourceStatus,
)

# ═════════════════════════════════════════════════════════════════════
# PU Classifier
# ═════════════════════════════════════════════════════════════════════


class BasePUClassifier(BaseEstimator, ClassifierMixin, ABC):
    """Abstract base class for all PU classifiers.

    Every PU classifier **must** implement:

    * ``fit(X, y_pu, *, class_prior=None, sample_weight=None)``
    * ``_predict(X)`` — called by the public ``predict(X)`` wrapper
    * ``_decision_function(X)`` — called by ``decision_function(X)`` and
      the default ``score_samples(X)``

    The following are optional but recommended:

    * ``score_samples(X)`` — override only when the score convention differs
      from ``decision_function(X)``
    * ``predict_proba(X)`` — calibrated P(y=1 | x)
    * ``predict_label_proba(X)`` — P(s=1 | x) for propensity-aware models
    * ``get_pu_metadata()`` — dict of assumption / scenario / diagnostics

    All estimators must support ``get_params()`` and ``set_params()``
    (provided by :class:`sklearn.base.BaseEstimator`) for compatibility
    with ``Pipeline`` and ``GridSearchCV``.
    """

    # ── Metadata (override in subclasses) ──────────────────────────
    family: AlgorithmFamily = AlgorithmFamily.UNKNOWN
    assumption: tuple[Assumption, ...] = (Assumption.UNKNOWN,)
    scenario: tuple[Scenario, ...] = (Scenario.UNKNOWN,)
    requires_class_prior: bool = False
    implementation_status: ImplementationStatus = ImplementationStatus.API_ONLY
    source_status: SourceStatus = SourceStatus.UNKNOWN
    backend: Backend = Backend.NUMPY
    maturity: Maturity = Maturity.EXPERIMENTAL

    # ── Internal state ─────────────────────────────────────────────

    def __init__(self) -> None:
        """Initialise instance attributes.

        Class-level annotations are only type hints; actual instance
        state must be set here so that ``sklearn.base.clone()`` and
        pickle work correctly.
        """
        self._is_fitted = False
        self._class_prior: float | None = None
        self._X_shape_: tuple[int, int] | None = None

    # ── Core API (public: fit-check → delegate to private) ─────────

    @abstractmethod
    def fit(
        self,
        X: np.ndarray | sparse.spmatrix,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> BasePUClassifier:
        """Fit the PU classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y_pu : array-like of shape (n_samples,)
            PU labels.  Accepted conventions: ``{+1, 0}``, ``{+1, -1}``,
            ``{1, 0}``, ``{1, -1}``.
        class_prior : float, optional
            Known class prior π = P(y=1).  Required for risk-estimation
            methods; optional for SCAR-based calibration methods.
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights.

        Returns
        -------
        self : BasePUClassifier
            Fitted estimator.
        """
        ...

    def predict(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Predict binary class labels {0, 1}.

        Enforces ``fit``-before-``predict`` via :meth:`_check_is_fitted`,
        then delegates to :meth:`_predict`.
        """
        self._check_is_fitted()
        return self._predict(X)

    @abstractmethod
    def _predict(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Core prediction logic (subclasses override this).

        Returns
        -------
        np.ndarray of shape (n_samples,) and dtype int
        """
        ...

    def decision_function(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Raw decision scores (higher → more likely positive).

        Enforces ``fit``-before-``predict`` via :meth:`_check_is_fitted`,
        then delegates to :meth:`_decision_function`.

        Subclasses implement :meth:`_decision_function`. Override
        :meth:`score_samples` only when a different score convention is needed.
        """
        self._check_is_fitted()
        return self._decision_function(X)

    @abstractmethod
    def _decision_function(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Core decision-function logic (subclasses override this)."""
        ...

    def score_samples(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Anomaly / outlier scores (can be an alias for decision_function).

        Some PU methods (e.g. biased SVM) use score convention where
        *lower* is more positive.  Subclasses may override.
        """
        self._check_is_fitted()
        return self._decision_function(X)

    def predict_proba(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Estimate P(y=1 | x).

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Column 0 = P(y=0|x), column 1 = P(y=1|x).

        Raises
        ------
        NotImplementedError
            The base class does not implement ``predict_proba``.
            Subclasses that provide calibrated probabilities must
            override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_proba. "
            "Use decision_function() or score_samples() instead."
        )

    def predict_label_proba(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray | None:
        """Estimate P(s=1 | x) — the labeling probability.

        Returns
        -------
        np.ndarray of shape (n_samples,) or None
            None means the estimator does not support this.
        """
        return None

    # ── Metadata ───────────────────────────────────────────────────

    def get_pu_metadata(self) -> dict:
        """Return PU-specific metadata dict.

        Includes assumption, scenario, class prior (if known),
        implementation status, source status, and diagnostics.
        """
        return {
            "family": self.family.value,
            "assumption": [a.value for a in self.assumption],
            "scenario": [s.value for s in self.scenario],
            "requires_class_prior": self.requires_class_prior,
            "class_prior": self._class_prior,
            "implementation_status": self.implementation_status.value,
            "source_status": self.source_status.value,
            "backend": self.backend.value,
            "maturity": self.maturity.value,
            "is_fitted": self._is_fitted,
            "n_features": self._X_shape_[1] if self._X_shape_ else None,
        }

    # ── Convenience ────────────────────────────────────────────────

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(f"{self.__class__.__name__} is not fitted. Call fit() first.")

    # NOTE: Do NOT override __sklearn_tags__.  The parent classes
    # (BaseEstimator, ClassifierMixin) return a Tags namedtuple that
    # sklearn 1.6+ expects.  Overriding with a plain dict breaks
    # Pipeline, GridSearchCV, and check_estimator.  PU-specific
    # metadata lives in get_pu_metadata().


# ═════════════════════════════════════════════════════════════════════
# Class Prior Estimator
# ═════════════════════════════════════════════════════════════════════


class BasePriorEstimator(BaseEstimator, ABC):
    """Abstract base class for class-prior estimators π = P(y=1).

    Subclasses **must** implement:

    * ``fit(X, y_pu)``
    * ``estimate()`` → float

    ``confidence_interval()`` is optional; return ``None`` if
    the estimator does not provide uncertainty quantification.
    """

    implementation_status: ImplementationStatus = ImplementationStatus.API_ONLY

    @abstractmethod
    def fit(self, X: np.ndarray | sparse.spmatrix, y_pu: np.ndarray) -> BasePriorEstimator:
        """Estimate class prior π from PU data.

        Returns
        -------
        self
        """
        ...

    @abstractmethod
    def estimate(self) -> float:
        """Return estimated class prior π ∈ (0, 1)."""
        ...

    def confidence_interval(self, alpha: float = 0.05) -> tuple[float, float] | None:
        """Return (lower, upper) confidence bounds for π.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 → 95 % CI).

        Returns
        -------
        (float, float) or None
            ``None`` means the estimator does not support CIs.
        """
        return None


# ═════════════════════════════════════════════════════════════════════
# Propensity Estimator
# ═════════════════════════════════════════════════════════════════════


class BasePropensityEstimator(BaseEstimator, ABC):
    """Abstract base class for labeling-propensity estimators c = P(s=1 | y=1).

    Subclasses **must** implement:

    * ``fit(X, y_pu)``
    * ``estimate()``
    * ``predict_propensity(X)``
    """

    implementation_status: ImplementationStatus = ImplementationStatus.API_ONLY

    @abstractmethod
    def fit(self, X: np.ndarray | sparse.spmatrix, y_pu: np.ndarray) -> BasePropensityEstimator:
        """Fit the propensity model.

        Returns
        -------
        self
        """
        ...

    @abstractmethod
    def estimate(self) -> float | np.ndarray:
        """Return estimated labeling propensity.

        Returns
        -------
        float or np.ndarray
            SCAR constant ``c`` (float) or SAR instance-dependent
            ``c(x)`` (ndarray of shape ``(n_train_samples,)``).
        """
        ...

    @abstractmethod
    def predict_propensity(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Estimate c(x) for new samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Propensity values ∈ (0, 1].
        """
        ...


# ═════════════════════════════════════════════════════════════════════
# PU Loss
# ═════════════════════════════════════════════════════════════════════


class BasePULoss(ABC):
    """Abstract base class for PU risk/loss functions.

    Subclasses **must** implement ``__call__``.

    Notes
    -----
    PU losses do **not** inherit from sklearn's BaseEstimator because
    they are stateless callables; they should, however, support
    ``get_params`` / ``set_params`` if they carry hyper-parameters.
    """

    requires_class_prior: bool = True

    @abstractmethod
    def __call__(
        self,
        positive_scores: np.ndarray,
        unlabeled_scores: np.ndarray,
        *,
        class_prior: float,
    ) -> float:
        """Compute the PU risk given model scores.

        Parameters
        ----------
        positive_scores : np.ndarray of shape (n_P,)
            Model scores (or logits) for labeled-positive samples.
        unlabeled_scores : np.ndarray of shape (n_U,)
            Model scores (or logits) for unlabeled samples.
        class_prior : float
            Class prior π = P(y=1).

        Returns
        -------
        float
            Scalar loss value.
        """
        ...
