"""Regrouping class-prior estimation (ReCPE).

The implementation follows Algorithm 1 in ``Rethinking Class-Prior
Estimation for Positive-Unlabeled Learning``.  The regrouping stage is
independent of the CPE method used afterwards, so ``base_estimator`` may be
replaced by another estimator implementing ``fit(X, y_pu)`` and ``estimate()``.
"""

# The public API follows the project's conventional X/y naming.
# ruff: noqa: N803, N806

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..core.base import BasePriorEstimator
from ..core.exceptions import NotFittedError
from ..core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    ImplementationStatus,
    Maturity,
    Scenario,
    SourceStatus,
)
from ..core.validation import validate_pu_X_y


class _DensityRatioCPE(BasePriorEstimator):
    """Small dependency-free MPE baseline used when no base CPE is supplied.

    A balanced classifier estimates ``P(P|x)``.  With equal sampling priors,
    ``(1 - q(x)) / q(x)`` estimates ``p_U(x) / p_P(x)``; its lower quantile on
    positive samples is a practical estimate of the mixture proportion.
    """

    def __init__(self, quantile: float = 0.01, max_iter: int = 1000) -> None:
        self.quantile = quantile
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y_pu: np.ndarray) -> _DensityRatioCPE:
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="base_cpe")
        if not 0.0 <= self.quantile < 1.0:
            raise ValueError("quantile must be in [0, 1)")
        p = np.asarray(X)[y_pu == 1]
        u = np.asarray(X)[y_pu == 0]
        if len(u) == 0:
            raise ValueError("ReCPE requires at least one unlabeled sample")
        self._model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=self.max_iter, random_state=0)
        )
        self._model.fit(np.vstack([p, u]), np.r_[np.ones(len(p)), np.zeros(len(u))])
        q = np.clip(self._model.predict_proba(p)[:, 1], 1e-6, 1.0 - 1e-6)
        ratio = (1.0 - q) / q
        self._estimate = float(np.clip(np.quantile(ratio, self.quantile), 0.0, 1.0))
        self._is_fitted = True
        return self

    def estimate(self) -> float:
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("_DensityRatioCPE is not fitted. Call fit() first.")
        return self._estimate


class ReCPEEstimator(BasePriorEstimator):
    """Regrouping CPE estimator from Liu et al.

    Parameters
    ----------
    copy_fraction : float, default=0.1
        Fraction of unlabeled examples copied into the positive sample.
    base_estimator : estimator or None, default=None
        A fitted-by-``fit`` CPE estimator implementing ``estimate``.  ``None``
        uses the built-in classifier-based mixture-proportion baseline.
    classifier : estimator or None, default=None
        Binary classifier distinguishing positive samples from unlabeled ones.
        Its positive-class probability is used to rank unlabeled examples.
    classifier_max_iter : int, default=1000
        Maximum iterations for the default logistic classifier.
    """

    family: AlgorithmFamily = AlgorithmFamily.CLASS_PRIOR_ESTIMATION
    assumption: tuple[Assumption, ...] = (Assumption.SCAR,)
    scenario: tuple[Scenario, ...] = (Scenario.SINGLE_TRAINING_SET, Scenario.CASE_CONTROL)
    requires_class_prior: bool = False
    implementation_status: ImplementationStatus = ImplementationStatus.NATIVE
    source_status: SourceStatus = SourceStatus.OFFICIAL_EXACT
    backend: Backend = Backend.NUMPY
    maturity: Maturity = Maturity.STABLE

    def __init__(
        self,
        copy_fraction: float = 0.1,
        base_estimator: Any = None,
        classifier: Any = None,
        classifier_max_iter: int = 1000,
    ) -> None:
        self.copy_fraction = copy_fraction
        self.base_estimator = base_estimator
        self.classifier = classifier
        self.classifier_max_iter = classifier_max_iter

    def fit(self, X: np.ndarray, y_pu: np.ndarray) -> ReCPEEstimator:
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="ReCPEEstimator")
        if not 0.0 < self.copy_fraction < 1.0:
            raise ValueError("copy_fraction must be strictly between 0 and 1")
        X = np.asarray(X, dtype=float)
        p = X[y_pu == 1]
        u = X[y_pu == 0]
        if len(u) == 0:
            raise ValueError("ReCPE requires at least one unlabeled sample")

        if self.classifier is None:
            ranker = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=self.classifier_max_iter, random_state=0),
            )
        else:
            ranker = clone(self.classifier)
        ranker.fit(np.vstack([p, u]), np.r_[np.ones(len(p)), np.zeros(len(u))])
        positive_probability = ranker.predict_proba(u)[:, 1]
        n_copy = max(1, int(np.ceil(self.copy_fraction * len(u))))
        selected = np.argsort(positive_probability)[-n_copy:]

        regrouped_X = np.vstack([p, u])
        regrouped_y = np.r_[np.ones(len(p), dtype=int), np.zeros(len(u), dtype=int)]
        regrouped_y[len(p) + selected] = 1

        if self.base_estimator is None:
            base = _DensityRatioCPE()
        else:
            try:
                base = clone(self.base_estimator)
            except TypeError:
                # Lightweight user-defined adapters do not always inherit
                # sklearn's BaseEstimator but can still satisfy the protocol.
                base = self.base_estimator
        base.fit(regrouped_X, regrouped_y)

        self.base_estimator_ = base
        self.classifier_ = ranker
        self.selected_indices_ = selected
        self.copy_count_ = n_copy
        self.class_prior_ = float(np.clip(base.estimate(), 0.0, 1.0))
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def estimate(self) -> float:
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("ReCPEEstimator is not fitted. Call fit() first.")
        return self.class_prior_

    def get_metadata(self) -> dict[str, Any]:
        """Return implementation diagnostics for reproducibility."""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("ReCPEEstimator is not fitted. Call fit() first.")
        return {
            "method": "ReCPE",
            "copy_fraction": self.copy_fraction,
            "copy_count": self.copy_count_,
            "class_prior": self.class_prior_,
            "base_estimator": self.base_estimator_.__class__.__name__,
        }
