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
from .kernel_mean import KernelMeanPriorEstimator


class _DensityRatioCPE(BasePriorEstimator):
    """Small dependency-free MPE baseline used when no base CPE is supplied.

    A balanced classifier estimates ``P(P|x)``.  With equal sampling priors,
    ``(1 - q(x)) / q(x)`` estimates ``p_U(x) / p_P(x)``; its lower quantile on
    positive samples is a practical estimate of the mixture proportion.
    """

    estimate_target = "native_unlabeled"
    estimation_pool = "native_unlabeled"

    def __init__(self, quantile: float = 0.25, max_iter: int = 1000) -> None:
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
    """Regrouping CPE estimator from Yao et al.

    Parameters
    ----------
    copy_fraction : float, default=0.1
        Fraction of unlabeled examples copied into the positive sample.
    base_estimator : estimator or None, default=None
        A fitted-by-``fit`` CPE estimator implementing ``estimate``.  ``None``
        uses the built-in kernel-mean KM2 mixture-proportion estimator.
    classifier : estimator or None, default=None
        Binary classifier distinguishing positive samples from unlabeled ones.
        Its positive-class probability is used to rank unlabeled examples.
    classifier_max_iter : int, default=1000
        Maximum iterations for the default logistic classifier.

    Notes
    -----
    Selected U rows are copied into the positive component; the complete U
    remains the mixture. The base estimator returns the regrouped-component
    weight, used as an approximation to the original native-U positive prior.
    For region A its population analogue is
    ``pi_regrouped = pi_U + (1-pi_U) * P_negative(A)`` (paper Eq. 7).
    Metadata records this approximation and the copy policy.
    """

    family: AlgorithmFamily = AlgorithmFamily.CLASS_PRIOR_ESTIMATION
    assumption: tuple[Assumption, ...] = (Assumption.SCAR,)
    scenario: tuple[Scenario, ...] = (Scenario.SINGLE_TRAINING_SET, Scenario.CASE_CONTROL)
    requires_class_prior: bool = False
    implementation_status: ImplementationStatus = ImplementationStatus.NATIVE
    source_status: SourceStatus = SourceStatus.OFFICIAL_EXACT
    backend: Backend = Backend.NUMPY
    maturity: Maturity = Maturity.STABLE
    estimate_target = "native_unlabeled"
    estimation_pool = "native_unlabeled"

    def __init__(
        self,
        copy_fraction: float = 0.1,
        base_estimator: Any = None,
        classifier: Any = None,
        classifier_max_iter: int = 1000,
    ) -> None:
        # v1.2.1: the default base CPE is the official-style kernel-mean KM2
        # (the paper's reference base); the previous density-ratio + 1%
        # quantile baseline collapsed on regular SCAR data.  KM2 keeps the
        # estimate from collapsing while staying conservative.
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

        component = np.vstack([p, u[selected]])
        regrouped_X = np.vstack([component, u])
        regrouped_y = np.r_[np.ones(len(component), dtype=int), np.zeros(len(u), dtype=int)]

        if self.base_estimator is None:
            base = KernelMeanPriorEstimator(variant="km2")
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
            "estimate_target": self.estimate_target,
            "estimation_pool": self.estimation_pool,
            "copy_policy": "append_to_positive_keep_unlabeled",
            "estimate_kind": "regrouping_approximation",
            "auxiliary_target": "regrouped_component_weight",
        }
