# ruff: noqa: N803, N806, E501

"""Instance-dependent labeling-bias estimation (LBE).

This module implements the paper's latent-variable likelihood with a linear
logistic ``P(y=1|x)`` model and a logistic ``P(s=1|y=1,x)`` propensity model.
The alternating weighted-logistic updates are deliberately explicit so the
estimated classifier and propensity can both be inspected.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ...core.base import BasePUClassifier
from ...core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    ImplementationStatus,
    Maturity,
    SampleWeightSupport,
    Scenario,
    SourceStatus,
)
from ...core.validation import check_scalar_in_range, validate_pu_X_y


class LBEClassifier(BasePUClassifier):
    """Estimate class posterior and instance-dependent labeling propensity."""

    family = AlgorithmFamily.BIAS_AWARE
    label_semantics = "pu"
    assumption = (Assumption.SAR,)
    scenario = (Scenario.SINGLE_TRAINING_SET, Scenario.SELECTION_BIASED)
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_EXACT
    backend = Backend.SKLEARN
    maturity = Maturity.RESEARCH
    sample_weight_support = SampleWeightSupport.IGNORED

    def __init__(self, *, max_iter: int = 1000, n_em_iter: int = 20, C: float = 1.0) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.n_em_iter = n_em_iter
        self.C = C

    @staticmethod
    def _weighted_binary_fit(model, X, q):
        # A soft target q is represented by two weighted copies.
        X2 = np.vstack([X, X])
        y2 = np.r_[np.ones(len(X)), np.zeros(len(X))]
        w2 = np.r_[q, 1.0 - q]
        model.fit(X2, y2, logisticregression__sample_weight=w2)

    @staticmethod
    def _positive_posterior(s, p, c):
        """E-step: positive class posterior conditional on the observed label."""
        return np.where(s == 1, 1.0, p * (1.0 - c) / (1.0 - p * c))

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        """Fit the LBE instance-dependent labeling-bias model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.  Must be dense.
        y_pu : array-like of shape (n_samples,)
            PU labels.  +1 = labeled positive, 0 = unlabeled.
        class_prior : float, optional
            Initialize the unlabeled latent posterior; not a prior constraint.
        sample_weight : ignored (present for sklearn compatibility; PU risk estimators do not use instance weights)

        Returns
        -------
        self : LBEClassifier
        """
        self._is_fitted = False
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="LBEClassifier")
        X = np.asarray(X, dtype=float)
        for name in ("max_iter", "n_em_iter"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.C, (bool, np.bool_))
            or not isinstance(self.C, Real)
            or not np.isfinite(self.C)
            or self.C <= 0
        ):
            raise ValueError("C must be finite and positive")
        if class_prior is not None and (
            isinstance(class_prior, (bool, np.bool_)) or not isinstance(class_prior, Real)
        ):
            raise ValueError("class_prior must be a scalar in (0, 1)")
        s = (y_pu == 1).astype(float)
        p0 = float(class_prior) if class_prior is not None else max(0.05, min(0.95, s.mean() * 2.0))
        check_scalar_in_range(p0, 0.0, 1.0, "class_prior", inclusive=False)
        self.classifier_ = make_pipeline(
            StandardScaler(), LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=0)
        )
        self.propensity_model_ = make_pipeline(
            StandardScaler(), LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=0)
        )
        q = np.where(s == 1, 1.0, p0)
        self.converged_ = False
        self.observed_log_likelihood_history_ = []
        for iteration in range(self.n_em_iter):
            self._weighted_binary_fit(self.classifier_, X, q)
            p = np.clip(self.classifier_.predict_proba(X)[:, 1], 1e-5, 1 - 1e-5)
            positive_weight = np.where(s == 1, 1.0, q)
            self.propensity_model_.fit(X, s, logisticregression__sample_weight=positive_weight)
            c = np.clip(self.propensity_model_.predict_proba(X)[:, 1], 1e-5, 1 - 1e-5)
            q_new = self._positive_posterior(s, p, c)
            self.observed_log_likelihood_history_.append(
                float(np.sum(s * (np.log(p) + np.log(c)) + (1.0 - s) * np.log1p(-p * c)))
            )
            self.n_iter_ = iteration + 1
            if np.max(np.abs(q_new - q)) < 1e-5:
                q = q_new
                self.converged_ = True
                break
            q = q_new
        self._latent_positive_probability_ = q
        self._class_prior = float(np.mean(p))
        self.classes_ = np.array([0, 1])
        self._X_shape_ = X.shape
        self._is_fitted = True
        return self

    def _decision_function(self, X):
        return self.classifier_.decision_function(X)

    def _predict(self, X):
        return (self.classifier_.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        """Return class probabilities, not observed-label probabilities."""
        self._check_is_fitted()
        return self.classifier_.predict_proba(X)

    def predict_label_proba(self, X):
        """Return P(S=1|X) = P(Y=1|X) P(S=1|Y=1,X)."""
        self._check_is_fitted()
        p = self.classifier_.predict_proba(X)[:, 1]
        c = self.propensity_model_.predict_proba(X)[:, 1]
        return p * c
