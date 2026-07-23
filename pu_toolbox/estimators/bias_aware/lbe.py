# ruff: noqa: N803, N806, E501

"""Instance-dependent labeling-bias estimation (LBE).

This module implements the paper's latent-variable likelihood with a linear
logistic ``P(y=1|x)`` model and a logistic ``P(s=1|y=1,x)`` propensity model.
The alternating weighted-logistic updates are deliberately explicit so the
estimated classifier and propensity can both be inspected.
"""

from __future__ import annotations

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
    Scenario,
    SourceStatus,
)
from ...core.validation import validate_pu_X_y


class LBEClassifier(BasePUClassifier):
    """Estimate class posterior and instance-dependent labeling propensity."""

    family = AlgorithmFamily.BIAS_AWARE
    assumption = (Assumption.SAR,)
    scenario = (Scenario.SINGLE_TRAINING_SET, Scenario.SELECTION_BIASED)
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_EXACT
    backend = Backend.SKLEARN
    maturity = Maturity.RESEARCH

    def __init__(self, *, max_iter: int = 1000, n_em_iter: int = 20, C: float = 1.0) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.n_em_iter = n_em_iter
        self.C = C

    @staticmethod
    def _weighted_binary_fit(model, X, q, *, random_state=0):
        # A soft target q is represented by two weighted copies.
        X2 = np.vstack([X, X])
        y2 = np.r_[np.ones(len(X)), np.zeros(len(X))]
        w2 = np.r_[q, 1.0 - q]
        model.fit(X2, y2, logisticregression__sample_weight=w2)

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="LBEClassifier")
        X = np.asarray(X, dtype=float)
        if self.n_em_iter < 1 or self.C <= 0:
            raise ValueError("n_em_iter must be >= 1 and C must be positive")
        s = (y_pu == 1).astype(float)
        p0 = float(class_prior) if class_prior is not None else max(0.05, min(0.95, s.mean() * 2.0))
        if not 0.0 < p0 < 1.0:
            raise ValueError("class_prior must be in (0, 1)")
        self.classifier_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=0))
        self.propensity_model_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=0))
        q = np.where(s == 1, 1.0, p0)
        for _ in range(self.n_em_iter):
            self._weighted_binary_fit(self.classifier_, X, q)
            p = np.clip(self.classifier_.predict_proba(X)[:, 1], 1e-5, 1 - 1e-5)
            positive_weight = np.where(s == 1, 1.0, q)
            self.propensity_model_.fit(X, s, logisticregression__sample_weight=positive_weight)
            c = np.clip(self.propensity_model_.predict_proba(X)[:, 1], 1e-5, 1 - 1e-5)
            q_new = np.where(s == 1, 1.0, p * (1.0 - c) / np.clip(1.0 - p * c, 1e-5, None))
            if np.max(np.abs(q_new - q)) < 1e-5:
                q = q_new
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
        return self.classifier_.predict_proba(X)

    def predict_label_proba(self, X):
        p = self.classifier_.predict_proba(X)[:, 1]
        c = self.propensity_model_.predict_proba(X)[:, 1]
        return p * c
