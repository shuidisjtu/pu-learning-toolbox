# ruff: noqa: N803, N806, E501

"""A linear P/U source-classification baseline for selection-biased PU data.

This baseline fits a logistic classifier to the observed P-vs-U indicator. It
is not the paper's partially identified scoring algorithm and its sigmoid
output is not a calibrated estimate of the latent class posterior.
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
    SampleWeightSupport,
    Scenario,
    SourceStatus,
)
from ...core.validation import validate_pu_X_y


class PUSBClassifier(BasePUClassifier):
    """Fit a linear observed-P-vs-U baseline; not a paper-faithful PUSB model."""

    family = AlgorithmFamily.BIAS_AWARE
    label_semantics = "pu"
    assumption = (Assumption.SAR,)
    scenario = (Scenario.CASE_CONTROL, Scenario.SELECTION_BIASED)
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_RELATED
    backend = Backend.SKLEARN
    maturity = Maturity.RESEARCH
    sample_weight_support = SampleWeightSupport.SUPPORTED

    def __init__(self, *, threshold: float = 0.5, C: float = 1.0, max_iter: int = 1000) -> None:
        super().__init__()
        self.threshold = threshold
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="PUSBClassifier")
        if not 0.0 < self.threshold < 1.0 or self.C <= 0:
            raise ValueError("threshold must be in (0, 1) and C must be positive")
        X = np.asarray(X, dtype=float)
        u = X[y_pu == 0]
        if len(u) == 0:
            raise ValueError("PUSBClassifier requires unlabeled samples")
        self.model_ = make_pipeline(
            StandardScaler(), LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=0)
        )
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        fit_params = {} if weights is None else {"logisticregression__sample_weight": weights}
        self.model_.fit(X, y_pu, **fit_params)
        self.classes_ = np.array([0, 1])
        self._class_prior = class_prior
        self._X_shape_ = X.shape
        self._is_fitted = True
        return self

    def _decision_function(self, X):
        return self.model_.decision_function(X)

    def _predict(self, X):
        return (self.model_.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
