# ruff: noqa: N803
"""SCAR-constrained diagnostic, NOT a replacement or registered LBE method."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.core.tags import Assumption, Scenario
from pu_toolbox.core.validation import validate_pu_X_y


def scar_objective(parameters, features, observed, propensity, C):
    """Observed Bernoulli likelihood gradient; no hidden Y enters this objective."""
    w, intercept = parameters[:-1], parameters[-1]
    logits = features @ w + intercept
    p = expit(logits)
    # Stable log(1-c*p) even when p saturates; c is strictly below one.
    log_label = np.log(propensity) - np.logaddexp(0, -logits)
    log_unlabeled = np.log1p(-propensity * p)
    value = -np.mean(observed * log_label + (1 - observed) * log_unlabeled)
    value += np.dot(w, w) / (2 * C * len(observed))
    residual = (propensity * p - observed) * (1 - p) / (1 - propensity * p)
    gradient = np.r_[
        features.T @ residual / len(observed) + w / (C * len(observed)), np.mean(residual)
    ]
    return float(value), gradient


class SCARFixedPropensityDiagnostic(BasePUClassifier):
    """Fix c=observed_rate/known_pi under SCAR, optimizing only class logits.

    This adds a modeling assumption and removes LBE's flexible propensity
    branch. It is a diagnostic contrast, not a fair single-variable capacity
    ablation or a new paper reproduction. No estimator registration.
    """

    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.SINGLE_TRAINING_SET,)
    requires_class_prior = True

    def __init__(self, *, C=1.0, max_iter=1000):
        super().__init__()
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y_pu, *, class_prior=None):  # noqa: N803
        self._is_fitted = False
        features, observed = validate_pu_X_y(X, y_pu, accept_sparse=False)
        if class_prior is None or not np.isfinite(class_prior) or not 0 < class_prior < 1:
            raise ValueError("SCAR diagnostic requires known population prior in (0,1)")
        if (
            not np.isfinite(self.C)
            or self.C <= 0
            or type(self.max_iter) is not int
            or self.max_iter < 1
        ):
            raise ValueError("SCAR diagnostic needs finite positive C and integer max_iter")
        self.propensity_ = float(np.mean(observed) / class_prior)
        if not 0 < self.propensity_ < 1:
            raise ValueError("observed rate / population prior must lie in (0,1)")
        self.scaler_ = StandardScaler().fit(features)
        scaled = self.scaler_.transform(features)
        initial = np.r_[np.zeros(scaled.shape[1]), logit(class_prior)]
        result = minimize(
            scar_objective,
            initial,
            args=(scaled, observed, self.propensity_, self.C),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "gtol": 1e-7, "ftol": 1e-12},
        )
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise ValueError("SCAR diagnostic solver produced nonfinite values")
        self.parameters_ = result.x
        self.optimizer_success_ = bool(result.success)
        self.optimizer_message_ = str(result.message)
        self.n_iter_ = int(result.nit)
        self.gradient_max_abs_ = float(np.max(np.abs(result.jac)))
        self._class_prior = float(class_prior)
        self._X_shape_ = features.shape
        self.classes_ = np.array([0, 1])
        self._is_fitted = True
        return self

    def _decision_function(self, X):  # noqa: N803
        features = self.scaler_.transform(X)
        return features @ self.parameters_[:-1] + self.parameters_[-1]

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)
