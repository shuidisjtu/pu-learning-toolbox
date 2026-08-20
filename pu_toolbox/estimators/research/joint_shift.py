# ruff: noqa: N803, N806

"""Research-stage PU learning under source/target joint distribution shift.

The implementation follows the relative joint importance-weight idea of
Kumagai et al. (AISTATS 2025), while using probabilistic class membership
and class-conditional logistic domain models as a practical sklearn solver.
It is deliberately exposed as research-stage rather than registry-stable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    ImplementationStatus,
    Maturity,
    SampleWeightSupport,
    Scenario,
    SourceStatus,
)
from pu_toolbox.core.validation import check_scalar_in_range, validate_pu_X_y
from pu_toolbox.estimators.classic import ElkanNotoClassifier

__all__ = ["JointShiftPUClassifier", "relative_joint_weight"]


def relative_joint_weight(ratio: Any, *, alpha: float) -> np.ndarray:
    """Apply ``r / (alpha*r + 1-alpha)`` with its exact ``1/alpha`` bound."""
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must lie in (0, 1].")
    values = np.asarray(ratio, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("ratio must be a finite non-negative 1-D array.")
    return values / (alpha * values + (1.0 - alpha))


class JointShiftPUClassifier(BasePUClassifier):
    """Alternating class-conditional relative joint-weight PU classifier.

    ``fit`` requires both source and target PU datasets plus both class priors.
    Each iteration estimates positive/negative conditional domain ratios from
    soft class membership, forms a bounded joint ratio, and refits a PU model
    on target data plus weighted source data.
    """

    family = AlgorithmFamily.BIAS_AWARE
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.SELECTION_BIASED,)
    requires_class_prior = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.NOT_FOUND
    backend = Backend.SKLEARN
    maturity = Maturity.RESEARCH
    sample_weight_support = SampleWeightSupport.SUPPORTED

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        target_mix: float = 0.5,
        max_iter: int = 5,
        tolerance: float = 1e-3,
        n_cv_folds: int = 3,
        probability_clip: float = 1e-5,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must lie in (0, 1].")
        if not np.isfinite(target_mix) or not 0 < target_mix <= 1:
            raise ValueError("target_mix must lie in (0, 1].")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1.")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative.")
        if n_cv_folds < 2:
            raise ValueError("n_cv_folds must be >= 2.")
        if not np.isfinite(probability_clip) or not 0 < probability_clip < 0.5:
            raise ValueError("probability_clip must lie in (0, 0.5).")
        self.alpha = alpha
        self.target_mix = target_mix
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_cv_folds = n_cv_folds
        self.probability_clip = probability_clip
        self.random_state = random_state

    def fit(
        self,
        X: Any,
        y_pu: Any,
        *,
        X_target: Any | None = None,
        y_target_pu: Any | None = None,
        class_prior: float | None = None,
        target_class_prior: float | None = None,
        sample_weight: Any | None = None,
    ) -> JointShiftPUClassifier:
        if X_target is None or y_target_pu is None:
            raise ValueError("X_target and y_target_pu are required for joint-shift training.")
        if class_prior is None or target_class_prior is None:
            raise ValueError("class_prior and target_class_prior are both required.")
        check_scalar_in_range(class_prior, 0.0, 1.0, "class_prior", inclusive=False)
        check_scalar_in_range(target_class_prior, 0.0, 1.0, "target_class_prior", inclusive=False)
        X_source, source_labels = validate_pu_X_y(
            X, y_pu, estimator_name="JointShiftPUClassifier(source)"
        )
        X_target_checked, target_labels = validate_pu_X_y(
            X_target, y_target_pu, estimator_name="JointShiftPUClassifier(target)"
        )
        if X_source.shape[1] != X_target_checked.shape[1]:
            raise ValueError("Source and target must have the same number of features.")
        if sample_weight is not None:
            supplied = np.asarray(sample_weight, dtype=float)
            if supplied.shape != (len(source_labels),):
                raise ValueError(
                    f"sample_weight must have shape ({len(source_labels)},); got {supplied.shape}."
                )
            if not np.isfinite(supplied).all() or np.any(supplied < 0):
                raise ValueError("sample_weight must be finite and non-negative.")
        else:
            supplied = np.ones(len(source_labels), dtype=float)
        combined_X = (
            sparse.vstack([X_source, X_target_checked])
            if sparse.issparse(X_source)
            else np.vstack([X_source, X_target_checked])
        )
        combined_y = np.concatenate([source_labels, target_labels])
        n_source = len(source_labels)
        n_target = len(target_labels)
        initial_weight = np.concatenate(
            [np.full(n_source, 0.5 / n_source), np.full(n_target, 0.5 / n_target)]
        )
        model = self._new_pu_model().fit(combined_X, combined_y, sample_weight=initial_weight)
        previous_weights = np.ones(n_source, dtype=float)
        trace: list[dict[str, float]] = []
        for iteration in range(1, self.max_iter + 1):
            source_p = _calibrate_prior(model.decision_function(X_source), float(class_prior))
            target_p = _calibrate_prior(
                model.decision_function(X_target_checked), float(target_class_prior)
            )
            positive_ratio = self._conditional_ratio(
                X_source, X_target_checked, source_p, target_p
            ) * (float(target_class_prior) / float(class_prior))
            negative_ratio = self._conditional_ratio(
                X_source, X_target_checked, 1.0 - source_p, 1.0 - target_p
            ) * ((1.0 - float(target_class_prior)) / (1.0 - float(class_prior)))
            joint_ratio = source_p * positive_ratio + (1.0 - source_p) * negative_ratio
            weights = relative_joint_weight(joint_ratio, alpha=self.alpha) * supplied
            weights /= np.mean(weights)
            source_arm = weights * (1.0 - self.target_mix) / n_source
            target_arm = np.full(n_target, self.target_mix / n_target)
            model = self._new_pu_model().fit(
                combined_X,
                combined_y,
                sample_weight=np.concatenate([source_arm, target_arm]),
            )
            weight_change = float(np.mean(np.abs(weights - previous_weights)))
            trace.append(
                {
                    "iteration": float(iteration),
                    "mean_absolute_weight_change": weight_change,
                    "effective_sample_fraction": float(
                        np.square(weights.sum()) / (len(weights) * np.square(weights).sum())
                    ),
                    "maximum_weight": float(weights.max()),
                }
            )
            previous_weights = weights
            if weight_change <= self.tolerance:
                break
        self.model_ = model
        self.relative_joint_weights_ = previous_weights
        self.training_trace_ = tuple(trace)
        self.n_iter_ = len(trace)
        self.source_class_prior_ = float(class_prior)
        self.target_class_prior_ = float(target_class_prior)
        self._class_prior = float(target_class_prior)
        self._X_shape_ = (n_source + n_target, X_source.shape[1])
        self.classes_ = np.array([0, 1])
        self._is_fitted = True
        return self

    def _new_pu_model(self) -> ElkanNotoClassifier:
        return ElkanNotoClassifier(
            n_cv_folds=self.n_cv_folds,
            random_state=self.random_state,
        )

    def _conditional_ratio(
        self,
        X_source: Any,
        X_target: Any,
        source_membership: np.ndarray,
        target_membership: np.ndarray,
    ) -> np.ndarray:
        domain_X = (
            sparse.vstack([X_source, X_target])
            if sparse.issparse(X_source)
            else np.vstack([X_source, X_target])
        )
        domain_y = np.concatenate(
            [
                np.zeros(len(source_membership), dtype=int),
                np.ones(len(target_membership), dtype=int),
            ]
        )
        membership = np.concatenate([source_membership, target_membership])
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(domain_X, domain_y, sample_weight=np.maximum(membership, self.probability_clip))
        q = np.clip(
            model.predict_proba(X_source)[:, 1],
            self.probability_clip,
            1.0 - self.probability_clip,
        )
        source_mass = float(source_membership.sum())
        target_mass = float(target_membership.sum())
        return q / (1.0 - q) * source_mass / target_mass

    def _predict(self, X: Any) -> np.ndarray:
        return self.model_.predict(X)

    def _decision_function(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.decision_function(X), dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        scores = np.clip(self._decision_function(X), 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])

    def get_pu_metadata(self) -> dict[str, Any]:
        metadata = super().get_pu_metadata()
        metadata.update(
            {
                "solver": "alternating_soft_class_conditional_domain_ratio",
                "relative_weight_alpha": self.alpha,
                "target_mix": self.target_mix,
                "n_iter": getattr(self, "n_iter_", None),
                "guarantee": "research_joint_shift_approximation",
            }
        )
        return metadata


def _calibrate_prior(probability: Any, prior: float) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(values / (1.0 - values))
    lower, upper = -30.0, 30.0
    for _ in range(60):
        middle = (lower + upper) / 2.0
        mean = float(np.mean(1.0 / (1.0 + np.exp(-(logits + middle)))))
        if mean < prior:
            lower = middle
        else:
            upper = middle
    return 1.0 / (1.0 + np.exp(-(logits + (lower + upper) / 2.0)))
