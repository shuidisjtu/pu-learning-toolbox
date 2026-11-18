# ruff: noqa: N803, N806

"""Elkan–Noto PU classifier (KDD 2008).

Clean-room implementation with pulearn/pulearn (BSD-3-Clause) as a reference.
"""

from __future__ import annotations

import warnings
from numbers import Integral, Real
from typing import Literal

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted as _sklearn_check_is_fitted

from ...core.base import BasePUClassifier
from ...core.config import CLASS_PRIOR_CLIP_EPS, PROPENSITY_CLIP_EPS
from ...core.exceptions import ValidationError
from ...core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    Maturity,
    SampleWeightSupport,
    Scenario,
    SourceStatus,
)
from ...core.validation import validate_pu_X_y, validate_sample_weight


class ElkanNotoClassifier(BasePUClassifier):
    """Elkan–Noto PU classifier with probability correction or weighted retraining.

    Parameters
    ----------
    base_estimator : sklearn estimator, optional
        Binary estimator. Defaults to ``LogisticRegression()``.
    calibration_method : {"sigmoid", "isotonic"}, default "sigmoid"
        Calibration for estimators other than ``LogisticRegression``.
    internal_cv_folds : int, default 3
        Folds for estimating labeling propensity. Must be at least 2, with
        at least that many labeled positives and that many unlabeled samples.
    eps : float, default 1e-12
        Probability clipping margin for computing retraining weights.
    mode : {"probability_correction", "weighted_retraining"}, default "weighted_retraining"
        Scale labeling probabilities by ``1 / c`` or refit on soft labels.
    random_state : int or None, default None
        Seed for stratified splits and the default estimator.

    Attributes
    ----------
    propensity_ : float
        Estimated labeling propensity ``c = P(s=1 | y=1)``.
    class_prior_ : float
        Estimated population prior ``P(y=1)`` under single-training-set sampling.
    label_classifier_ : sklearn estimator
        Fitted model for ``P(s=1 | x)``.
    final_estimator_ : sklearn estimator or None
        Retrained classifier; ``None`` in probability-correction mode.
    """

    family: AlgorithmFamily = AlgorithmFamily.CLASSIC_CALIBRATION
    label_semantics = "pu"
    assumption: tuple[Assumption, ...] = (Assumption.SCAR,)
    scenario: tuple[Scenario, ...] = (Scenario.SINGLE_TRAINING_SET,)
    requires_class_prior: bool = False
    source_status: SourceStatus = SourceStatus.THIRD_PARTY_ONLY
    backend: Backend = Backend.SKLEARN
    maturity: Maturity = Maturity.STABLE
    sample_weight_support: SampleWeightSupport = SampleWeightSupport.SUPPORTED

    def __init__(
        self,
        base_estimator=None,
        calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
        internal_cv_folds: int = 3,
        eps: float = 1e-12,
        mode: Literal["probability_correction", "weighted_retraining"] = "weighted_retraining",
        random_state=None,
    ) -> None:
        super().__init__()
        self.base_estimator = base_estimator
        self.calibration_method = calibration_method
        self.eps = eps
        self.mode = mode
        self.random_state = random_state

        if not isinstance(internal_cv_folds, Integral) or internal_cv_folds < 2:
            raise ValidationError(
                f"internal_cv_folds must be an integer >= 2; got {internal_cv_folds!r}."
            )
        self.internal_cv_folds = internal_cv_folds

        if not isinstance(eps, Real) or not eps > 0:
            raise ValidationError(f"eps must be > 0; got {eps!r}.")

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> ElkanNotoClassifier:
        """Fit on labeled positives and unlabeled samples.

        ``class_prior`` is ignored. ``sample_weight`` applies to the label
        models, not directly to the final weighted refit.
        """
        X, y_pu = validate_pu_X_y(X, y_pu, estimator_name="ElkanNotoClassifier")

        self._X_shape_ = X.shape
        self.classes_ = np.array([0, 1])

        n_samples = X.shape[0]
        sample_weight = validate_sample_weight(sample_weight, n_samples)
        labeled = y_pu == 1
        n_labeled = int(np.count_nonzero(labeled))
        n_unlabeled = n_samples - n_labeled
        if min(n_labeled, n_unlabeled) < self.internal_cv_folds:
            raise ValidationError(
                f"Need at least internal_cv_folds={self.internal_cv_folds} samples of each "
                f"PU label, but found {n_labeled} labeled positives and {n_unlabeled} "
                "unlabeled samples. Reduce internal_cv_folds or collect more data."
            )

        self.propensity_ = self._estimate_propensity(X, y_pu, sample_weight)

        base = self._resolve_base_estimator()
        self.label_classifier_ = self._wrap_with_calibration(base, y_pu)
        if sample_weight is None:
            self.label_classifier_.fit(X, y_pu)
        else:
            self.label_classifier_.fit(X, y_pu, sample_weight=sample_weight)

        self.class_prior_ = float(
            np.clip(
                n_labeled / (n_samples * self.propensity_),
                CLASS_PRIOR_CLIP_EPS,
                1.0 - CLASS_PRIOR_CLIP_EPS,
            )
        )

        if self.mode == "weighted_retraining":
            self.final_estimator_ = self._retrain_weighted(X, labeled)
        else:
            self.final_estimator_ = None

        self._is_fitted = True
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return (self._decision_function(X) >= 0.5).astype(int)

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        if self.mode == "weighted_retraining" and self.final_estimator_ is not None:
            return self.final_estimator_.predict_proba(X)[:, 1]
        return self._predict_label_proba(X) / self.propensity_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return negative- and positive-class scores in that column order.

        In probability-correction mode, the scores are not clipped to [0, 1].
        """
        self._check_is_fitted()
        positive_scores = self._decision_function(X)
        return np.column_stack([1.0 - positive_scores, positive_scores])

    def predict_label_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return ``P(s=1 | x)``, or ``None`` in weighted-retraining mode."""
        self._check_is_fitted()
        if self.mode == "weighted_retraining":
            return None
        return self._predict_label_proba(X)

    def get_pu_metadata(self) -> dict:
        """Return PU-specific metadata including propensity and mode."""
        meta = super().get_pu_metadata()
        meta["propensity"] = getattr(self, "propensity_", None)
        meta["mode"] = self.mode
        return meta

    def _resolve_base_estimator(self):
        if self.base_estimator is None:
            return LogisticRegression(random_state=self.random_state)
        return clone(self.base_estimator)

    def _wrap_with_calibration(self, base, labels: np.ndarray):
        if isinstance(base, LogisticRegression):
            return base

        min_label_count = int(np.bincount(labels, minlength=2).min())
        cv = min(self.internal_cv_folds, 5, min_label_count)
        if cv < 2:
            raise ValidationError(
                "Probability calibration requires at least two samples of each label "
                "in its training data."
            )
        return CalibratedClassifierCV(estimator=base, method=self.calibration_method, cv=cv)

    def _estimate_propensity(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> float:
        """Estimate ``c`` without systematically changing the labeling rate in training folds."""
        splitter = StratifiedKFold(
            n_splits=self.internal_cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        labeled = y_pu == 1
        fold_scores: list[np.ndarray] = []
        base = self._resolve_base_estimator()

        for train_idx, holdout_idx in splitter.split(X, y_pu):
            X_train, s_train = X[train_idx], y_pu[train_idx]
            g_model = self._wrap_with_calibration(clone(base), s_train)
            if sample_weight is None:
                g_model.fit(X_train, s_train)
            else:
                g_model.fit(X_train, s_train, sample_weight=sample_weight[train_idx])

            holdout_pos = holdout_idx[labeled[holdout_idx]]
            fold_scores.append(g_model.predict_proba(X[holdout_pos])[:, 1])

        c_hat = float(np.mean(np.concatenate(fold_scores)))
        c_hat = float(np.clip(c_hat, PROPENSITY_CLIP_EPS, 1.0))

        if c_hat <= PROPENSITY_CLIP_EPS:
            warnings.warn(
                f"Estimated propensity c={c_hat:.2e} is near zero. "
                "Check calibration quality, data split, or SCAR assumption.",
                UserWarning,
                stacklevel=2,
            )

        return c_hat

    def _predict_label_proba(self, X: np.ndarray) -> np.ndarray:
        _sklearn_check_is_fitted(self.label_classifier_)
        return self.label_classifier_.predict_proba(X)[:, 1]

    def _retrain_weighted(self, X: np.ndarray, labeled: np.ndarray):
        unlabeled = ~labeled
        positive_weight = self._compute_weights(self._predict_label_proba(X)[unlabeled])
        X_labeled, X_unlabeled = X[labeled], X[unlabeled]

        X_aug = np.vstack([X_labeled, X_unlabeled, X_unlabeled])
        y_aug = np.concatenate([np.ones(len(X), dtype=int), np.zeros(len(X_unlabeled), dtype=int)])
        weights = np.concatenate([np.ones(len(X_labeled)), positive_weight, 1.0 - positive_weight])

        final = clone(self._resolve_base_estimator())
        if not hasattr(final, "predict_proba"):
            final = self._wrap_with_calibration(final, y_aug)
        final.fit(X_aug, y_aug, sample_weight=weights)
        return final

    def _compute_weights(self, g_unlabeled: np.ndarray) -> np.ndarray:
        """Estimate ``P(y=1 | x, s=0)`` for the unlabeled samples."""
        c = self.propensity_
        g = np.clip(g_unlabeled, self.eps, 1.0 - self.eps)

        n_clipped = int(np.sum((g_unlabeled < self.eps) | (g_unlabeled > 1.0 - self.eps)))
        if n_clipped > 0:
            warnings.warn(
                f"{n_clipped}/{len(g_unlabeled)} g(x) values were clipped "
                f"to [{self.eps}, {1.0 - self.eps}] for weight computation. "
                "Consider checking calibration quality.",
                UserWarning,
                stacklevel=2,
            )

        positive_weight = (1.0 - c) * g / (c * (1.0 - g))
        # Keep the negative-copy weight nonnegative when g exceeds c.
        return np.clip(positive_weight, 0.0, 1.0)
