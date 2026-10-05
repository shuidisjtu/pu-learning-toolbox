# ruff: noqa: N803, N806
"""Native PU-specialized Rank Pruning, Northcutt et al., UAI 2017.

Clean-room implementation of confident counts, rank pruning and reweighting.
Only the clean-positive PU setting (rho0=0), not general two-sided noisy labels.
"""

from __future__ import annotations

import inspect

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array

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


def confident_positive_noise_rate(labels, probabilities):
    """Estimate rho1=P(s=0|Y=1) from held-out confident positive counts."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if probabilities.shape != labels.shape or labels.ndim != 1:
        raise ValueError("confident counts require matching one-dimensional inputs")
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(probabilities).all():
        raise ValueError("confident counts require both binary labels and finite probabilities")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    threshold = probabilities[labels == 1].mean()
    confident = probabilities >= threshold
    return min(float(np.sum(confident & (labels == 0)) / confident.sum()), 0.9999)


def rank_pruning_pu_mask(labels, probabilities, rho1, *, min_retained=10):
    """Author-aligned strict rank threshold (ties kept), P never removed."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    confident_positive_noise_rate(labels, probabilities)  # input validation
    if not np.isfinite(rho1) or not 0 <= rho1 < 1:
        raise ValueError("rho1 must be in [0, 1)")
    if type(min_retained) is not int or min_retained < 1:
        raise ValueError("min_retained must be a positive integer")
    ps1 = np.mean(labels)
    estimated_prior = ps1 / (1 - rho1)
    pi0 = float(np.clip(rho1 * estimated_prior / (1 - ps1), 0, 0.9999))
    unlabeled = labels == 0
    n_unlabeled = int(unlabeled.sum())
    remove_count = max(0, min(int(n_unlabeled * pi0), n_unlabeled - min_retained))
    mask = np.zeros(len(labels), dtype=bool)
    if remove_count > 0:
        cutoff = -np.partition(-probabilities[unlabeled], remove_count)[remove_count]
        mask = unlabeled & (probabilities > cutoff)
    return mask, float(estimated_prior), pi0


class RankPruningClassifier(BasePUClassifier):
    """Cross-fit observed PU labels, prune uncertain U, reweight and refit."""

    family = AlgorithmFamily.CLASSIC_CALIBRATION
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.SINGLE_TRAINING_SET,)
    requires_class_prior = False
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_RELATED
    backend = Backend.SKLEARN
    maturity = Maturity.EXPERIMENTAL
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED
    native_architectures = frozenset()
    input_ndims = frozenset({2})
    encoder_parameter = None
    trains_encoder = False

    def __init__(
        self,
        *,
        base_estimator=None,
        n_cv_folds=3,
        frac_pos2neg=None,
        min_retained=10,
        random_state=0,
    ):
        super().__init__()
        self.base_estimator = base_estimator
        self.n_cv_folds = n_cv_folds
        self.frac_pos2neg = frac_pos2neg
        self.min_retained = min_retained
        self.random_state = random_state

    @staticmethod
    def _positive_proba(estimator, X):
        probabilities = np.asarray(estimator.predict_proba(X))
        classes = np.asarray(estimator.classes_)
        if probabilities.shape != (len(X), 2) or set(classes) != {0, 1}:
            raise ValueError("base_estimator must produce binary probabilities and classes_")
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError("base_estimator returned invalid probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError("base_estimator probabilities must sum to one")
        return probabilities[:, int(np.flatnonzero(classes == 1)[0])]

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None, os_or_ts="os"):
        self._is_fitted = False
        if sample_weight is not None:
            raise NotImplementedError("Rank Pruning does not implement external sample_weight")
        if os_or_ts != "os":
            raise ValueError("Rank Pruning is native OS; ts risk substitution is not applicable")
        X, y_pu = validate_pu_X_y(
            X, y_pu, accept_sparse=False, estimator_name="RankPruningClassifier"
        )
        X = check_array(X, dtype=float)
        if type(self.n_cv_folds) is not int or self.n_cv_folds < 2:
            raise ValueError("n_cv_folds must be an integer >= 2")
        if np.min(np.bincount(y_pu, minlength=2)) < self.n_cv_folds:
            raise ValueError("each PU label group must contain at least n_cv_folds rows")
        if type(self.min_retained) is not int or self.min_retained < 1:
            raise ValueError("min_retained must be a positive integer")
        if self.frac_pos2neg is not None and (
            not np.isfinite(self.frac_pos2neg) or not 0 <= self.frac_pos2neg < 1
        ):
            raise ValueError("frac_pos2neg must be in [0, 1)")
        if class_prior is not None and (not np.isfinite(class_prior) or not 0 < class_prior < 1):
            raise ValueError("class_prior, if supplied, must be in (0, 1)")
        base = (
            self.base_estimator
            if self.base_estimator is not None
            else LogisticRegression(max_iter=1000, random_state=self.random_state)
        )
        if not callable(getattr(base, "predict_proba", None)):
            raise TypeError("base_estimator must implement predict_proba")
        if "sample_weight" not in inspect.signature(base.fit).parameters:
            raise TypeError("base_estimator.fit must accept sample_weight for RP reweighting")
        self.oof_probabilities_ = np.empty(len(y_pu), dtype=float)
        self.fold_noise_rates_, self.cv_fold_indices_ = [], []
        splitter = StratifiedKFold(
            n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_state
        )
        for train, heldout in splitter.split(X, y_pu):
            estimator = clone(base).fit(X[train], y_pu[train])
            probabilities = self._positive_proba(estimator, X[heldout])
            self.oof_probabilities_[heldout] = probabilities
            self.fold_noise_rates_.append(
                confident_positive_noise_rate(y_pu[heldout], probabilities)
            )
            self.cv_fold_indices_.append({"train": train.copy(), "heldout": heldout.copy()})
        self.rho1_ = (
            float(np.mean(self.fold_noise_rates_))
            if self.frac_pos2neg is None
            else float(self.frac_pos2neg)
        )
        self.rho0_ = 0.0  # PU clean-positive specialization, not estimated from noisy P.
        self.pruned_mask_, self.estimated_prior_, self.pi0_ = rank_pruning_pu_mask(
            y_pu, self.oof_probabilities_, self.rho1_, min_retained=self.min_retained
        )
        self.prior_diagnostic_ = (
            "valid_noise_model_estimate"
            if 0 < self.estimated_prior_ < 1
            else "incompatible_noise_model_estimate_not_a_valid_prior"
        )
        retained = ~self.pruned_mask_
        self.retained_indices_ = np.flatnonzero(retained)
        self.final_sample_weights_ = np.where(y_pu[retained] == 1, 1 / (1 - self.rho1_), 1.0)
        self.estimator_ = clone(base).fit(
            X[retained], y_pu[retained], sample_weight=self.final_sample_weights_
        )
        self.classes_ = np.array([0, 1])
        self.n_features_in_, self._X_shape_ = X.shape[1], X.shape
        self._class_prior = None  # External pi is not used by the noise-estimation pipeline.
        self.training_view_, self.calibration_applied_ = "os", False
        self._is_fitted = True
        return self

    def _decision_function(self, X):
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Rank Pruning feature count differs from training")
        return self._positive_proba(self.estimator_, X) - 0.5

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        self._check_is_fitted()
        positive = self._decision_function(X) + 0.5
        return np.column_stack((1 - positive, positive))
