# ruff: noqa: N803, N806

"""Elkan–Noto PU classifier (KDD 2008).

Native clean-room implementation of the classic SCAR-based PU learning
method.  Reference: pulearn/pulearn (BSD-3-Clause) for algorithmic
validation; API contract follows ``BasePUClassifier``.

Reference
---------
Elkan, C. & Noto, K.  "Learning Classifiers from Only Positive and
Unlabeled Data."  KDD, 2008.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted as _sklearn_check_is_fitted

from ...core.base import BasePUClassifier
from ...core.config import CLASS_PRIOR_CLIP_EPS, PROPENSITY_CLIP_EPS
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


class ElkanNotoClassifier(BasePUClassifier):
    """Elkan–Noto PU classifier with probability correction or weighted retraining.

    Under the SCAR assumption, a positive example is labeled with constant
    probability ``c = P(s=1 | y=1)``.  This classifier first trains a
    probabilistic model ``g(x) ≈ P(s=1 | x)``, then estimates ``c`` via
    stratified K-fold out-of-fold predictions on the labeled positives.
    The traditional positive-class probability is recovered as
    ``f(x) = g(x) / c``.

    Two modes are available:

    * ``"probability_correction"`` (default) — return ``f(x)`` directly.
    * ``"weighted_retraining"`` — replicate each unlabeled sample as
      weighted positive + negative copies and refit a fresh estimator.

    Parameters
    ----------
    base_estimator : sklearn estimator, optional
        Base binary classifier.  Must implement ``predict_proba``.
        Default: ``LogisticRegression()``.
    calibration_method : {"sigmoid", "isotonic"}, default "sigmoid"
        Probability calibration method used when the base estimator is
        not a linear model with natural probabilistic output.
    n_cv_folds : int, default 3
        Number of stratified cross-validation folds for estimating
        ``c`` via out-of-fold predictions.  Must be ≥ 2.
    eps : float, default 1e-12
        Numerical clipping threshold for ``g(x)`` to avoid division
        by zero in weight computation.
    mode : {"probability_correction", "weighted_retraining"}, \
            default "probability_correction"
        * ``"probability_correction"`` — scale ``g(x)`` by ``1 / c``.
        * ``"weighted_retraining"`` — duplicate unlabeled samples with
          computed weights and refit the base estimator.
    random_state : int or None, default None
        Random seed for reproducible K-fold splits.

    Attributes
    ----------
    propensity_ : float
        Estimated labeling propensity ``c_hat = P(s=1 | y=1)``.
    class_prior_ : float
        Estimated class prior ``P(y=1)`` (only valid for
        single-training-set scenario).
    label_classifier_ : sklearn estimator
        The fitted ``g(x)`` model that predicts ``P(s=1 | x)``.
    final_estimator_ : sklearn estimator or None
        The retrained model used in weighted-retraining mode.
        ``None`` in probability-correction mode.
    """

    # ── Class-level metadata ────────────────────────────────────────
    family: AlgorithmFamily = AlgorithmFamily.CLASSIC_CALIBRATION
    assumption: tuple[Assumption, ...] = (Assumption.SCAR,)
    scenario: tuple[Scenario, ...] = (Scenario.SINGLE_TRAINING_SET,)
    requires_class_prior: bool = False
    implementation_status: ImplementationStatus = ImplementationStatus.NATIVE
    source_status: SourceStatus = SourceStatus.THIRD_PARTY_ONLY
    backend: Backend = Backend.SKLEARN
    maturity: Maturity = Maturity.STABLE

    def __init__(
        self,
        base_estimator=None,
        calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
        n_cv_folds: int = 3,
        eps: float = 1e-12,
        mode: Literal["probability_correction", "weighted_retraining"] = "probability_correction",
        random_state=None,
    ) -> None:
        super().__init__()
        self.base_estimator = base_estimator
        self.calibration_method = calibration_method
        self.eps = eps
        self.mode = mode
        self.random_state = random_state

        # ── Parameter validation ──────────────────────────────────
        if n_cv_folds < 2:
            raise ValueError(
                f"n_cv_folds must be >= 2 for stratified K-fold; got {n_cv_folds}."
            )
        self.n_cv_folds = n_cv_folds

        if eps <= 0:
            raise ValueError(f"eps must be > 0; got {eps}.")

    # ── Public fit API ──────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> ElkanNotoClassifier:
        """Fit the Elkan–Noto classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y_pu : np.ndarray of shape (n_samples,)
            PU labels: 1 = labeled positive, 0 = unlabeled.
        class_prior : float, optional
            Accepted for API compatibility; ignored because Elkan–Noto
            estimates ``c`` internally.
        sample_weight : np.ndarray of shape (n_samples,), optional
            Per-sample weights forwarded to the base estimator.

        Returns
        -------
        self : ElkanNotoClassifier
            Fitted estimator.
        """
        X, y_pu = validate_pu_X_y(X, y_pu, estimator_name="ElkanNotoClassifier")

        n_samples, n_features = X.shape
        self._X_shape_ = (n_samples, n_features)
        self.classes_ = np.array([0, 1])

        # Validate and canonicalise sample_weight early
        sw: np.ndarray | None = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            if sw.shape != (n_samples,):
                raise ValueError(
                    f"sample_weight must have shape ({n_samples},); got {sw.shape}."
                )

        mask_pos = y_pu == 1
        mask_unl = y_pu == 0
        n_labeled = int(np.sum(mask_pos))

        if n_labeled < self.n_cv_folds:
            raise ValueError(
                f"Need at least n_cv_folds={self.n_cv_folds} labeled positives "
                f"for stratified K-fold, but only {n_labeled} were found. "
                "Reduce n_cv_folds or collect more labeled data."
            )

        # ── Build calibrated g(x) classifier ────────────────────────
        base = self._resolve_base_estimator()
        self.label_classifier_ = self._wrap_with_calibration(base)

        # ── Estimate c via stratified K-fold out-of-fold ────────────
        self.propensity_ = self._estimate_propensity(X, y_pu, mask_pos, sw)

        # ── Fit final g(x) on all data ──────────────────────────────
        # K-fold above used temporary models for c estimation only;
        # now train the actual label classifier on the full dataset.
        s_labels = y_pu.astype(int)
        if sw is not None:
            self.label_classifier_.fit(X, s_labels, sample_weight=sw)
        else:
            self.label_classifier_.fit(X, s_labels)

        # ── Class prior ─────────────────────────────────────────────
        self.class_prior_ = float(
            np.clip(
                n_labeled / (n_samples * self.propensity_),
                CLASS_PRIOR_CLIP_EPS,
                1.0 - CLASS_PRIOR_CLIP_EPS,
            )
        )

        # ── Weighted retraining mode ─────────────────────────────────
        if self.mode == "weighted_retraining":
            self.final_estimator_ = self._retrain_weighted(X, mask_pos, mask_unl)
        else:
            self.final_estimator_ = None

        self._is_fitted = True
        return self

    # ── Core prediction methods ─────────────────────────────────────

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions via threshold on ``f(x) >= 0.5``."""
        self._check_is_fitted()
        f = self._decision_function(X)
        return (f >= 0.5).astype(int)

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return ``f(x) = P(y=1 | x)`` estimates."""
        self._check_is_fitted()

        if self.mode == "weighted_retraining" and self.final_estimator_ is not None:
            proba = self.final_estimator_.predict_proba(X)
            return proba[:, 1]

        g = self._predict_label_proba(X)
        return g / self.propensity_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``P(y=0|x)`` and ``P(y=1|x)``.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Column 0 = ``P(y=0|x)``, column 1 = ``P(y=1|x)``.
            Values may exceed 1 when ``c < 1`` — this is expected
            behaviour under the Elkan–Noto model (see method card §6.2).
        """
        self._check_is_fitted()
        f = self._decision_function(X)
        return np.column_stack([1.0 - f, f])

    def predict_label_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return ``g(x) = P(s=1 | x)``.

        Returns ``None`` in weighted-retraining mode (the final
        estimator's probabilities represent ``f(x)``, not ``g(x)``).
        """
        self._check_is_fitted()
        if self.mode == "weighted_retraining":
            return None
        return self._predict_label_proba(X)

    # ── Metadata ────────────────────────────────────────────────────

    def get_pu_metadata(self) -> dict:
        """Return PU-specific metadata including propensity and mode."""
        meta = super().get_pu_metadata()
        meta["propensity"] = getattr(self, "propensity_", None)
        meta["mode"] = self.mode
        return meta

    # ═════════════════════════════════════════════════════════════════
    # Private helpers
    # ═════════════════════════════════════════════════════════════════

    def _resolve_base_estimator(self):
        """Return the base estimator, defaulting to LogisticRegression."""
        if self.base_estimator is None:
            return LogisticRegression(random_state=self.random_state)
        return clone(self.base_estimator)

    def _wrap_with_calibration(self, base):
        """Wrap the base estimator with calibration when needed.

        LogisticRegression (and other linear models with natural
        probabilistic output) are used directly.  Other models are
        wrapped in ``CalibratedClassifierCV`` to ensure reliable
        ``g(x) ≈ P(s=1 | x)`` estimates.
        """
        # LogisticRegression already outputs well-calibrated probabilities
        # when using the default lbfgs solver.
        if isinstance(base, LogisticRegression):
            return base

        return CalibratedClassifierCV(
            estimator=base,
            method=self.calibration_method,
            cv=min(self.n_cv_folds, 5),
        )

    # NOTE: CalibratedClassifierCV with integer cv uses StratifiedKFold
    # internally, which defaults to shuffle=False — random_state is not
    # needed for deterministic splits in this configuration.  Sub-estimator
    # randomness is controlled by the base estimator's own random_state.

    def _estimate_propensity(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        mask_pos: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> float:
        """Estimate ``c = P(s=1 | y=1)`` via stratified K-fold out-of-fold.

        Splits labeled positives into ``n_cv_folds`` folds, trains
        ``g(x)`` on the complement of each fold, and averages the
        predicted probabilities on the held-out positives.

        Per the method card, only ``c_hat_1`` (mean over positive
        hold-out predictions) is exposed.
        """
        X_pos = X[mask_pos]
        n_pos = X_pos.shape[0]

        # Use dummy labels for StratifiedKFold (all 1 — only positives)
        dummy_y = np.ones(n_pos, dtype=int)

        skf = StratifiedKFold(
            n_splits=self.n_cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        g_scores = np.empty(n_pos)

        # Pre-compute unlabeled portion (invariant across folds)
        X_unl = X[~mask_pos]
        n_unl = X_unl.shape[0]
        base = self._resolve_base_estimator()

        # Extract sample_weight arrays for unlabeled and positive subsets
        sw = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            sw_unl = sw[~mask_pos]
            sw_pos = sw[mask_pos]

        for train_idx, holdout_idx in skf.split(X_pos, dummy_y):
            X_pos_train = X_pos[train_idx]
            X_pos_holdout = X_pos[holdout_idx]

            X_train = np.vstack([X_unl, X_pos_train])
            s_train = np.hstack([np.zeros(n_unl, dtype=int), np.ones(len(train_idx), dtype=int)])

            # Forward sample_weight when provided
            fit_kwargs = {}
            if sw is not None:
                sw_pos_train = sw_pos[train_idx]
                fit_kwargs["sample_weight"] = np.hstack([sw_unl, sw_pos_train])

            g_model = clone(base)
            g_model = self._wrap_with_calibration(g_model)
            g_model.fit(X_train, s_train, **fit_kwargs)

            proba = g_model.predict_proba(X_pos_holdout)
            g_scores[holdout_idx] = proba[:, 1]

        c_hat = float(np.mean(g_scores))
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
        """Return ``g(x) = P(s=1 | x)`` from the fitted label classifier."""
        _sklearn_check_is_fitted(self.label_classifier_)
        proba = self.label_classifier_.predict_proba(X)
        return proba[:, 1]

    def _retrain_weighted(
        self,
        X: np.ndarray,
        mask_pos: np.ndarray,
        mask_unl: np.ndarray,
    ):
        """Create a weighted dataset and refit a fresh estimator.

        For each unlabeled sample, two weighted copies are created
        (positive with weight ``w(x)``, negative with weight ``1-w(x)``).
        Labeled positives are kept with weight 1.

        Method card §5.2, §6.4: the new estimator is a **fresh clone**
        and replaces ``g(x)`` as the final model.
        """
        g = self._predict_label_proba(X)
        w = self._compute_weights(g[mask_unl])

        X_pos = X[mask_pos]
        X_unl = X[mask_unl]

        # Each unlabeled sample → positive copy + negative copy
        X_aug = np.vstack([X_pos, X_unl, X_unl])
        y_aug = np.hstack(
            [
                np.ones(len(X_pos), dtype=int),
                np.ones(len(X_unl), dtype=int),
                np.zeros(len(X_unl), dtype=int),
            ]
        )
        sw_aug = np.hstack(
            [
                np.ones(len(X_pos)),
                w,
                1.0 - w,
            ]
        )

        final = clone(self._resolve_base_estimator())
        # Ensure the retrained model can produce probabilities
        if not hasattr(final, "predict_proba"):
            final = self._wrap_with_calibration(final)
        final.fit(X_aug, y_aug, sample_weight=sw_aug)
        return final

    def _compute_weights(self, g_unlabeled: np.ndarray) -> np.ndarray:
        """Compute ``w(x)`` for unlabeled samples.

        ``w(x) = (1 - c) * g(x) / (c * (1 - g(x)))``

        Numerators and denominators are clipped to ``eps`` to avoid
        division-by-zero and log(0) downstream.  A warning is emitted
        when clipping actually occurs.
        """
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

        w = (1.0 - c) * g / (c * (1.0 - g))
        return w
