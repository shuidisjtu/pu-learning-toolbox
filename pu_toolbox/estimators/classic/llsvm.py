# ruff: noqa: N803, N806

"""LLSVM: Large-Margin Label-Calibrated SVM for PU Learning.

Implements the linear LLSVM classifier from:

    Gong, C., Liu, T., Yang, J., & Tao, D.
    "Large-Margin Label-Calibrated Support Vector Machines
    for Positive and Unlabeled Learning."
    IEEE TNNLS, 30(11), 3471-3482, 2019.

Formulas follow the official MATLAB code (see method card §4.3).
"""

from __future__ import annotations

import numpy as np

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
from ...losses.llsvm import llsvm_objective


class LLSVMClassifier(BasePUClassifier):
    """Linear PU classifier with squared hinge, hat, and calibration losses.

    Uses minibatch SGD to minimise a non-convex objective combining:
    (1) squared hinge loss on labeled positives,
    (2) Gaussian-like hat loss pushing unlabeled samples away from the
        decision boundary, and
    (3) a label calibration term preventing degenerate all-positive
        predictions.

    Parameters
    ----------
    alpha : float, default 2.0
        Weight for the positive squared hinge loss.
    beta : float, default 1.0
        Weight for the unlabeled hat loss.
    gamma : float, default 10.0
        Weight for the label calibration loss.
    squash_scale : float, default 10.0
        Scaling parameter *A* in the squash function A/pi * arctan(f).
    reg_lambda : float, default 1.0
        L2 regularisation strength.
    learning_rate : float, default 5e-6
        Fixed SGD step size.
    max_epochs : int, default 3000
        Maximum number of training epochs.
    n_batches : int, default 20
        Number of mini-batches per epoch.
    fit_intercept : bool, default True
        Whether to fit an intercept term via feature augmentation.
    intercept_scale : float, default 10.0
        Value of the constant feature appended when fit_intercept is True.
    shuffle : bool, default True
        Whether to shuffle training data each epoch.
    random_state : int or None, default None
        Random seed for initialisation and shuffling.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Fitted linear weights.
    intercept_ : float
        Fitted intercept (0.0 if fit_intercept is False).
    class_prior_ : float
        Class prior used for training.
    calibration_target_ : float
        Calibration target t = 2*class_prior - 1.
    n_positive_ : int
        Number of positive samples.
    n_unlabeled_ : int
        Number of unlabeled samples.
    loss_history_ : list of float
        Objective value after each epoch.
    classes_ : np.ndarray
        Array [0, 1].
    """

    family = AlgorithmFamily.RISK_ESTIMATION
    assumption = (Assumption.SCAR, Assumption.SAR)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_EXACT
    backend = Backend.NUMPY
    maturity = Maturity.RESEARCH

    def __init__(
        self,
        *,
        alpha: float = 2.0,
        beta: float = 1.0,
        gamma: float = 10.0,
        squash_scale: float = 10.0,
        reg_lambda: float = 1.0,
        learning_rate: float = 5e-6,
        max_epochs: int = 3000,
        n_batches: int = 20,
        fit_intercept: bool = True,
        intercept_scale: float = 10.0,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.squash_scale = squash_scale
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_batches = n_batches
        self.fit_intercept = fit_intercept
        self.intercept_scale = intercept_scale
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> LLSVMClassifier:
        X, y_pu = validate_pu_X_y(
            X, y_pu, accept_sparse=False, estimator_name="LLSVMClassifier",
        )
        self._validate_hyperparams()
        self._X_shape_ = X.shape

        # Split P / U
        mask_p = y_pu == 1
        X_p = X[mask_p]
        X_u = X[~mask_p]
        self.n_positive_ = len(X_p)
        self.n_unlabeled_ = len(X_u)

        # Class prior
        if class_prior is not None:
            if not (0.0 < class_prior < 1.0):
                raise ValueError(
                    f"class_prior must be in (0, 1), got {class_prior}"
                )
            self.class_prior_ = float(class_prior)
        else:
            from ...prior.pen_l1 import ClassPriorEstimator
            estimator = ClassPriorEstimator()
            estimator.fit(X, y_pu)
            self.class_prior_ = estimator.estimate()

        self._class_prior = self.class_prior_
        self.calibration_target_ = 2.0 * self.class_prior_ - 1.0

        # Augment features for intercept
        if self.fit_intercept:
            bias_p = np.full((len(X_p), 1), self.intercept_scale)
            bias_u = np.full((len(X_u), 1), self.intercept_scale)
            X_p_aug = np.hstack([X_p, bias_p])
            X_u_aug = np.hstack([X_u, bias_u])
        else:
            X_p_aug = X_p
            X_u_aug = X_u

        # Initialise weights
        rng = np.random.RandomState(self.random_state)
        d = X_p_aug.shape[1]
        w = rng.randn(d)

        # Build training set with P/U labels for batching
        n_train = len(X_p_aug) + len(X_u_aug)
        X_train = np.vstack([X_p_aug, X_u_aug])
        is_positive = np.concatenate([
            np.ones(len(X_p_aug), dtype=bool),
            np.zeros(len(X_u_aug), dtype=bool),
        ])

        # SGD loop
        self.loss_history_ = []
        for epoch in range(self.max_epochs):
            if self.shuffle:
                perm = rng.permutation(n_train)
            else:
                if epoch == 0:
                    perm = rng.permutation(n_train)

            X_shuffled = X_train[perm]
            labels_shuffled = is_positive[perm]

            batch_size = n_train // self.n_batches
            for i in range(self.n_batches):
                if i < self.n_batches - 1:
                    idx = slice(i * batch_size, (i + 1) * batch_size)
                else:
                    idx = slice(i * batch_size, None)

                X_batch = X_shuffled[idx]
                labels_batch = labels_shuffled[idx]
                X_p_batch = X_batch[labels_batch]
                X_u_batch = X_batch[~labels_batch]

                _, grad = llsvm_objective(
                    w, X_p_batch, X_u_batch,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                    t=self.calibration_target_,
                    A=self.squash_scale,
                    reg_lambda=self.reg_lambda,
                )
                w = w - self.learning_rate * grad

            # Record full-data loss
            loss, _ = llsvm_objective(
                w, X_p_aug, X_u_aug,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                t=self.calibration_target_,
                A=self.squash_scale,
                reg_lambda=self.reg_lambda,
            )
            self.loss_history_.append(float(loss))

        # Extract coef and intercept
        if self.fit_intercept:
            self.coef_ = w[:-1].copy()
            self.intercept_ = float(w[-1] * self.intercept_scale)
        else:
            self.coef_ = w.copy()
            self.intercept_ = 0.0

        self.classes_ = np.array([0, 1])
        self._is_fitted = True
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def _predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def get_pu_metadata(self) -> dict:
        meta = super().get_pu_metadata()
        if self._is_fitted:
            meta.update({
                "class_prior": self.class_prior_,
                "calibration_target": self.calibration_target_,
                "n_positive": self.n_positive_,
                "n_unlabeled": self.n_unlabeled_,
                "n_epochs": len(self.loss_history_),
            })
        return meta

    def _validate_hyperparams(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}")
        if self.squash_scale <= 0:
            raise ValueError(f"squash_scale must be > 0, got {self.squash_scale}")
        if self.reg_lambda <= 0:
            raise ValueError(f"reg_lambda must be > 0, got {self.reg_lambda}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.n_batches < 1:
            raise ValueError(f"n_batches must be >= 1, got {self.n_batches}")
        if self.intercept_scale <= 0:
            raise ValueError(
                f"intercept_scale must be > 0, got {self.intercept_scale}"
            )
