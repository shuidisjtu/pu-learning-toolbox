"""PU-aware cross-validation splitters.

Ensures every fold contains enough labeled positive samples for meaningful
PU training.  Follows the sklearn splitter API: ``get_n_splits()``,
``split(X, y)`` yielding ``(train_idx, test_idx)`` pairs.
"""
# ruff: noqa: N803

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from pu_toolbox.core.config import MIN_POSITIVE_SAMPLES, POSITIVE_LABEL
from pu_toolbox.core.labels import normalize_pu_labels

__all__ = ["PUStratifiedKFold", "PUStratifiedShuffleSplit"]


class PUStratifiedKFold:
    """K-Fold cross-validation preserving the P/U ratio in each fold.

    Internally performs stratified splitting on the PU labels so that
    every training fold contains a sufficient number of labeled positives.

    Parameters
    ----------
    n_splits : int, default=5
    shuffle : bool, default=False
    random_state : int or None, default=None
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y, groups=None):
        """Yield (train_idx, test_idx) for each fold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            PU labels in {+1, 0} (or any format accepted by normalize_pu_labels).
        """
        y = normalize_pu_labels(np.asarray(y))
        n_positive = int(np.sum(y == POSITIVE_LABEL))
        if n_positive < self.n_splits:
            raise ValueError(
                f"Cannot split {n_positive} positive samples into "
                f"{self.n_splits} folds. Reduce n_splits or add more positives."
            )

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        yield from skf.split(X, y)


class PUStratifiedShuffleSplit:
    """Random train/test split preserving the P/U ratio.

    Each split draws a stratified test set so that the P/U ratio in both
    train and test approximately matches the original data.

    Parameters
    ----------
    n_splits : int, default=10
    test_size : float, default=0.2
    random_state : int or None, default=None
    """

    def __init__(
        self,
        n_splits: int = 10,
        *,
        test_size: float = 0.2,
        random_state: int | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y, groups=None):
        """Yield (train_idx, test_idx) for each split.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            PU labels in {+1, 0}.
        """
        y = normalize_pu_labels(np.asarray(y))
        n_positive = int(np.sum(y == POSITIVE_LABEL))
        n_test_p = max(1, int(round(n_positive * self.test_size)))
        n_train_p = n_positive - n_test_p
        if n_train_p < MIN_POSITIVE_SAMPLES:
            raise ValueError(
                f"test_size={self.test_size} leaves only {n_train_p} "
                f"positive samples in training (need >= {MIN_POSITIVE_SAMPLES})."
            )

        sss = StratifiedShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        yield from sss.split(X, y)
