"""Tests for pu_toolbox.model_selection.split."""
# ruff: noqa: N806

import numpy as np
import pytest

from pu_toolbox.core.config import MIN_POSITIVE_SAMPLES
from pu_toolbox.model_selection import PUStratifiedKFold, PUStratifiedShuffleSplit


@pytest.fixture()
def pu_data():
    rng = np.random.RandomState(42)
    n_p, n_u = 30, 120
    X = rng.randn(n_p + n_u, 5)
    y_pu = np.array([1] * n_p + [0] * n_u)
    return X, y_pu


class TestPUStratifiedKFold:
    @pytest.mark.unit
    def test_n_splits(self, pu_data):
        X, y = pu_data
        splitter = PUStratifiedKFold(n_splits=5)
        assert splitter.get_n_splits() == 5
        folds = list(splitter.split(X, y))
        assert len(folds) == 5

    @pytest.mark.unit
    def test_no_overlap_full_coverage(self, pu_data):
        X, y = pu_data
        n = len(y)
        for train_idx, test_idx in PUStratifiedKFold(n_splits=5).split(X, y):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0
            assert len(train_idx) + len(test_idx) == n

    @pytest.mark.unit
    def test_every_fold_has_positives(self, pu_data):
        X, y = pu_data
        for train_idx, _test_idx in PUStratifiedKFold(n_splits=5).split(X, y):
            n_train_p = int(np.sum(y[train_idx] == 1))
            assert n_train_p >= MIN_POSITIVE_SAMPLES

    @pytest.mark.unit
    def test_pu_ratio_approximately_preserved(self, pu_data):
        X, y = pu_data
        overall_ratio = np.mean(y == 1)
        for train_idx, _ in PUStratifiedKFold(n_splits=5).split(X, y):
            fold_ratio = np.mean(y[train_idx] == 1)
            assert abs(fold_ratio - overall_ratio) < 0.05

    @pytest.mark.unit
    def test_too_few_positives_raises(self):
        X = np.zeros((10, 2))
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        with pytest.raises(ValueError, match="positive samples"):
            list(PUStratifiedKFold(n_splits=5).split(X, y))

    @pytest.mark.unit
    def test_shuffle_reproducible(self, pu_data):
        X, y = pu_data
        s1 = list(PUStratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(X, y))
        s2 = list(PUStratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(X, y))
        for (tr1, te1), (tr2, te2) in zip(s1, s2, strict=True):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)


class TestPUStratifiedShuffleSplit:
    @pytest.mark.unit
    def test_n_splits(self, pu_data):
        X, y = pu_data
        splitter = PUStratifiedShuffleSplit(n_splits=10, random_state=42)
        assert splitter.get_n_splits() == 10
        splits = list(splitter.split(X, y))
        assert len(splits) == 10

    @pytest.mark.unit
    def test_test_size(self, pu_data):
        X, y = pu_data
        n = len(y)
        for train_idx, test_idx in PUStratifiedShuffleSplit(
            n_splits=3, test_size=0.2, random_state=42
        ).split(X, y):
            assert len(test_idx) == pytest.approx(n * 0.2, abs=2)
            assert len(train_idx) + len(test_idx) == n

    @pytest.mark.unit
    def test_every_split_has_positives(self, pu_data):
        X, y = pu_data
        for train_idx, test_idx in PUStratifiedShuffleSplit(
            n_splits=5, random_state=42
        ).split(X, y):
            assert np.sum(y[train_idx] == 1) >= MIN_POSITIVE_SAMPLES
            assert np.sum(y[test_idx] == 1) >= 1

    @pytest.mark.unit
    def test_no_overlap(self, pu_data):
        X, y = pu_data
        for train_idx, test_idx in PUStratifiedShuffleSplit(
            n_splits=3, random_state=42
        ).split(X, y):
            assert len(set(train_idx) & set(test_idx)) == 0

    @pytest.mark.unit
    def test_reproducible(self, pu_data):
        X, y = pu_data
        s1 = list(PUStratifiedShuffleSplit(n_splits=3, random_state=0).split(X, y))
        s2 = list(PUStratifiedShuffleSplit(n_splits=3, random_state=0).split(X, y))
        for (tr1, te1), (tr2, te2) in zip(s1, s2, strict=True):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)
