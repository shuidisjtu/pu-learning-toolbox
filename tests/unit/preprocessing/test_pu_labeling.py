# ruff: noqa: N802, N803, N806

"""Tests for fixed-count construction and retained PNU and Gaussian utilities."""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.preprocessing import PULabelingConfig, SourcePartition, construct_pu_dataset
from pu_toolbox.preprocessing.pu_labeling import (
    make_gaussian_pu_data,
    make_pnu_labels,
)

# ═════════════════════════════════════════════════════════════════════
# OS fixed-count construction
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFixedCountConstruction:
    """Covers same-pool labeling, quota validation, and determinism."""

    @staticmethod
    def construct(y_true, n_labeled, seed=42):
        features = np.arange(len(y_true)).reshape(-1, 1)
        return construct_pu_dataset(
            SourcePartition(features, y_true, "fixture", "train"),
            PULabelingConfig("scar", "fixed_count", seed, n_labeled=n_labeled),
            positive_classes=(1,),
            negative_classes=(0,),
        ).observations.labels

    def test_basic_and_edge(self):
        y_true = np.array([1] * 50 + [0] * 50)
        y_pu = self.construct(y_true, n_labeled=10)
        assert y_pu.shape == y_true.shape
        assert np.sum(y_pu == 1) == 10
        assert np.all(y_pu[y_true == 0] == 0)
        assert np.array_equal(y_pu, self.construct(y_true, n_labeled=10))
        assert not np.array_equal(y_pu, self.construct(y_true, n_labeled=10, seed=7))

        y_true = np.array([1] * 20 + [0] * 80)
        y_pu = self.construct(y_true, n_labeled=20)
        assert np.all(y_pu[y_true == 1] == 1)
        assert np.all(y_pu[y_true == 0] == 0)

    @pytest.mark.parametrize("n_labeled", [0, -1, 1.5, True])
    def test_invalid_quota(self, n_labeled):
        with pytest.raises(ValueError, match="n_labeled"):
            self.construct(np.array([1, 0, 1]), n_labeled=n_labeled)

    @pytest.mark.parametrize("c", [0.0, -0.5, 1.5])
    def test_invalid_label_frequency(self, c):
        truth = np.array([1, 0, 1])
        with pytest.raises(ValueError, match="c must be in"):
            construct_pu_dataset(
                SourcePartition(np.ones((3, 1)), truth, "fixture", "train"),
                PULabelingConfig("scar", "fixed_count", 42, c=c),
                positive_classes=(1,),
                negative_classes=(0,),
            )


# ═════════════════════════════════════════════════════════════════════
# make_pnu_labels
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMakePnuLabels:
    """Covers: basic output, param validation, determinism."""

    def test_basic(self):
        y_true = np.array([1] * 30 + [0] * 70)
        y = make_pnu_labels(y_true, n_negatives=15, random_state=42)
        assert y.shape == y_true.shape
        assert np.sum(y == 1) == 30  # all positives
        assert np.sum(y == -1) == 15  # selected negatives
        assert np.sum(y == 0) == 55  # rest
        assert set(np.unique(y)) == {1, -1, 0}
        # Determinism
        y2 = make_pnu_labels(y_true, n_negatives=15, random_state=42)
        assert np.array_equal(y, y2)

    @pytest.mark.parametrize(
        "y_true, n_neg, match",
        [
            ([1] * 30 + [0] * 5, 10, "exceeds"),
            ([1, 0, 1, 0], 0, "n_negatives must be >= 1"),
            ([0] * 20, 5, "both 0.*and.*1"),
            ([1] * 20, 5, "both 0.*and.*1"),
        ],
    )
    def test_invalid(self, y_true, n_neg, match):
        with pytest.raises(ValueError, match=match):
            make_pnu_labels(np.array(y_true), n_negatives=n_neg)


# ═════════════════════════════════════════════════════════════════════
# make_gaussian_pu_data
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSyntheticDatasets:
    """Covers the retained Gaussian generator."""

    def test_determ_gaussian_pu_data(self):
        X, y_pu, cp = make_gaussian_pu_data(n_p=30, n_u=70, random_state=42)
        assert X.shape == (100, 5)
        assert y_pu.shape == (100,)
        assert np.sum(y_pu == 1) == 30
        assert cp == pytest.approx(0.3)

        # Determinism
        X2, y2, cp2 = make_gaussian_pu_data(n_p=30, n_u=70, random_state=42)
        assert np.allclose(X, X2)
        assert np.array_equal(y_pu, y2)

        # Custom separation produces more spread
        X3, _, _ = make_gaussian_pu_data(separation=10.0, random_state=42)
        assert X3.std() > 2.0
