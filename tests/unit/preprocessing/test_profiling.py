# ruff: noqa: N802, N803, N806

"""Tests for pu_toolbox.preprocessing.profiling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from pu_toolbox.preprocessing.profiling import (
    pnu_data_summary,
    pu_data_summary,
)

# ═════════════════════════════════════════════════════════════════════
# MARK: pu_data_summary
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPuDataSummary:
    """Tests for :func:`pu_data_summary`."""

    def test_basic_counts(self, rng):
        X = rng.randn(100, 5)
        y_pu = np.array([1] * 30 + [0] * 70)
        s = pu_data_summary(X, y_pu)
        assert s["n_samples"] == 100
        assert s["n_features"] == 5
        assert s["n_positives"] == 30
        assert s["n_unlabeled"] == 70
        assert s["is_sparse"] is False

    def test_pu_ratio(self):
        X = np.zeros((100, 5))
        y_pu = np.array([1] * 20 + [0] * 80)
        s = pu_data_summary(X, y_pu)
        assert s["pu_ratio"] == 4.0  # 80 / 20

    def test_zero_positives_gives_inf_ratio(self):
        X = np.zeros((100, 5))
        y_pu = np.zeros(100, dtype=int)
        s = pu_data_summary(X, y_pu)
        assert s["pu_ratio"] == float("inf")
        assert s["n_positives"] == 0
        assert s["positive_fraction"] == 0.0

    def test_positive_fraction(self):
        X = np.zeros((100, 5))
        y_pu = np.array([1] * 30 + [0] * 70)
        s = pu_data_summary(X, y_pu)
        assert s["positive_fraction"] == 0.3

    def test_nan_detection(self, rng):
        X = rng.randn(100, 5)
        X[5, 2] = np.nan
        y_pu = np.array([1] * 30 + [0] * 70)
        s = pu_data_summary(X, y_pu)
        assert s["has_nan"] is True
        assert s["has_inf"] is False

    def test_inf_detection(self, rng):
        X = rng.randn(100, 5)
        X[10, 1] = np.inf
        y_pu = np.array([1] * 30 + [0] * 70)
        s = pu_data_summary(X, y_pu)
        assert s["has_inf"] is True
        assert s["has_nan"] is False

    def test_sparse_detection(self, rng):
        X = sparse.csr_matrix(rng.randn(100, 5))
        y_pu = np.array([1] * 30 + [0] * 70)
        s = pu_data_summary(X, y_pu)
        assert s["is_sparse"] is True
        assert s["n_features"] == 5

    def test_n_features_out_alias(self, rng):
        X = rng.randn(100, 10)
        y_pu = np.array([1] * 30 + [0] * 70)
        s = pu_data_summary(X, y_pu)
        assert s["n_features_out"] == 10

    def test_shape_mismatch_raises(self):
        X = np.zeros((100, 5))
        y_pu = np.array([1] * 30 + [0] * 50)  # only 80 labels
        with pytest.raises(ValueError, match="has 100 samples"):
            pu_data_summary(X, y_pu)

    def test_non_canonical_labels_normalized(self):
        """Labels in {1, -1} should be normalized to {+1, 0} before counting."""
        X = np.zeros((100, 5))
        y_pu = np.array([1] * 30 + [-1] * 70)
        s = pu_data_summary(X, y_pu)
        assert s["n_positives"] == 30
        assert s["n_unlabeled"] == 70

    def test_all_positive(self):
        X = np.zeros((50, 3))
        y_pu = np.ones(50, dtype=int)
        s = pu_data_summary(X, y_pu)
        assert s["n_positives"] == 50
        assert s["n_unlabeled"] == 0
        assert s["pu_ratio"] == 0.0  # 0 / 50

    def test_single_sample(self):
        X = np.zeros((1, 3))
        y_pu = np.array([1])
        s = pu_data_summary(X, y_pu)
        assert s["n_samples"] == 1
        assert s["n_features"] == 3
        assert s["n_positives"] == 1
        assert s["n_unlabeled"] == 0


# ═════════════════════════════════════════════════════════════════════
# MARK: pnu_data_summary
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPnuDataSummary:
    """Tests for :func:`pnu_data_summary`."""

    def test_basic_counts(self, rng):
        X = rng.randn(100, 5)
        y_pnu = np.array([1] * 20 + [-1] * 30 + [0] * 50)
        s = pnu_data_summary(X, y_pnu)
        assert s["n_samples"] == 100
        assert s["n_positives"] == 20
        assert s["n_negatives"] == 30
        assert s["n_unlabeled"] == 50

    def test_ratios(self, rng):
        X = rng.randn(100, 5)
        y_pnu = np.array([1] * 20 + [-1] * 30 + [0] * 50)
        s = pnu_data_summary(X, y_pnu)
        assert s["pu_ratio"] == 2.5  # 50 / 20
        assert s["nu_ratio"] == pytest.approx(50 / 30)
        assert s["pn_ratio"] == pytest.approx(20 / 30)

    def test_nan_inf_flags(self, rng):
        X = rng.randn(100, 5)
        X[0, 0] = np.nan
        X[1, 1] = np.inf
        y_pnu = np.array([1] * 20 + [-1] * 30 + [0] * 50)
        s = pnu_data_summary(X, y_pnu)
        assert s["has_nan"] is True
        assert s["has_inf"] is True

    def test_shape_mismatch_raises(self):
        X = np.zeros((100, 5))
        y_pnu = np.array([1] * 10 + [-1] * 10 + [0] * 10)  # 30 labels
        with pytest.raises(ValueError, match="has 100 samples"):
            pnu_data_summary(X, y_pnu)

    def test_sparse_input(self, rng):
        X = sparse.csr_matrix(rng.randn(100, 5))
        y_pnu = np.array([1] * 20 + [-1] * 30 + [0] * 50)
        s = pnu_data_summary(X, y_pnu)
        assert s["is_sparse"] is True
        assert s["n_features"] == 5
