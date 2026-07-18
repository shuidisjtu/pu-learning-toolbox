# ruff: noqa: N802, N803, N806

"""Tests for pu_toolbox.preprocessing.pu_labeling."""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.preprocessing.pu_labeling import (
    make_case_control_labels,
    make_gaussian_pu_data,
    make_pnu_labels,
    make_pu_labels,
    make_scar_dataset,
    make_scar_labels,
)

# ═════════════════════════════════════════════════════════════════════
# MARK: make_scar_labels
# ═════════════════════════════════════════════════════════════════════


class TestMakeScarLabels:
    """Tests for :func:`make_scar_labels`."""

    def test_basic_output(self, rng):
        y_true = np.array([1] * 50 + [0] * 50)
        y_pu = make_scar_labels(y_true, c=0.5, random_state=42)
        assert y_pu.shape == y_true.shape
        assert y_pu.dtype == int
        assert set(np.unique(y_pu)).issubset({1, 0})

    def test_c_one_labels_all_positives(self, rng):
        y_true = np.array([1] * 30 + [0] * 70)
        y_pu = make_scar_labels(y_true, c=1.0, random_state=42)
        pos_mask = y_true == 1
        # All positives should be labeled with c=1
        assert np.all(y_pu[pos_mask] == 1)
        # All negatives should be unlabeled
        assert np.all(y_pu[~pos_mask] == 0)

    def test_c_zero_raises(self):
        y_true = np.array([1, 0, 1, 0])
        with pytest.raises(ValueError, match="c must be in"):
            make_scar_labels(y_true, c=0.0)

    def test_c_negative_raises(self):
        y_true = np.array([1, 0])
        with pytest.raises(ValueError, match="c must be in"):
            make_scar_labels(y_true, c=-0.5)

    def test_c_above_one_raises(self):
        y_true = np.array([1, 0])
        with pytest.raises(ValueError, match="c must be in"):
            make_scar_labels(y_true, c=1.5)

    def test_no_true_positives(self):
        y_true = np.array([0] * 50)
        y_pu = make_scar_labels(y_true, c=0.5, random_state=42)
        assert np.all(y_pu == 0)

    def test_no_true_negatives(self, rng):
        y_true = np.ones(20, dtype=int)
        y_pu = make_scar_labels(y_true, c=0.5, random_state=42)
        # Some labeled, some unlabeled — all should be in {1, 0}
        assert set(np.unique(y_pu)).issubset({1, 0})

    def test_deterministic(self):
        y_true = np.array([1] * 50 + [0] * 50)
        y1 = make_scar_labels(y_true, c=0.5, random_state=42)
        y2 = make_scar_labels(y_true, c=0.5, random_state=42)
        assert np.array_equal(y1, y2)

    def test_output_length_preserved(self, rng):
        y_true = np.array([1] * 30 + [0] * 20)
        y_pu = make_scar_labels(y_true, c=0.5, random_state=42)
        assert len(y_pu) == len(y_true)

    def test_invalid_y_true_2d_raises(self):
        y_true = np.array([[1, 0], [1, 0]])
        with pytest.raises(ValueError, match="1-D"):
            make_scar_labels(y_true, c=0.5)

    def test_invalid_y_true_values_raises(self):
        y_true = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="0, 1"):
            make_scar_labels(y_true, c=0.5)

    def test_random_state_none_uses_global(self, rng):
        y_true = np.array([1] * 10 + [0] * 10)
        y_pu = make_scar_labels(y_true, c=0.5)
        assert set(np.unique(y_pu)).issubset({1, 0})


# ═════════════════════════════════════════════════════════════════════
# MARK: make_case_control_labels
# ═════════════════════════════════════════════════════════════════════


class TestMakeCaseControlLabels:
    """Tests for :func:`make_case_control_labels`."""

    def test_basic_output(self, rng):
        y_true = np.array([1] * 50 + [0] * 50)
        y_pu = make_case_control_labels(y_true, n_labeled=10, random_state=42)
        assert y_pu.shape == y_true.shape
        assert np.sum(y_pu == 1) == 10
        assert set(np.unique(y_pu)).issubset({1, 0})

    def test_n_labeled_exceeds_positives_raises(self):
        y_true = np.array([1] * 10 + [0] * 90)
        with pytest.raises(ValueError, match="exceeds"):
            make_case_control_labels(y_true, n_labeled=20)

    def test_n_labeled_negative_raises(self):
        y_true = np.array([1, 0, 1])
        with pytest.raises(ValueError, match="n_labeled must be >= 1"):
            make_case_control_labels(y_true, n_labeled=0)

    def test_deterministic(self):
        y_true = np.array([1] * 50 + [0] * 50)
        y1 = make_case_control_labels(y_true, n_labeled=5, random_state=42)
        y2 = make_case_control_labels(y_true, n_labeled=5, random_state=42)
        assert np.array_equal(y1, y2)

    def test_negatives_always_unlabeled(self, rng):
        y_true = np.array([1] * 10 + [0] * 90)
        y_pu = make_case_control_labels(y_true, n_labeled=5, random_state=42)
        neg_mask = y_true == 0
        assert np.all(y_pu[neg_mask] == 0)

    def test_no_true_positives_raises(self):
        y_true = np.array([0] * 10)
        with pytest.raises(ValueError, match="both 0"):
            make_case_control_labels(y_true, n_labeled=5)

    def test_no_true_negatives_raises(self):
        y_true = np.array([1] * 10)
        with pytest.raises(ValueError, match="both 0"):
            make_case_control_labels(y_true, n_labeled=5)

    def test_all_positives_labeled(self, rng):
        y_true = np.array([1] * 20 + [0] * 80)
        y_pu = make_case_control_labels(y_true, n_labeled=20, random_state=42)
        assert np.sum(y_pu == 1) == 20


# ═════════════════════════════════════════════════════════════════════
# MARK: make_pu_labels (dispatcher)
# ═════════════════════════════════════════════════════════════════════


class TestMakePuLabels:
    """Tests for :func:`make_pu_labels` dispatcher."""

    def test_scar_delegation(self, rng):
        y_true = np.array([1] * 50 + [0] * 50)
        y1 = make_pu_labels(y_true, mechanism="scar", c=0.5, random_state=42)
        y2 = make_scar_labels(y_true, c=0.5, random_state=42)
        assert np.array_equal(y1, y2)

    def test_case_control_delegation(self, rng):
        y_true = np.array([1] * 50 + [0] * 50)
        y1 = make_pu_labels(
            y_true,
            mechanism="case_control",
            n_labeled=10,
            random_state=42,
        )
        y2 = make_case_control_labels(y_true, n_labeled=10, random_state=42)
        assert np.array_equal(y1, y2)

    def test_invalid_mechanism_raises(self):
        y_true = np.array([1, 0])
        with pytest.raises(ValueError, match="Unknown mechanism"):
            make_pu_labels(y_true, mechanism="invalid", c=0.5)

    def test_scar_missing_c_raises(self):
        y_true = np.array([1, 0])
        with pytest.raises(ValueError, match="requires the 'c' parameter"):
            make_pu_labels(y_true, mechanism="scar")

    def test_case_control_missing_n_labeled_raises(self):
        y_true = np.array([1, 0])
        with pytest.raises(ValueError, match="requires the 'n_labeled' parameter"):
            make_pu_labels(y_true, mechanism="case_control")


# ═════════════════════════════════════════════════════════════════════
# MARK: make_pnu_labels
# ═════════════════════════════════════════════════════════════════════


class TestMakePnuLabels:
    """Tests for :func:`make_pnu_labels`."""

    def test_basic_output(self, rng):
        y_true = np.array([1] * 30 + [0] * 70)
        y_pnu = make_pnu_labels(y_true, n_negatives=15, random_state=42)
        assert y_pnu.shape == y_true.shape
        assert np.sum(y_pnu == 1) == 30  # all positives
        assert np.sum(y_pnu == -1) == 15  # selected negatives
        unlabeled = np.sum(y_pnu == 0)
        assert unlabeled == 55  # 70 - 15
        assert set(np.unique(y_pnu)) == {1, -1, 0}

    def test_n_negatives_exceeds_available_raises(self):
        y_true = np.array([1] * 30 + [0] * 5)
        with pytest.raises(ValueError, match="exceeds"):
            make_pnu_labels(y_true, n_negatives=10)

    def test_n_negatives_zero_raises(self):
        y_true = np.array([1, 0, 1, 0])
        with pytest.raises(ValueError, match="n_negatives must be >= 1"):
            make_pnu_labels(y_true, n_negatives=0)

    def test_deterministic(self):
        y_true = np.array([1] * 30 + [0] * 70)
        y1 = make_pnu_labels(y_true, n_negatives=10, random_state=42)
        y2 = make_pnu_labels(y_true, n_negatives=10, random_state=42)
        assert np.array_equal(y1, y2)

    def test_no_true_positives_raises(self):
        y_true = np.array([0] * 20)
        with pytest.raises(ValueError, match="both 0"):
            make_pnu_labels(y_true, n_negatives=5)

    def test_no_true_negatives_raises(self):
        y_true = np.array([1] * 20)
        with pytest.raises(ValueError, match="both 0"):
            make_pnu_labels(y_true, n_negatives=5)


# ═════════════════════════════════════════════════════════════════════
# MARK: make_gaussian_pu_data
# ═════════════════════════════════════════════════════════════════════


class TestMakeGaussianPuData:
    """Tests for :func:`make_gaussian_pu_data`."""

    def test_basic_output(self, rng):
        X, y_pu, cp = make_gaussian_pu_data(random_state=42)
        assert X.shape == (150, 5)
        assert y_pu.shape == (150,)
        assert np.sum(y_pu == 1) == 50
        assert np.sum(y_pu == 0) == 100
        assert 0.0 < cp < 1.0

    def test_class_prior_correct(self):
        X, y_pu, cp = make_gaussian_pu_data(n_p=30, n_u=70, random_state=42)
        assert cp == pytest.approx(30 / 100)

    def test_deterministic(self):
        X1, y1, cp1 = make_gaussian_pu_data(random_state=42)
        X2, y2, cp2 = make_gaussian_pu_data(random_state=42)
        assert np.allclose(X1, X2)
        assert np.array_equal(y1, y2)
        assert cp1 == cp2

    def test_custom_separation(self, rng):
        X, y_pu, cp = make_gaussian_pu_data(separation=10.0, random_state=42)
        # Classes should be well-separated
        pos = X[y_pu == 1]
        unl = X[y_pu == 0]
        pos_mean = pos.mean(axis=0)
        unl_mean = unl.mean(axis=0)
        diff = np.linalg.norm(pos_mean - unl_mean)
        # With separation=10.0, should be reasonably far apart
        assert diff > 2.0

    def test_random_state_none_works(self):
        X, y_pu, cp = make_gaussian_pu_data()
        assert X.shape == (150, 5)


# ═════════════════════════════════════════════════════════════════════
# MARK: make_scar_dataset
# ═════════════════════════════════════════════════════════════════════


class TestMakeScarDataset:
    """Tests for :func:`make_scar_dataset`."""

    def test_basic_output(self, rng):
        X, y_pu, y_true = make_scar_dataset(random_state=42)
        assert X.shape == (200, 5)
        assert y_pu.shape == (200,)
        assert y_true.shape == (200,)
        assert set(np.unique(y_pu)).issubset({1, 0})
        assert set(np.unique(y_true)) == {0, 1}

    def test_y_true_consistency(self, rng):
        """Every labeled positive in y_pu must be a true positive in y_true."""
        X, y_pu, y_true = make_scar_dataset(random_state=42)
        pos_in_pu = y_pu == 1
        assert np.all(y_true[pos_in_pu] == 1)

    def test_balanced_true_labels(self, rng):
        X, y_pu, y_true = make_scar_dataset(n=100, random_state=42)
        assert np.sum(y_true == 1) == 100
        assert np.sum(y_true == 0) == 100

    def test_deterministic(self):
        X1, y1, t1 = make_scar_dataset(random_state=42)
        X2, y2, t2 = make_scar_dataset(random_state=42)
        assert np.allclose(X1, X2)
        assert np.array_equal(y1, y2)
        assert np.array_equal(t1, t2)

    def test_custom_separation(self, rng):
        X, y_pu, y_true = make_scar_dataset(separation=10.0, random_state=42)
        # With strong separation, features should have larger range
        assert X.std() > 1.0

    def test_c_one_all_labeled(self, rng):
        X, y_pu, y_true = make_scar_dataset(c=1.0, random_state=42)
        pos_in_true = y_true == 1
        assert np.all(y_pu[pos_in_true] == 1)
