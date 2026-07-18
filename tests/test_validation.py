"""Tests for pu_toolbox.core.validation (validate_pu_X_y, validate_pnu_X_y)."""

import numpy as np
import pytest
from scipy import sparse

from pu_toolbox.core.exceptions import ValidationError
from pu_toolbox.core.validation import validate_pnu_X_y, validate_pu_X_y


class TestValidatePuXY:
    """Unit tests for PU data validation."""

    # ── Happy path ─────────────────────────────────────────────────
    def test_valid_dense(self, simple_x_y_pu):
        x, y_pu, _ = simple_x_y_pu
        x_out, y_out = validate_pu_X_y(x, y_pu)
        assert x_out.shape[0] == y_out.shape[0]
        assert set(np.unique(y_out)) <= {0, 1}

    def test_sparse_accepted_by_default(self):
        x = sparse.csr_matrix(np.eye(10))
        y_pu = np.array([1, 1, 0] + [0] * 7)  # 2 positives
        x_out, y_out = validate_pu_X_y(x, y_pu)
        assert sparse.issparse(x_out)
        assert y_out[0] == 1

    def test_str_labels_remapped(self):
        x = np.random.randn(20, 3)
        y_pu = np.array([1] * 5 + [-1] * 15)  # {+1, -1} convention
        x_out, y_out = validate_pu_X_y(x, y_pu)
        assert set(np.unique(y_out)) == {0, 1}
        assert np.sum(y_out == 1) == 5

    # ── Error cases ────────────────────────────────────────────────
    def test_raises_if_no_positives(self):
        x = np.random.randn(10, 3)
        y_pu = np.zeros(10, dtype=int)
        with pytest.raises(ValidationError, match="labeled positives"):
            validate_pu_X_y(x, y_pu)

    def test_raises_if_only_one_positive(self):
        x = np.random.randn(10, 3)
        y_pu = np.array([1] + [0] * 9)
        with pytest.raises(ValidationError, match="labeled positives"):
            validate_pu_X_y(x, y_pu)

    def test_raises_shape_mismatch(self):
        x = np.random.randn(10, 3)
        y_pu = np.array([1, 0, 1])
        with pytest.raises(ValidationError, match="has 10 samples but y_pu has 3"):
            validate_pu_X_y(x, y_pu)

    def test_raises_ndim_when_allow_nd_false(self):
        x = np.random.randn(10, 4, 4)
        y_pu = np.array([1, 1] + [0] * 8)  # 2 positives
        with pytest.raises(ValidationError, match="Expected X to be 2-D"):
            validate_pu_X_y(x, y_pu)

    def test_3d_allowed_when_flag_set(self):
        x = np.random.randn(10, 4, 4)
        y_pu = np.array([1, 1] + [0] * 8)  # 2 positives
        x_out, y_out = validate_pu_X_y(x, y_pu, allow_nd=True)
        assert x_out.shape == (10, 4, 4)

    def test_sparse_rejected_when_disabled(self):
        x = sparse.csr_matrix(np.eye(10))
        y_pu = np.array([1, 1] + [0] * 8)  # 2 positives
        with pytest.raises(ValidationError, match="Sparse input is not supported"):
            validate_pu_X_y(x, y_pu, accept_sparse=False)

    # ── Warning cases ──────────────────────────────────────────────
    def test_warns_high_pu_ratio(self):
        x = np.random.randn(2004, 3)
        y_pu = np.array([1, 1] + [0] * 2002)  # ratio = 2002/2 = 1001:1 > MAX_PU_RATIO
        with pytest.warns(UserWarning, match="Unlabeled-to-positive ratio"):
            validate_pu_X_y(x, y_pu, estimator_name="TestEstimator")


class TestValidatePnuXY:
    """Unit tests for PNU data validation."""

    # ── Happy path ─────────────────────────────────────────────────
    def test_valid_dense(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 5)
        y = np.concatenate([
            np.full(20, 1), np.full(20, -1), np.zeros(20),
        ]).astype(int)
        X_out, y_out = validate_pnu_X_y(X, y)
        assert X_out.shape[0] == y_out.shape[0]
        assert set(np.unique(y_out)) == {-1, 0, 1}

    def test_accepts_one_of_each_class(self):
        X = np.random.randn(3, 3)
        y = np.array([1, -1, 0])
        X_out, y_out = validate_pnu_X_y(X, y)
        np.testing.assert_array_equal(y_out, y)

    # ── Error: missing class ───────────────────────────────────────
    def test_no_positive_raises(self):
        X = np.random.randn(30, 3)
        y = np.concatenate([np.full(15, -1), np.zeros(15)])
        with pytest.raises(ValidationError, match="must contain all"):
            validate_pnu_X_y(X, y)

    def test_no_negative_raises(self):
        X = np.random.randn(30, 3)
        y = np.concatenate([np.full(15, 1), np.zeros(15)])
        with pytest.raises(ValidationError, match="must contain all"):
            validate_pnu_X_y(X, y)

    def test_no_unlabeled_raises(self):
        X = np.random.randn(30, 3)
        y = np.concatenate([np.full(15, 1), np.full(15, -1)])
        with pytest.raises(ValidationError, match="must contain all"):
            validate_pnu_X_y(X, y)

    # ── Error: other ───────────────────────────────────────────────
    def test_shape_mismatch_raises(self):
        X = np.random.randn(10, 3)
        y = np.array([1, -1, 0, 1, -1, 0])  # 6 labels, 10 rows
        with pytest.raises(ValidationError, match="has 10 samples"):
            validate_pnu_X_y(X, y)

    def test_ndim_raises_by_default(self):
        X = np.random.randn(10, 4, 4)
        y = np.concatenate([np.full(4, 1), np.full(3, -1), np.full(3, 0)])
        with pytest.raises(ValidationError, match="Expected X to be 2-D"):
            validate_pnu_X_y(X, y)

    def test_3d_allowed_when_flag_set(self):
        X = np.random.randn(10, 4, 4)
        y = np.concatenate([np.full(4, 1), np.full(3, -1), np.full(3, 0)])
        X_out, y_out = validate_pnu_X_y(X, y, allow_nd=True)
        assert X_out.shape == (10, 4, 4)

    def test_unrecognised_values_raises(self):
        X = np.random.randn(10, 3)
        y = np.concatenate([np.full(3, 1), np.full(3, -1), np.full(3, 0), [2]])
        with pytest.raises(ValidationError, match="Unrecognised label"):
            validate_pnu_X_y(X, y)
