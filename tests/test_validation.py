"""Tests for pu_toolbox.core.validation.validate_pu_X_y."""

import numpy as np
import pytest
from scipy import sparse

from pu_toolbox.core.validation import validate_pu_X_y
from pu_toolbox.core.exceptions import ValidationError


class TestValidatePuXY:
    """Unit tests for PU data validation."""

    # ── Happy path ─────────────────────────────────────────────────
    def test_valid_dense(self, simple_X_y_pu):
        X, y_pu, _ = simple_X_y_pu
        X_out, y_out = validate_pu_X_y(X, y_pu)
        assert X_out.shape[0] == y_out.shape[0]
        assert set(np.unique(y_out)) <= {0, 1}

    def test_sparse_accepted_by_default(self):
        X = sparse.csr_matrix(np.eye(10))
        y_pu = np.array([1, 1, 0] + [0] * 7)  # 2 positives
        X_out, y_out = validate_pu_X_y(X, y_pu)
        assert sparse.issparse(X_out)
        assert y_out[0] == 1

    def test_str_labels_remapped(self):
        X = np.random.randn(20, 3)
        y_pu = np.array([1] * 5 + [-1] * 15)  # {+1, -1} convention
        X_out, y_out = validate_pu_X_y(X, y_pu)
        assert set(np.unique(y_out)) == {0, 1}
        assert np.sum(y_out == 1) == 5

    # ── Error cases ────────────────────────────────────────────────
    def test_raises_if_no_positives(self):
        X = np.random.randn(10, 3)
        y_pu = np.zeros(10, dtype=int)
        with pytest.raises(ValidationError, match="labeled positives"):
            validate_pu_X_y(X, y_pu)

    def test_raises_if_only_one_positive(self):
        X = np.random.randn(10, 3)
        y_pu = np.array([1] + [0] * 9)
        with pytest.raises(ValidationError, match="labeled positives"):
            validate_pu_X_y(X, y_pu)

    def test_raises_shape_mismatch(self):
        X = np.random.randn(10, 3)
        y_pu = np.array([1, 0, 1])
        with pytest.raises(ValidationError, match="has 10 samples but y_pu has 3"):
            validate_pu_X_y(X, y_pu)

    def test_raises_ndim_when_allow_nd_false(self):
        X = np.random.randn(10, 4, 4)
        y_pu = np.array([1, 1] + [0] * 8)  # 2 positives
        with pytest.raises(ValidationError, match="Expected X to be 2-D"):
            validate_pu_X_y(X, y_pu)

    def test_3d_allowed_when_flag_set(self):
        X = np.random.randn(10, 4, 4)
        y_pu = np.array([1, 1] + [0] * 8)  # 2 positives
        X_out, y_out = validate_pu_X_y(X, y_pu, allow_nd=True)
        assert X_out.shape == (10, 4, 4)

    def test_sparse_rejected_when_disabled(self):
        X = sparse.csr_matrix(np.eye(10))
        y_pu = np.array([1, 1] + [0] * 8)  # 2 positives
        with pytest.raises(ValidationError, match="Sparse input is not supported"):
            validate_pu_X_y(X, y_pu, accept_sparse=False)

    # ── Warning cases ──────────────────────────────────────────────
    def test_warns_high_pu_ratio(self):
        X = np.random.randn(2004, 3)
        y_pu = np.array([1, 1] + [0] * 2002)  # ratio = 2002/2 = 1001:1 > MAX_PU_RATIO
        with pytest.warns(UserWarning, match="Unlabeled-to-positive ratio"):
            validate_pu_X_y(X, y_pu, estimator_name="TestEstimator")
