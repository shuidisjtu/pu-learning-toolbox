# ruff: noqa: N806, E501

"""Tests for pu_toolbox.core.validation (validate_pu_X_y, validate_pnu_X_y)."""

import numpy as np
import pytest
from scipy import sparse

from pu_toolbox.core.exceptions import ValidationError
from pu_toolbox.core.validation import validate_pnu_X_y, validate_pu_X_y


@pytest.mark.unit
class TestValidatePuXY:
    """Unit tests for PU data validation — basic, errors, edge."""

    # ── Basic / happy path ────────────────────────────────────────────
    def test_basic_validation(self, simple_x_y_pu):
        """Dense input, sparse input, and {+1, -1} remapping."""
        # Dense with fixture
        x, y_pu, _ = simple_x_y_pu
        x_out, y_out = validate_pu_X_y(x, y_pu)
        assert x_out.shape[0] == y_out.shape[0]
        assert set(np.unique(y_out)) <= {0, 1}

        # Sparse
        xs = sparse.csr_matrix(np.eye(10))
        ys = np.array([1, 1, 0] + [0] * 7)
        xo, yo = validate_pu_X_y(xs, ys)
        assert sparse.issparse(xo)

        # {+1, -1} → {+1, 0}
        xr = np.random.randn(20, 3)
        yr = np.array([1] * 5 + [-1] * 15)
        _, yo2 = validate_pu_X_y(xr, yr)
        assert set(np.unique(yo2)) == {0, 1}
        assert np.sum(yo2 == 1) == 5

    # ── Error cases ───────────────────────────────────────────────────
    @pytest.mark.parametrize(
        "y_pu, match",
        [
            (np.zeros(10, dtype=int), "labeled positives"),
            (np.array([1] + [0] * 9), "labeled positives"),
        ],
    )
    def test_no_or_one_positive_raises(self, y_pu, match):
        x = np.random.randn(10, 3)
        with pytest.raises(ValidationError, match=match):
            validate_pu_X_y(x, y_pu)

    @pytest.mark.parametrize(
        "x, y_pu, kwargs, match",
        [
            (np.random.randn(10, 3), np.array([1, 0, 1]), {}, "has 10 samples"),
            (np.random.randn(10, 4, 4), np.array([1, 1] + [0] * 8), {}, "Expected X to be 2-D"),
            (
                sparse.csr_matrix(np.eye(10)),
                np.array([1, 1] + [0] * 8),
                {"accept_sparse": False},
                "Sparse input is not supported",
            ),
        ],
    )
    def test_shape_and_dimension_errors(self, x, y_pu, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            validate_pu_X_y(x, y_pu, **kwargs)

    def test_edge_3d_allowed_when_flag_set(self):
        """allow_nd=True permits 3-D input."""
        x = np.random.randn(10, 4, 4)
        y_pu = np.array([1, 1] + [0] * 8)
        x_out, y_out = validate_pu_X_y(x, y_pu, allow_nd=True)
        assert x_out.shape == (10, 4, 4)

    def test_edge_high_pu_ratio_warns(self):
        """PU ratio > MAX_PU_RATIO triggers UserWarning."""
        x = np.random.randn(2004, 3)
        y_pu = np.array([1, 1] + [0] * 2002)
        with pytest.warns(UserWarning, match="Unlabeled-to-positive ratio"):
            validate_pu_X_y(x, y_pu, estimator_name="TestEstimator")
        # Determinism: same input → same output
        x2, y2 = validate_pu_X_y(x, y_pu)
        x3, y3 = validate_pu_X_y(x, y_pu)
        np.testing.assert_array_equal(x2, x3)
        np.testing.assert_array_equal(y2, y3)


@pytest.mark.unit
class TestValidatePnuXY:
    """Unit tests for PNU data validation — basic, errors, edge."""

    # ── Basic / happy path ────────────────────────────────────────────
    def test_basic_validation(self):
        """Dense PNU data and minimal (1, -1, 0) input."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 5)
        y = np.concatenate([np.full(20, 1), np.full(20, -1), np.zeros(20)]).astype(int)
        X_out, y_out = validate_pnu_X_y(X, y)
        assert X_out.shape[0] == y_out.shape[0]
        assert set(np.unique(y_out)) == {-1, 0, 1}

        # Minimal one-of-each
        X_min = np.random.randn(3, 3)
        y_min = np.array([1, -1, 0])
        _, yo = validate_pnu_X_y(X_min, y_min)
        np.testing.assert_array_equal(yo, y_min)

    # ── Error: missing class ──────────────────────────────────────────
    @pytest.mark.parametrize(
        "y, match",
        [
            (np.array([-1, -1, 0, 0], dtype=int), "must contain all"),   # no positive
            (np.array([1, 1, 0, 0], dtype=int), "must contain all"),     # no negative
            (np.array([1, -1, 1, -1], dtype=int), "must contain all"),   # no unlabeled
        ],
    )
    def test_missing_class_raises(self, y, match):
        X = np.random.randn(len(y), 3)
        with pytest.raises(ValidationError, match=match):
            validate_pnu_X_y(X, y)

    # ── Error: shape / dimension ──────────────────────────────────────
    @pytest.mark.parametrize(
        "x, y, kwargs, match",
        [
            (np.random.randn(10, 3), np.array([1, -1, 0, 1, -1, 0]), {}, "has 10 samples"),
            (np.random.randn(10, 4, 4), np.array([1, 1, 1, -1, -1, -1, 0, 0, 0, 0]), {}, "Expected X to be 2-D"),
            (
                np.random.randn(10, 3),
                np.array([1, 1, 1, -1, -1, -1, 0, 0, 0, 2]),
                {},
                "Unrecognised label",
            ),
        ],
    )
    def test_shape_and_dimension_errors(self, x, y, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            validate_pnu_X_y(x, y, **kwargs)

    def test_edge_3d_allowed(self):
        """allow_nd=True permits 3-D input for PNU."""
        X = np.random.randn(10, 4, 4)
        y = np.concatenate([np.full(4, 1), np.full(3, -1), np.full(3, 0)])
        X_out, y_out = validate_pnu_X_y(X, y, allow_nd=True)
        assert X_out.shape == (10, 4, 4)

    def test_edge_determinism(self):
        """Same input → same output (deterministic validation)."""
        X = np.random.randn(10, 3)
        y = np.concatenate([np.full(3, 1), np.full(3, -1), np.full(4, 0)])
        x1, y1 = validate_pnu_X_y(X, y)
        x2, y2 = validate_pnu_X_y(X, y)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
