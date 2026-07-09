"""Tests for pu_toolbox.core.labels.normalize_pu_labels."""

import numpy as np
import pytest

from pu_toolbox.core.labels import normalize_pu_labels
from pu_toolbox.core.exceptions import ValidationError


class TestNormalizePuLabels:
    """Unit tests for label normalisation."""

    # ── Canonical / passthrough ────────────────────────────────────
    def test_canonical_passthrough(self):
        y = np.array([1, 0, 1, 1, 0])
        result = normalize_pu_labels(y)
        np.testing.assert_array_equal(result, y)

    def test_all_positive(self):
        y = np.array([1, 1, 1])
        result = normalize_pu_labels(y)
        np.testing.assert_array_equal(result, y)

    def test_all_unlabeled(self):
        y = np.array([0, 0, 0])
        result = normalize_pu_labels(y)
        np.testing.assert_array_equal(result, y)

    # ── Remapping conventions ──────────────────────────────────────
    @pytest.mark.parametrize(
        "input_labels, expected",
        [
            (np.array([1, -1, 1]), np.array([1, 0, 1])),
            (np.array([-1, -1, 1]), np.array([0, 0, 1])),
            (np.array([1.0, 0.0, 1.0]), np.array([1, 0, 1])),
        ],
    )
    def test_remap(self, input_labels, expected):
        result = normalize_pu_labels(input_labels)
        np.testing.assert_array_equal(result, expected)

    # ── Errors ─────────────────────────────────────────────────────
    def test_2d_array_raises(self):
        y = np.array([[1, 0], [0, 1]])
        with pytest.raises(ValidationError, match="must be 1-D"):
            normalize_pu_labels(y)

    def test_unrecognised_values_raises(self):
        y = np.array([2, 3, 4])
        with pytest.raises(ValidationError, match="Unrecognised label values"):
            normalize_pu_labels(y)

    def test_mixed_invalid_values_raises(self):
        y = np.array([1, 5, 0])
        with pytest.raises(ValidationError, match="Unrecognised label values"):
            normalize_pu_labels(y)

    # ── Edge cases ─────────────────────────────────────────────────
    def test_empty_array(self):
        y = np.array([], dtype=float)
        result = normalize_pu_labels(y)
        assert result.shape == (0,)
        assert result.dtype.kind == "i"

    def test_dtype_int_output(self):
        y = np.array([1.0, 0.0, 1.0])
        result = normalize_pu_labels(y)
        assert result.dtype.kind == "i"
