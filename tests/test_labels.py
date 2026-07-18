"""Tests for pu_toolbox.core.labels (normalize_pu_labels, normalize_pnu_labels)."""

import numpy as np
import pytest

from pu_toolbox.core.exceptions import ValidationError
from pu_toolbox.core.labels import normalize_pnu_labels, normalize_pu_labels


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


class TestNormalizePnuLabels:
    """Unit tests for P/N/U label normalisation."""

    # ── Canonical / passthrough ────────────────────────────────────
    def test_canonical_passthrough(self):
        y = np.array([1, -1, 0, 1, -1, 0])
        result = normalize_pnu_labels(y)
        np.testing.assert_array_equal(result, y)

    def test_remap_one_to_positive(self):
        """Standard binary {1, -1, 0} → canonical {+1, -1, 0}."""
        y = np.array([1, -1, 0, 1, 0])
        result = normalize_pnu_labels(y)
        expected = np.array([1, -1, 0, 1, 0])
        np.testing.assert_array_equal(result, expected)

    # ── Require all three classes ──────────────────────────────────
    def test_missing_negative_raises(self):
        y = np.array([1, 0, 1, 0])
        with pytest.raises(ValidationError, match="must contain all"):
            normalize_pnu_labels(y)

    def test_missing_positive_raises(self):
        y = np.array([-1, 0, -1, 0])
        with pytest.raises(ValidationError, match="must contain all"):
            normalize_pnu_labels(y)

    def test_missing_unlabeled_raises(self):
        y = np.array([1, -1, 1, -1])
        with pytest.raises(ValidationError, match="must contain all"):
            normalize_pnu_labels(y)

    def test_only_two_classes_raises(self):
        """{+1, -1} without 0 must be rejected (not treated as P/U)."""
        y = np.array([1, -1, 1, -1])
        with pytest.raises(ValidationError, match="must contain all"):
            normalize_pnu_labels(y)

    # ── Errors ─────────────────────────────────────────────────────
    def test_2d_array_raises(self):
        y = np.array([[1, -1, 0], [1, -1, 0]])
        with pytest.raises(ValidationError, match="must be 1-D"):
            normalize_pnu_labels(y)

    def test_unrecognised_values_raises(self):
        y = np.array([1, -1, 0, 2])
        with pytest.raises(ValidationError, match="Unrecognised label values"):
            normalize_pnu_labels(y)

    def test_four_value_labels_raises(self):
        y = np.array([1, -1, 0, 3])
        with pytest.raises(ValidationError, match="Unrecognised label values"):
            normalize_pnu_labels(y)

    # ── Edge cases ─────────────────────────────────────────────────
    def test_minimal_three_samples(self):
        """Smallest valid PNU input: exactly one P, N, U each."""
        y = np.array([1, -1, 0])
        result = normalize_pnu_labels(y)
        np.testing.assert_array_equal(result, y)

    def test_dtype_int_output(self):
        y = np.array([1.0, -1.0, 0.0])
        result = normalize_pnu_labels(y)
        assert result.dtype.kind == "i"

    def test_large_arrays(self):
        """Smoke test with many samples."""
        rng = np.random.RandomState(42)
        y = rng.choice([1, -1, 0], size=1000)
        result = normalize_pnu_labels(y)
        assert result.shape == (1000,)
        assert set(np.unique(result)) == {-1, 0, 1}
