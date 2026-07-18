"""Tests for pu_toolbox.core.labels (normalize_pu_labels, normalize_pnu_labels)."""

import numpy as np
import pytest

from pu_toolbox.core.exceptions import ValidationError
from pu_toolbox.core.labels import normalize_pnu_labels, normalize_pu_labels


@pytest.mark.unit
class TestNormalizePuLabels:
    """Unit tests for PU label normalisation — passthrough, remap, errors, edge."""

    def test_basic_and_passthrough(self):
        """Canonical {+1, 0} passes through; dtype is always int."""
        for y in [
            np.array([1, 0, 1, 1, 0]),
            np.array([1, 1, 1]),
            np.array([0, 0, 0]),
        ]:
            result = normalize_pu_labels(y)
            np.testing.assert_array_equal(result, y)
            assert result.dtype.kind == "i"

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

    @pytest.mark.parametrize(
        "y, match",
        [
            (np.array([[1, 0], [0, 1]]), "must be 1-D"),
            (np.array([2, 3, 4]), "Unrecognised label values"),
            (np.array([1, 5, 0]), "Unrecognised label values"),
        ],
    )
    def test_invalid_inputs_raises(self, y, match):
        with pytest.raises(ValidationError, match=match):
            normalize_pu_labels(y)

    def test_edge_empty_array(self):
        y = np.array([], dtype=float)
        result = normalize_pu_labels(y)
        assert result.shape == (0,)
        assert result.dtype.kind == "i"


@pytest.mark.unit
class TestNormalizePnuLabels:
    """Unit tests for PNU label normalisation — passthrough, missing classes, errors, edge."""

    def test_basic_and_passthrough(self):
        """Canonical {+1,-1,0} passes through, dtype is int, large arrays work."""
        # Canonical passthrough
        y = np.array([1, -1, 0, 1, -1, 0])
        result = normalize_pnu_labels(y)
        np.testing.assert_array_equal(result, y)
        assert result.dtype.kind == "i"
        # Minimal three samples
        y_min = np.array([1, -1, 0])
        np.testing.assert_array_equal(normalize_pnu_labels(y_min), y_min)
        # Large arrays smoke test
        rng = np.random.RandomState(42)
        y_large = rng.choice([1, -1, 0], size=1000)
        result_l = normalize_pnu_labels(y_large)
        assert result_l.shape == (1000,)
        assert set(np.unique(result_l)) == {-1, 0, 1}

    @pytest.mark.parametrize(
        "y, match",
        [
            (np.array([1, 0, 1, 0]), "must contain all"),        # missing negative
            (np.array([-1, 0, -1, 0]), "must contain all"),       # missing positive
            (np.array([1, -1, 1, -1]), "must contain all"),       # missing unlabeled
        ],
    )
    def test_missing_class_raises(self, y, match):
        with pytest.raises(ValidationError, match=match):
            normalize_pnu_labels(y)

    @pytest.mark.parametrize(
        "y, match",
        [
            (np.array([[1, -1, 0], [1, -1, 0]]), "must be 1-D"),
            (np.array([1, -1, 0, 2]), "Unrecognised label values"),
            (np.array([1, -1, 0, 3]), "Unrecognised label values"),
        ],
    )
    def test_invalid_inputs_raises(self, y, match):
        with pytest.raises(ValidationError, match=match):
            normalize_pnu_labels(y)

    def test_edge_minimal_and_determinism(self):
        """Smallest valid input and deterministic output."""
        y = np.array([1, -1, 0])
        r1 = normalize_pnu_labels(y)
        r2 = normalize_pnu_labels(y)
        np.testing.assert_array_equal(r1, r2)
        assert r1.dtype.kind == "i"
