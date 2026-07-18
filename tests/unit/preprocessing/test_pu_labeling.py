# ruff: noqa: N802, N803, N806

"""Tests for pu_toolbox.preprocessing.pu_labeling (8 tests).

Covers all 6 public functions: smoke, parameter validation, edge cases,
determinism, and the unified dispatcher.
"""

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
# make_scar_labels
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestScarLabels:
    """Covers: basic output, c=1, c out of range, invalid y_true, determinism."""

    def test_basic_and_c_one(self):
        """Output values, shapes, and c=1 labels all positives."""
        y_true = np.array([1] * 30 + [0] * 70)

        # c=0.5 — subset of positives labeled
        y1 = make_scar_labels(y_true, c=0.5, random_state=42)
        assert y1.shape == y_true.shape
        assert y1.dtype == int
        assert set(np.unique(y1)).issubset({1, 0})
        assert 0 < np.sum(y1 == 1) < 30  # some but not all

        # c=1 — all positives labeled, negatives still 0
        y2 = make_scar_labels(y_true, c=1.0, random_state=42)
        assert np.all(y2[y_true == 1] == 1)
        assert np.all(y2[y_true == 0] == 0)

    @pytest.mark.parametrize(
        "c, y_true_vals, match",
        [
            (0.0, [1, 0, 1], "c must be in"),
            (-0.5, [1, 0], "c must be in"),
            (1.5, [1, 0], "c must be in"),
            (0.5, [1, 2, 3], "0, 1"),
            (0.5, [[1, 0], [1, 0]], "1-D"),
        ],
    )
    def test_invalid_inputs(self, c, y_true_vals, match):
        y_true = np.array(y_true_vals)
        with pytest.raises(ValueError, match=match):
            make_scar_labels(y_true, c=c)

    def test_edge_cases_and_determinism(self):
        """No positives → all 0.  Same seed → same output."""
        # All negatives
        y_all_neg = np.zeros(50, dtype=int)
        assert np.all(make_scar_labels(y_all_neg, c=0.5, random_state=42) == 0)

        # All positives — some labeled, some unlabeled, all in {+1, 0}
        y_all_pos = np.ones(20, dtype=int)
        y_pu = make_scar_labels(y_all_pos, c=0.5, random_state=42)
        assert set(np.unique(y_pu)).issubset({1, 0})

        # Determinism
        y_true = np.array([1] * 50 + [0] * 50)
        a = make_scar_labels(y_true, c=0.5, random_state=42)
        b = make_scar_labels(y_true, c=0.5, random_state=42)
        assert np.array_equal(a, b)


# ═════════════════════════════════════════════════════════════════════
# make_case_control_labels
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestCaseControlLabels:
    """Covers: basic output, param validation, determinism."""

    def test_basic_and_edge(self):
        y_true = np.array([1] * 50 + [0] * 50)
        y_pu = make_case_control_labels(y_true, n_labeled=10, random_state=42)
        assert y_pu.shape == y_true.shape
        assert np.sum(y_pu == 1) == 10
        # All negatives unlabeled
        assert np.all(y_pu[y_true == 0] == 0)
        # Determinism
        y2 = make_case_control_labels(y_true, n_labeled=10, random_state=42)
        assert np.array_equal(y_pu, y2)
        # All positives labeled
        y3 = make_case_control_labels(
            np.array([1] * 20 + [0] * 80),
            n_labeled=20,
            random_state=42,
        )
        assert np.sum(y3 == 1) == 20

    @pytest.mark.parametrize(
        "y_true, n_labeled, match",
        [
            ([1] * 10 + [0] * 90, 20, "exceeds"),
            ([1, 0, 1], 0, "n_labeled must be >= 1"),
            ([0] * 10, 5, "both 0.*and.*1"),
            ([1] * 10, 5, "both 0.*and.*1"),
        ],
    )
    def test_invalid_inputs(self, y_true, n_labeled, match):
        with pytest.raises(ValueError, match=match):
            make_case_control_labels(np.array(y_true), n_labeled=n_labeled)


# ═════════════════════════════════════════════════════════════════════
# make_pu_labels (dispatcher)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMakePuLabels:
    """Covers: dispatcher delegation and invalid-mechanism paths."""

    def test_delegation(self):
        """scar → make_scar_labels, case_control → make_case_control_labels."""
        y_true = np.array([1] * 50 + [0] * 50)
        a = make_pu_labels(y_true, mechanism="scar", c=0.5, random_state=42)
        b = make_scar_labels(y_true, c=0.5, random_state=42)
        assert np.array_equal(a, b)

        c = make_pu_labels(
            y_true,
            mechanism="case_control",
            n_labeled=10,
            random_state=42,
        )
        d = make_case_control_labels(y_true, n_labeled=10, random_state=42)
        assert np.array_equal(c, d)

    @pytest.mark.parametrize(
        "mechanism, kwargs, match",
        [
            ("invalid", {"c": 0.5}, "Unknown mechanism"),
            ("scar", {}, "requires the 'c' parameter"),
            ("case_control", {}, "requires the 'n_labeled' parameter"),
        ],
    )
    def test_invalid(self, mechanism, kwargs, match):
        y_true = np.array([1, 0])
        with pytest.raises(ValueError, match=match):
            make_pu_labels(y_true, mechanism=mechanism, **kwargs)


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
# make_gaussian_pu_data + make_scar_dataset
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSyntheticDatasets:
    """Covers: both data generators — shapes, class balance, determinism."""

    def test_gaussian_pu_data(self):
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

    def test_scar_dataset(self):
        X, y_pu, y_true = make_scar_dataset(n=80, c=0.5, random_state=42)
        assert X.shape == (160, 5)
        assert y_pu.shape == (160,)
        assert y_true.shape == (160,)
        assert np.sum(y_true == 1) == 80  # balanced
        assert np.all(y_true[y_pu == 1] == 1)  # labeled ⊆ true positives
        assert set(np.unique(y_pu)).issubset({1, 0})

        # Determinism
        X2, y2, t2 = make_scar_dataset(n=80, c=0.5, random_state=42)
        assert np.allclose(X, X2)
        assert np.array_equal(y_pu, y2)
        assert np.array_equal(y_true, t2)

        # c=1 → all positives labeled
        _, y3, yt3 = make_scar_dataset(n=50, c=1.0, random_state=42)
        assert np.all(y3[yt3 == 1] == 1)
