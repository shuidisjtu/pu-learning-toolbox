# ruff: noqa: S101

"""Unit tests for the shared class-prior helpers.

Covers ``_class_prior`` helpers:
- basic: formula evaluation and the closed upper bound
- param: invalid ``n`` / ``h`` rejections
- edge: boundary values (``h = 1``, derived prior ``<= 0`` / ``> 1``)
- error: near-singular centroid denominator rejection
- determ: pure-function determinism
"""

from __future__ import annotations

import pytest

from pu_toolbox.estimators.risk._class_prior import (
    solve_prior_from_positive_fraction,
    stable_centroid_denominator,
)


@pytest.mark.unit
class TestSolvePriorFromPositiveFraction:
    """Shared helper backing the KLDCE / LDCE derived class prior."""

    def test_basic_prior_formula(self):
        """p = k / (n * (1 - h)) — 10 / (100 * 0.5) = 0.2."""
        assert solve_prior_from_positive_fraction(10, 100, 0.5) == 0.2

    def test_basic_prior_one_boundary_accepted(self):
        """p == 1.0 is inside the closed upper bound (0, 1]."""
        assert solve_prior_from_positive_fraction(1, 2, 0.5) == 1.0

    def test_param_nonpositive_n_raises(self):
        """n <= 0 is rejected before any computation."""
        with pytest.raises(ValueError, match="n must be positive"):
            solve_prior_from_positive_fraction(1, 0, 0.3)

    def test_param_h_at_one_raises(self):
        """h = 1.0 zeroes the denominator — rejected."""
        with pytest.raises(ValueError, match="h must be in"):
            solve_prior_from_positive_fraction(1, 100, 1.0)

    def test_param_h_outside_range_raises(self):
        """h outside [0, 1) is rejected."""
        with pytest.raises(ValueError, match="h must be in"):
            solve_prior_from_positive_fraction(1, 100, 1.5)

    def test_edge_derived_prior_over_one_raises(self):
        """k / (n * (1 - h)) > 1 raises the derived-prior error."""
        with pytest.raises(ValueError, match="class prior"):
            solve_prior_from_positive_fraction(90, 100, 0.2)

    def test_edge_derived_prior_nonpositive_raises(self):
        """k <= 0 yields p <= 0 — rejected by the (0, 1] range check."""
        with pytest.raises(ValueError, match="class prior"):
            solve_prior_from_positive_fraction(0, 100, 0.3)

    def test_determ_identical_results(self):
        """Pure function: identical inputs yield identical outputs."""
        assert solve_prior_from_positive_fraction(7, 50, 0.4) == (
            solve_prior_from_positive_fraction(7, 50, 0.4)
        )

    def test_basic_centroid_denominator(self):
        """The shared denominator helper returns 1 - 2ph."""
        assert stable_centroid_denominator(0.2, 0.5) == 0.8

    def test_error_near_singular_centroid_denominator(self):
        """Near-zero values are rejected with an actionable error."""
        with pytest.raises(ValueError, match="near-zero.*ill-conditioned"):
            stable_centroid_denominator(1.0, 0.5)
