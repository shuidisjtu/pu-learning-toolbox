# ruff: noqa: N802, N803, N806

"""Tests for pu_toolbox.preprocessing.profiling (4 tests).

Covers: pu_data_summary and pnu_data_summary — counts, ratios, edge
cases, NaN/Inf flags, sparse input, and label normalisation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from pu_toolbox.preprocessing.profiling import (
    pnu_data_summary,
    pu_data_summary,
    scar_diagnostic,
)

# ═════════════════════════════════════════════════════════════════════
# pu_data_summary
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPuDataSummary:
    """Covers: counts, ratios, edge cases, NaN/Inf, sparse, validation."""

    def test_basic_counts(self):
        X = np.zeros((100, 5))
        y_pu = np.array([1] * 30 + [0] * 70)
        s = pu_data_summary(X, y_pu)
        assert s["n_samples"] == 100
        assert s["n_features"] == s["n_features_out"] == 5
        assert s["n_positives"] == 30
        assert s["n_unlabeled"] == 70
        assert s["pu_ratio"] == pytest.approx(70 / 30)
        assert s["positive_fraction"] == 0.3
        assert s["is_sparse"] is False

    def test_edge_cases(self):
        # Zero positives
        s = pu_data_summary(np.zeros((100, 5)), np.zeros(100, dtype=int))
        assert s["n_positives"] == 0
        assert s["pu_ratio"] == float("inf")
        assert s["positive_fraction"] == 0.0

        # All positive
        s = pu_data_summary(np.zeros((50, 3)), np.ones(50, dtype=int))
        assert s["n_positives"] == 50
        assert s["n_unlabeled"] == 0
        assert s["pu_ratio"] == 0.0

        # Single sample
        s = pu_data_summary(np.zeros((1, 3)), np.array([1]))
        assert s["n_samples"] == 1

        # Sparse input
        X = sparse.csr_matrix(np.zeros((100, 5)))
        s = pu_data_summary(X, np.array([1] * 30 + [0] * 70))
        assert s["is_sparse"] is True

    def test_nan_and_inf(self):
        X = np.zeros((100, 5))
        X[5, 2] = np.nan
        X[10, 1] = np.inf
        s = pu_data_summary(X, np.array([1] * 30 + [0] * 70))
        assert s["has_nan"] is True
        assert s["has_inf"] is True

    def test_validation(self):
        # Shape mismatch
        with pytest.raises(ValueError, match="has 100 samples"):
            pu_data_summary(np.zeros((100, 5)), np.array([1] * 30 + [0] * 50))

        # Non-canonical {1, -1} labels normalized to {+1, 0}
        s = pu_data_summary(np.zeros((100, 5)), np.array([1] * 30 + [-1] * 70))
        assert s["n_positives"] == 30
        assert s["n_unlabeled"] == 70


# ═════════════════════════════════════════════════════════════════════
# pnu_data_summary
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPnuDataSummary:
    """Covers: PNU-specific counts, ratios, and sparse input."""

    def test_basic(self):
        X = np.zeros((100, 5))
        y_pnu = np.array([1] * 20 + [-1] * 30 + [0] * 50)
        s = pnu_data_summary(X, y_pnu)
        assert s["n_samples"] == 100
        assert s["n_positives"] == 20
        assert s["n_negatives"] == 30
        assert s["n_unlabeled"] == 50
        assert s["pu_ratio"] == 2.5
        assert s["nu_ratio"] == pytest.approx(50 / 30)
        assert s["pn_ratio"] == pytest.approx(20 / 30)

        # Sparse
        s_sp = pnu_data_summary(sparse.csr_matrix(X), y_pnu)
        assert s_sp["is_sparse"] is True

        # Shape mismatch
        with pytest.raises(ValueError, match="has 100 samples"):
            pnu_data_summary(X, np.array([1] * 10 + [-1] * 10 + [0] * 10))

        # NaN / Inf
        X2 = X.copy()
        X2[0, 0] = np.nan
        X2[1, 1] = np.inf
        s2 = pnu_data_summary(X2, y_pnu)
        assert s2["has_nan"] and s2["has_inf"]


# ═════════════════════════════════════════════════════════════════════
# scar_diagnostic
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestScarDiagnostic:
    """Covers: SCAR plausibility check and return structure."""

    def test_scar_diagnostic_scar_data(self):
        """SCAR-generated labels should yield is_scar_plausible=True."""
        rng = np.random.RandomState(42)
        n = 300
        # Features are pure noise — independent of class and label.
        # Under SCAR, labeling probability does not depend on features,
        # so labeled-P and U should be inseparable in feature space.
        X = rng.randn(n, 5)
        # Assign labels uniformly at random (no feature dependence)
        y_pu = (rng.rand(n) < 0.3).astype(int)

        result = scar_diagnostic(X, y_pu)
        assert result["is_scar_plausible"] is True

    def test_scar_diagnostic_returns_dict(self):
        """Return value contains all required keys with correct types."""
        rng = np.random.RandomState(0)
        X = rng.randn(100, 3)
        y_pu = np.array([1] * 30 + [0] * 70)

        result = scar_diagnostic(X, y_pu)
        assert isinstance(result, dict)
        assert "separability_auc" in result
        assert "is_scar_plausible" in result
        assert "message" in result
        assert isinstance(result["separability_auc"], float)
        assert isinstance(result["is_scar_plausible"], bool)
        assert isinstance(result["message"], str)
        assert 0.0 <= result["separability_auc"] <= 1.0
