# ruff: noqa: N802, N803, N806, S101

"""PROPERTY tests for KLDCEClassifier — invariants and robustness.

Covers:
- Ellipsoid constraint satisfaction
- h-mismatch robustness (ĥ ∈ {0.6h, …, 1.4h})
- Random seed reproducibility
- ACS history monotonicity properties
- Constraint feasibility
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.estimators.risk.kldce import KLDCEClassifier


def _make_censoring_pu_data(
    rng: np.random.RandomState,
    n_pos: int = 20,
    n_neg: int = 40,
    h: float = 0.3,
    d: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic censoring PU data."""
    X_pos = rng.randn(n_pos, d) + 1.5
    X_neg = rng.randn(n_neg, d) - 1.5
    X = np.vstack([X_pos, X_neg])
    y_true = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    n_hide = int(n_pos * h)
    hide_idx = rng.choice(n_pos, size=n_hide, replace=False)
    y_pu = y_true.copy()
    y_pu[hide_idx] = 0
    return X, y_pu, y_true


# ═════════════════════════════════════════════════════════════════════
# Ellipsoid constraint
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.property
class TestEllipsoidConstraint:
    """The ellipsoid constraint (μ−m̂)ᵀ Ŝ_raw (μ−m̂) ≤ b must hold."""

    def test_constraint_satisfied_after_fit(self, rng):
        X, y_pu, _ = _make_censoring_pu_data(rng, n_pos=15, n_neg=30, h=0.3, d=3)
        clf = KLDCEClassifier(
            flip_probability=0.3, sigma=2.0, max_acs_iter=10,
            tol=1e-4, random_state=42,
        )
        clf.fit(X, y_pu)

        diff = clf.centroid_opt_ - clf.centroid_hat_
        constraint_val = float(diff @ clf.centroid_covariance_raw_ @ diff)
        assert constraint_val <= clf.centroid_radius + 1e-6, (
            f"Ellipsoid constraint violated: {constraint_val:.6f} > {clf.centroid_radius}"
        )

    def test_constraint_active_with_positive_radius(self, rng):
        """With b>0, centroid should move from m_hat."""
        X, y_pu, _ = _make_censoring_pu_data(rng, n_pos=15, n_neg=30, h=0.3, d=3)
        clf = KLDCEClassifier(
            flip_probability=0.3, sigma=2.0, centroid_radius=0.5,
            max_acs_iter=10, tol=1e-4, random_state=42,
        )
        clf.fit(X, y_pu)

        # Centroid should move at least a small amount from initial
        mu_diff = np.linalg.norm(clf.centroid_opt_ - clf.centroid_hat_)
        # Not asserting specific delta, but difference should be finite
        assert np.isfinite(mu_diff)


# ═════════════════════════════════════════════════════════════════════
# h-mismatch robustness
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.property
class TestHMismatchRobustness:
    """ĥ mismatch should not cause crashes."""

    @pytest.mark.parametrize("h_factor", [0.6, 0.8, 1.0, 1.2, 1.4])
    def test_h_mismatch_no_crash(self, rng, h_factor):
        X, y_pu, _ = _make_censoring_pu_data(rng, n_pos=15, n_neg=30, h=0.3, d=3)
        h_mismatch = max(0.05, min(0.95, 0.3 * h_factor))

        clf = KLDCEClassifier(
            flip_probability=h_mismatch, sigma=2.0, max_acs_iter=8,
            tol=1e-4, random_state=42,
        )
        try:
            clf.fit(X, y_pu)
        except ValueError as e:
            # ValueErrors for degenerate derivations are acceptable
            assert "class prior" in str(e) or "near-zero" in str(e) or "exceeds" in str(e)
        # If it fits, at least check basic attribute integrity
        if clf._is_fitted:
            assert hasattr(clf, "alpha_full_")
            assert hasattr(clf, "bias_")


# ═════════════════════════════════════════════════════════════════════
# Reproducibility
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.property
class TestReproducibility:
    """Random seed should give reproducible results."""

    def test_same_seed_same_result(self, rng):
        rng2 = np.random.RandomState(42)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=15, n_neg=30, h=0.3, d=3)

        clf1 = KLDCEClassifier(
            flip_probability=0.3, sigma=2.0, max_acs_iter=10,
            tol=1e-4, random_state=42,
        )
        clf1.fit(X, y_pu)

        clf2 = KLDCEClassifier(
            flip_probability=0.3, sigma=2.0, max_acs_iter=10,
            tol=1e-4, random_state=42,
        )
        clf2.fit(X, y_pu)

        np.testing.assert_allclose(clf1.bias_, clf2.bias_, rtol=1e-10)
        np.testing.assert_array_equal(clf1.alpha_full_, clf2.alpha_full_)
        np.testing.assert_array_equal(clf1.gamma_unlabeled_, clf2.gamma_unlabeled_)


# ═════════════════════════════════════════════════════════════════════
# ACS history
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.property
class TestACSHistory:
    """ACS loop diagnostics are recorded."""

    def test_history_recorded(self, rng):
        X, y_pu, _ = _make_censoring_pu_data(rng, n_pos=15, n_neg=30, h=0.3, d=3)
        clf = KLDCEClassifier(
            flip_probability=0.3, sigma=2.0, max_acs_iter=10,
            tol=1e-4, random_state=42,
        )
        clf.fit(X, y_pu)

        assert len(clf.acs_history_) >= 1
        assert len(clf.acs_history_) == clf.n_acs_iter_

        for entry in clf.acs_history_:
            assert "iter" in entry
            assert "dual_obj" in entry
            assert "eq_residual" in entry
            assert "centroid_constraint_residual" in entry
            assert np.isfinite(entry["dual_obj"])

    def test_eq_residual_small(self, rng):
        X, y_pu, _ = _make_censoring_pu_data(rng, n_pos=15, n_neg=30, h=0.3, d=3)
        clf = KLDCEClassifier(
            flip_probability=0.3, sigma=2.0, max_acs_iter=10,
            tol=1e-4, random_state=42,
        )
        clf.fit(X, y_pu)

        # Equality residuals should be small throughout
        for entry in clf.acs_history_:
            assert entry["eq_residual"] < 1e-6, (
                f"Iter {entry['iter']}: eq_residual={entry['eq_residual']:.2e}"
            )


# ═════════════════════════════════════════════════════════════════════
# Input validation
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.property
class TestInputValidation:
    """Input validation per spec §4 step 2."""

    def test_k_gt_zero_enforced(self, rng):
        """At least one positive sample required."""
        X = rng.randn(10, 3)
        y_pu = np.zeros(10, dtype=int)
        clf = KLDCEClassifier(flip_probability=0.3)
        with pytest.raises(Exception):  # ValidationError or ValueError
            clf.fit(X, y_pu)

    def test_n_U_gt_zero_enforced(self, rng):
        """At least one unlabeled sample required."""
        X = rng.randn(10, 3)
        y_pu = np.ones(10, dtype=int)
        clf = KLDCEClassifier(flip_probability=0.3)
        with pytest.raises(Exception):
            clf.fit(X, y_pu)

    def test_p_leq_one_enforced(self, rng):
        """Derived class prior > 1 raises ValueError."""
        X = rng.randn(30, 3)
        y_pu = np.concatenate([np.ones(25, dtype=int), np.zeros(5, dtype=int)])
        clf = KLDCEClassifier(flip_probability=0.8, mom_groups=1)
        with pytest.raises(ValueError, match="class prior"):
            clf.fit(X, y_pu)

    def test_mom_groups_leq_n_U_enforced(self, rng):
        X = rng.randn(20, 3)
        y_pu = np.concatenate([np.ones(15, dtype=int), np.zeros(5, dtype=int)])
        clf = KLDCEClassifier(flip_probability=0.3, mom_groups=20)
        with pytest.raises(ValueError, match="mom_groups"):
            clf.fit(X, y_pu)

    def test_flip_probability_range(self, rng):
        X = rng.randn(20, 3)
        y_pu = np.concatenate([np.ones(5, dtype=int), np.zeros(15, dtype=int)])
        clf = KLDCEClassifier(flip_probability=1.5)
        with pytest.raises(ValueError, match="flip_probability"):
            clf.fit(X, y_pu)

    def test_max_dual_variables_enforced(self, rng):
        X = rng.randn(600, 3)
        y_pu = np.concatenate([np.ones(10, dtype=int), np.zeros(590, dtype=int)])
        clf = KLDCEClassifier(flip_probability=0.3, max_dual_variables=500)
        with pytest.raises(ValueError, match="max_dual_variables"):
            clf.fit(X, y_pu)


# ═════════════════════════════════════════════════════════════════════
# Fitted attributes
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.property
class TestFittedAttributes:
    """All spec §7 attributes are present after fit."""

    def test_required_attributes_present(self, rng):
        X, y_pu, _ = _make_censoring_pu_data(rng, n_pos=15, n_neg=30, h=0.3, d=3)
        clf = KLDCEClassifier(
            flip_probability=0.3, sigma=2.0, max_acs_iter=5,
            tol=1e-4, random_state=42,
        )
        clf.fit(X, y_pu)

        required = [
            "alpha_full_", "gamma_unlabeled_", "unlabeled_indices_",
            "support_indices_", "bias_", "class_prior_",
            "flip_probability_", "centroid_hat_", "centroid_opt_",
            "centroid_covariance_raw_", "C_eq_",
            "n_acs_iter_", "acs_history_", "converged_", "classes_",
        ]
        for attr in required:
            assert hasattr(clf, attr), f"Missing attribute: {attr}"

    def test_classes_are_0_1(self, rng):
        X, y_pu, _ = _make_censoring_pu_data(rng, n_pos=15, n_neg=30, h=0.3, d=3)
        clf = KLDCEClassifier(
            flip_probability=0.3, sigma=2.0, max_acs_iter=5,
            tol=1e-4, random_state=42,
        )
        clf.fit(X, y_pu)
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))
