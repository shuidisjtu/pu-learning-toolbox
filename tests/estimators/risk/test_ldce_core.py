# ruff: noqa: N802, N803, N806, S101

"""Core behaviour tests for LDCEClassifier — fit/predict, attributes, labels."""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.estimators.risk.ldce import LDCEClassifier

# ═════════════════════════════════════════════════════════════════════
# Test helpers
# ═════════════════════════════════════════════════════════════════════


def _make_censoring_pu_data(
    rng: np.random.RandomState,
    n_pos: int = 50,
    n_neg: int = 100,
    h: float = 0.3,
    d: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic censoring PU data with known ground truth."""
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
# Prior calculation and parameter validation
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPriorCalculation:
    """Class-prior formula and edge cases."""

    def test_prior_formula(self, rng):
        rng2 = np.random.RandomState(4)
        X = rng2.randn(100, 3)
        y_true = np.concatenate([np.ones(40, dtype=int), np.zeros(60, dtype=int)])
        hide = rng2.choice(40, size=10, replace=False)
        y_pu = y_true.copy()
        y_pu[hide] = 0
        clf = LDCEClassifier(flip_probability=0.25, max_iter=5, random_state=42)
        clf.fit(X, y_pu)
        # p = k / [n (1-h)] = 30 / [100 * 0.75] = 0.4
        np.testing.assert_allclose(clf.class_prior_, 0.4, rtol=0.01)

    def test_derived_prior_over_one_raises(self, rng):
        rng2 = np.random.RandomState(6)
        X = rng2.randn(30, 3)
        y_pu = np.concatenate([np.ones(20, dtype=int), np.zeros(10, dtype=int)])
        clf = LDCEClassifier(flip_probability=0.8, max_iter=5)
        with pytest.raises(ValueError, match="class prior"):
            clf.fit(X, y_pu)

    def test_class_prior_override(self, rng):
        rng2 = np.random.RandomState(7)
        X = rng2.randn(100, 3)
        y_pu = np.concatenate([np.ones(40, dtype=int), np.zeros(60, dtype=int)])
        clf = LDCEClassifier(flip_probability=0.3, max_iter=5, random_state=42)
        clf.fit(X, y_pu, class_prior=0.55)
        assert clf.class_prior_ == 0.55

    def test_1_minus_2ph_near_zero_raises(self, rng):
        rng2 = np.random.RandomState(8)
        X = rng2.randn(100, 3)
        y_pu = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])
        # h=0.5 makes 1-2ph ≈ 0 (singular case)
        clf = LDCEClassifier(flip_probability=0.5, max_iter=5)
        with pytest.raises(ValueError, match="near-zero"):
            clf.fit(X, y_pu)


# ═════════════════════════════════════════════════════════════════════
# Fitted attributes
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFittedAttributes:
    """Verify all expected attributes are set after fit."""

    def test_all_attributes_present(self, rng):
        rng2 = np.random.RandomState(14)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=30, n_neg=60)
        clf = LDCEClassifier(flip_probability=0.3, max_iter=10, random_state=42)
        clf.fit(X, y_pu)

        assert hasattr(clf, "coef_")
        assert hasattr(clf, "class_prior_")
        assert hasattr(clf, "flip_probability_")
        assert hasattr(clf, "corrupted_centroid_")
        assert hasattr(clf, "true_unlabeled_centroid_")
        assert hasattr(clf, "centroid_covariance_")
        assert hasattr(clf, "n_labeled_")
        assert hasattr(clf, "n_unlabeled_")
        assert hasattr(clf, "n_iter_")
        assert hasattr(clf, "objective_history_")
        assert hasattr(clf, "converged_")
        assert hasattr(clf, "classes_")

        assert isinstance(clf.objective_history_, list)
        assert len(clf.objective_history_) == clf.n_iter_
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))
        assert clf._is_fitted

    def test_attribute_shapes(self, rng):
        rng2_data = np.random.RandomState(150)
        X, y_pu, _ = _make_censoring_pu_data(rng2_data, n_pos=30, n_neg=60)
        d = X.shape[1]
        clf = LDCEClassifier(flip_probability=0.3, max_iter=10, random_state=42)
        clf.fit(X, y_pu)

        assert clf.coef_.shape == (d,)
        assert clf.corrupted_centroid_.shape == (d,)
        assert clf.true_unlabeled_centroid_.shape == (d,)
        assert clf.centroid_covariance_.shape == (d, d)
        assert isinstance(clf.class_prior_, float)
        assert isinstance(clf.flip_probability_, float)


# ═════════════════════════════════════════════════════════════════════
# Label handling
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLabelHandling:
    """Various label conventions should work."""

    def test_1_0_labels(self, rng):
        rng2 = np.random.RandomState(20)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=30, n_neg=60)
        clf = LDCEClassifier(flip_probability=0.3, max_iter=5, random_state=42)
        clf.fit(X, y_pu)
        assert clf._is_fitted

    def test_1_minus1_labels(self, rng):
        rng2 = np.random.RandomState(21)
        X, y_pu_in, _ = _make_censoring_pu_data(rng2, n_pos=30, n_neg=60)
        y_pu = np.where(y_pu_in == 0, -1, 1)
        clf = LDCEClassifier(flip_probability=0.3, max_iter=5, random_state=42)
        clf.fit(X, y_pu)
        assert clf._is_fitted

    def test_positives_not_treated_as_unlabeled(self, rng):
        rng2 = np.random.RandomState(22)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=50, n_neg=100)
        clf = LDCEClassifier(
            flip_probability=0.3,
            max_iter=30,
            tol=1e-4,
            random_state=42,
        )
        clf.fit(X, y_pu)
        mask_P = y_pu == 1
        pred_P = clf.predict(X[mask_P])
        assert np.mean(pred_P == 1) > 0.5
