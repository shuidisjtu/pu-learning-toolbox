"""Tests for the regrouping CPE implementation (ReCPE — Yao et al. 2022).

Covers method card §9: basic flow, custom base estimator, boundary conditions,
and sklearn compatibility.
"""

# ruff: noqa: N803, N806

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.prior import ReCPEEstimator

# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════


def _make_recpe_data(rng, n_p=60, n_pos_u=50, n_neg_u=100):
    """Generate data suitable for ReCPE: overlapping positive class."""
    p = rng.normal(2.0, 0.5, size=(n_p, 2))
    u = np.vstack([
        rng.normal(2.0, 0.5, size=(n_pos_u, 2)),
        rng.normal(-2.0, 0.5, size=(n_neg_u, 2)),
    ])
    X = np.vstack([p, u])
    y = np.concatenate([np.ones(n_p, dtype=int), np.zeros(len(u), dtype=int)])
    return X, y


# ═════════════════════════════════════════════════════════════════════
# §9.1 — Basic flow
# ═════════════════════════════════════════════════════════════════════


class TestBasicFlow:
    """Method card §9.1 — fit, estimate, regrouping attributes."""

    def test_fit_estimate_and_regrouping(self, rng):
        X, y = _make_recpe_data(rng)
        estimator = ReCPEEstimator(copy_fraction=0.1).fit(X, y)

        assert 0.0 <= estimator.estimate() <= 1.0
        expected_n = int(np.ceil(0.1 * (y == 0).sum()))
        assert estimator.copy_count_ == expected_n
        assert len(estimator.selected_indices_) == estimator.copy_count_
        assert estimator.get_metadata()["method"] == "ReCPE"

    def test_small_copy_fraction(self, rng):
        """copy_fraction=0.05 still copies at least 1 sample."""
        X, y = _make_recpe_data(rng, n_p=10, n_pos_u=10, n_neg_u=20)
        estimator = ReCPEEstimator(copy_fraction=0.05).fit(X, y)
        assert estimator.copy_count_ >= 1
        assert 0.0 <= estimator.estimate() <= 1.0

    def test_classifier_is_stored(self, rng):
        """classifier_ attribute is set after fit."""
        X, y = _make_recpe_data(rng)
        estimator = ReCPEEstimator(copy_fraction=0.1).fit(X, y)
        assert hasattr(estimator, "classifier_")
        assert estimator.classifier_ is not None

    def test_base_estimator_is_stored(self, rng):
        """base_estimator_ (including default) is set after fit."""
        X, y = _make_recpe_data(rng)
        estimator = ReCPEEstimator(copy_fraction=0.1).fit(X, y)
        assert hasattr(estimator, "base_estimator_")
        assert estimator.base_estimator_ is not None

    def test_confidence_interval_returns_none(self, rng):
        """confidence_interval() is not supported → returns None."""
        X, y = _make_recpe_data(rng)
        estimator = ReCPEEstimator(copy_fraction=0.1).fit(X, y)
        assert estimator.confidence_interval() is None
        assert estimator.confidence_interval(alpha=0.1) is None


# ═════════════════════════════════════════════════════════════════════
# §9.2 — Custom base estimator
# ═════════════════════════════════════════════════════════════════════


class TestCustomBaseEstimator:
    """Method card §9.2 — base_estimator injection."""

    def test_custom_base_estimator_is_used(self, rng):
        class ConstantPrior:
            def fit(self, X, y):
                self.seen_positive = int(np.sum(y == 1))
                return self

            def estimate(self):
                return 0.37

        X = rng.normal(size=(30, 3))
        y = np.concatenate([np.ones(10, dtype=int), np.zeros(20, dtype=int)])
        estimator = ReCPEEstimator(
            copy_fraction=0.2, base_estimator=ConstantPrior(),
        ).fit(X, y)

        assert estimator.estimate() == pytest.approx(0.37)
        # After regrouping: 10 original + ceil(0.2 * 20) = 14 positives
        assert estimator.base_estimator_.seen_positive == 14

    def test_custom_classifier_is_used(self, rng):
        """Custom classifier replaces the default LogisticRegression."""
        X, y = _make_recpe_data(rng)
        estimator = ReCPEEstimator(
            copy_fraction=0.1,
            classifier=CalibratedClassifierCV(
                SVC(random_state=42), ensemble=False,
            ),
        ).fit(X, y)
        assert estimator.classifier_ is not None
        assert 0.0 <= estimator.estimate() <= 1.0


# ═════════════════════════════════════════════════════════════════════
# §9.3 — Boundary conditions
# ═════════════════════════════════════════════════════════════════════


class TestBoundaryConditions:
    """Method card §9.3 — parameter validation and edge cases."""

    def test_copy_fraction_at_zero_raises(self, rng):
        X, y = _make_recpe_data(rng, n_p=10, n_pos_u=5, n_neg_u=10)
        with pytest.raises(ValueError, match="copy_fraction"):
            ReCPEEstimator(copy_fraction=0.0).fit(X, y)

    def test_copy_fraction_negative_raises(self, rng):
        X, y = _make_recpe_data(rng, n_p=10, n_pos_u=5, n_neg_u=10)
        with pytest.raises(ValueError, match="copy_fraction"):
            ReCPEEstimator(copy_fraction=-0.1).fit(X, y)

    def test_copy_fraction_at_one_raises(self, rng):
        X, y = _make_recpe_data(rng)
        with pytest.raises(ValueError, match="copy_fraction"):
            ReCPEEstimator(copy_fraction=1.0).fit(X, y)

    def test_copy_fraction_above_one_raises(self, rng):
        X, y = _make_recpe_data(rng)
        with pytest.raises(ValueError, match="copy_fraction"):
            ReCPEEstimator(copy_fraction=1.5).fit(X, y)

    def test_not_fitted_estimate_raises(self):
        with pytest.raises(NotFittedError):
            ReCPEEstimator().estimate()

    def test_not_fitted_metadata_raises(self):
        with pytest.raises(NotFittedError):
            ReCPEEstimator().get_metadata()

    def test_no_positive_samples_raises(self, rng):
        X = rng.normal(size=(20, 2))
        y = np.zeros(20, dtype=int)
        with pytest.raises((ValueError, Exception)):
            ReCPEEstimator().fit(X, y)

    def test_no_unlabeled_samples_raises(self, rng):
        X = rng.normal(size=(10, 2))
        y = np.ones(10, dtype=int)
        with pytest.raises(ValueError):
            ReCPEEstimator().fit(X, y)


# ═════════════════════════════════════════════════════════════════════
# sklearn compatibility
# ═════════════════════════════════════════════════════════════════════


class TestSklearnCompatibility:
    """get_params / set_params round-trip."""

    def test_get_params_returns_dict(self):
        estimator = ReCPEEstimator(copy_fraction=0.2)
        params = estimator.get_params()
        assert isinstance(params, dict)
        assert params["copy_fraction"] == 0.2

    def test_set_params_updates(self):
        estimator = ReCPEEstimator(copy_fraction=0.2)
        estimator.set_params(copy_fraction=0.3)
        assert estimator.copy_fraction == 0.3

    def test_get_params_roundtrip(self):
        e1 = ReCPEEstimator(copy_fraction=0.15, classifier_max_iter=500)
        e2 = ReCPEEstimator()
        e2.set_params(**e1.get_params())
        assert e2.get_params() == e1.get_params()
