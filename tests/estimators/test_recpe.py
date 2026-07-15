"""Tests for the regrouping CPE implementation."""

# ruff: noqa: N803, N806

import numpy as np
import pytest

from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.prior import ReCPEEstimator


class TestReCPE:
    def test_fit_estimate_and_regrouping(self, rng):
        p = rng.normal(2.0, 0.5, size=(60, 2))
        u = np.vstack([rng.normal(2.0, 0.5, size=(50, 2)), rng.normal(-2.0, 0.5, size=(100, 2))])
        X = np.vstack([p, u])
        y = np.r_[np.ones(len(p), dtype=int), np.zeros(len(u), dtype=int)]

        estimator = ReCPEEstimator(copy_fraction=0.1).fit(X, y)

        assert 0.0 <= estimator.estimate() <= 1.0
        assert estimator.copy_count_ == int(np.ceil(0.1 * len(u)))
        assert len(estimator.selected_indices_) == estimator.copy_count_
        assert estimator.get_metadata()["method"] == "ReCPE"

    def test_custom_base_estimator_is_used(self, rng):
        class ConstantPrior:
            def fit(self, X, y):
                self.seen_positive = int(np.sum(y == 1))
                return self

            def estimate(self):
                return 0.37

        X = rng.normal(size=(30, 3))
        y = np.r_[np.ones(10, dtype=int), np.zeros(20, dtype=int)]
        estimator = ReCPEEstimator(copy_fraction=0.2, base_estimator=ConstantPrior()).fit(X, y)

        assert estimator.estimate() == pytest.approx(0.37)
        assert estimator.base_estimator_.seen_positive == 14

    def test_invalid_fraction_and_unfitted(self, rng):
        X = rng.normal(size=(10, 2))
        y = np.r_[np.ones(5, dtype=int), np.zeros(5, dtype=int)]
        with pytest.raises(ValueError, match="copy_fraction"):
            ReCPEEstimator(copy_fraction=1.0).fit(X, y)
        with pytest.raises(NotFittedError):
            ReCPEEstimator().estimate()
