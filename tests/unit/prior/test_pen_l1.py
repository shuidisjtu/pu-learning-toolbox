# ruff: noqa: N806, B017

import numpy as np
import pytest

from pu_toolbox.prior import ClassPriorEstimator


@pytest.mark.unit
def test_pen_l1_fit_is_deterministic_and_bounded(rng):
    positive = rng.normal(1.0, 0.5, size=(30, 2))
    unlabeled = np.vstack([rng.normal(1.0, 0.5, size=(40, 2)), rng.normal(-1.0, 0.5, size=(80, 2))])
    X = np.vstack([positive, unlabeled])
    y = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(unlabeled), dtype=int)]
    e1 = ClassPriorEstimator(n_centers=50).fit(X, y)
    e2 = ClassPriorEstimator(n_centers=50).fit(X, y)
    assert 0.0 <= e1.estimate() <= 1.0
    assert e1.estimate() == pytest.approx(e2.estimate())
    assert len(e1.objective_values_) == 99


@pytest.mark.unit
def test_pen_l1_rejects_missing_unlabeled(rng):
    X = rng.normal(size=(20, 2))
    with pytest.raises(Exception):
        ClassPriorEstimator().fit(X, np.ones(20, dtype=int))
