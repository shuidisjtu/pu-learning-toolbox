# ruff: noqa: N806

import numpy as np
import pytest

from pu_toolbox.prior import ClassPriorEstimator


def _data(rng):
    positive = rng.normal(1.0, 0.5, size=(30, 2))
    unlabeled = np.vstack([rng.normal(1.0, 0.5, size=(40, 2)), rng.normal(-1.0, 0.5, size=(80, 2))])
    X = np.vstack([positive, unlabeled])
    y = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(unlabeled), dtype=int)]
    return X, y


# ── basic ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_basic_estimate_is_bounded(rng):
    X, y = _data(rng)
    est = ClassPriorEstimator(n_centers=50).fit(X, y)
    assert 0.0 <= est.estimate() <= 1.0
    assert len(est.objective_values_) == 99


# ── param validation ───────────────────────────────────────────


@pytest.mark.unit
def test_invalid_sigma_raises(rng):
    X, y = _data(rng)
    with pytest.raises(ValueError):
        ClassPriorEstimator(sigma=-1.0).fit(X, y)


@pytest.mark.unit
def test_invalid_all_positive_raises(rng):
    X = rng.normal(size=(20, 2))
    with pytest.raises(ValueError):
        ClassPriorEstimator().fit(X, np.ones(20, dtype=int))


# ── edge ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_edge_none_n_centers_uses_all(rng):
    X, y = _data(rng)
    est = ClassPriorEstimator(n_centers=None).fit(X, y)
    assert est.n_centers_ == len(X)


@pytest.mark.unit
def test_edge_single_feature(rng):
    p = rng.normal(1.0, 0.5, size=(20, 1))
    u = rng.normal(-1.0, 0.5, size=(40, 1))
    X = np.vstack([p, u])
    y = np.r_[np.ones(len(p), dtype=int), np.zeros(len(u), dtype=int)]
    est = ClassPriorEstimator(n_centers=20).fit(X, y)
    assert 0.0 <= est.estimate() <= 1.0


# ── determ ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_deterministic_estimate_across_runs(rng):
    X, y = _data(rng)
    e1 = ClassPriorEstimator(n_centers=50).fit(X, y)
    e2 = ClassPriorEstimator(n_centers=50).fit(X, y)
    assert e1.estimate() == pytest.approx(e2.estimate())
