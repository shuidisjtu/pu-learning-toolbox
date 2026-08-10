# ruff: noqa: N803, N806

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from pu_toolbox.prior import ClassPriorEstimator
from tests.helpers import make_scar_data, make_scar_data_unbalanced

AUTO_SIGMA_FACTOR = 0.6  # must match pu_toolbox.prior.pen_l1._AUTO_SIGMA_FACTOR


def _auto_sigma(X):
    """Replicate the estimator's auto-sigma formula (standardize + median)."""
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
    d = pairwise_distances(Xs)
    return AUTO_SIGMA_FACTOR * float(np.median(d[np.triu_indices(len(Xs), k=1)]))


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


# ── math (auto-sigma golden) ──────────────────────────────────


@pytest.mark.math
def test_math_auto_sigma_matches_explicit_median_sigma(rng):
    """Auto sigma (default) reproduces the explicit median-distance sigma."""
    X, y_pu, _ = make_scar_data(rng, n=200, separation=2.0)
    auto = ClassPriorEstimator().fit(X, y_pu)
    med = _auto_sigma(X)
    explicit = ClassPriorEstimator(sigma=med).fit(X, y_pu)
    assert auto.sigma_ == pytest.approx(med)
    assert auto.estimate() == pytest.approx(explicit.estimate(), abs=0.02)


@pytest.mark.math
def test_math_auto_sigma_lands_in_band(rng):
    """Default auto-sigma lands in the acceptance band for prior 0.5.

    The band reflects the eta=0.6 compromise: strong separation (~0.33)
    vs heavy overlap (~0.74) both stay inside the wider 0.25-0.75 guard.
    """
    X, y_pu, _ = make_scar_data(rng, n=200, separation=2.0)
    est = ClassPriorEstimator().fit(X, y_pu).estimate()
    assert 0.30 <= est <= 0.70


# ── param (cross-separation / cross-prior regression guard) ────


@pytest.mark.unit
@pytest.mark.parametrize("separation", [0.5, 1.0, 2.0])
def test_param_separation_prior_band(rng, separation):
    """Prior-0.5 estimates stay within 0.5x-1.5x of the true prior."""
    X, y_pu, _ = make_scar_data_unbalanced(rng, n=150, prior=0.5, separation=separation)
    est = ClassPriorEstimator().fit(X, y_pu).estimate()
    assert 0.5 * 0.5 <= est <= 1.5 * 0.5


@pytest.mark.unit
@pytest.mark.parametrize("separation", [0.5, 1.0, 2.0])
def test_param_low_prior_sanity_band(rng, separation):
    """Prior-0.3 sanity band (auto sigma overestimates low prior + overlap)."""
    X, y_pu, _ = make_scar_data_unbalanced(rng, n=150, prior=0.3, separation=separation)
    est = ClassPriorEstimator().fit(X, y_pu).estimate()
    assert 0.2 * 0.3 <= est <= 2.0 * 0.3


# ── edge (auto-sigma fallback) ─────────────────────────────────


@pytest.mark.unit
def test_edge_auto_sigma_zero_variance_fallback(rng):
    """Zero-variance data falls back to sigma=1.0 instead of failing."""
    X = np.ones((50, 3))
    y = np.r_[np.ones(25, dtype=int), np.zeros(25, dtype=int)]
    est = ClassPriorEstimator().fit(X, y)
    assert est.sigma_ == pytest.approx(1.0)
    assert 0.0 <= est.estimate() <= 1.0


# ── determ ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_deterministic_estimate_across_runs(rng):
    X, y = _data(rng)
    e1 = ClassPriorEstimator(n_centers=50).fit(X, y)
    e2 = ClassPriorEstimator(n_centers=50).fit(X, y)
    assert e1.estimate() == pytest.approx(e2.estimate())
