# ruff: noqa: N806

import numpy as np
import pytest

from pu_toolbox.estimators.bias_aware import LBEClassifier, PUSBClassifier


def _data(rng):
    p = rng.normal(1.0, 0.7, size=(30, 3))
    u = rng.normal(0.0, 1.0, size=(80, 3))
    return np.vstack([p, u]), np.r_[np.ones(len(p), dtype=int), np.zeros(len(u), dtype=int)]


# ── basic ──────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("cls", [PUSBClassifier, LBEClassifier])
def test_basic_fit_predict_output_shapes(rng, cls):
    X, y = _data(rng)
    estimator = cls().fit(X, y)
    assert estimator.predict(X).shape == (len(X),)
    assert estimator.decision_function(X).shape == (len(X),)
    assert np.isfinite(estimator.predict_proba(X)).all()


@pytest.mark.unit
def test_basic_lbe_propensity_is_bounded(rng):
    X, y = _data(rng)
    model = LBEClassifier(n_em_iter=3).fit(X, y)
    propensity = model.predict_label_proba(X)
    assert np.all((propensity >= 0.0) & (propensity <= 1.0))


# ── param validation ───────────────────────────────────────────


@pytest.mark.unit
def test_invalid_lbe_params_raises(rng):
    X, y = _data(rng)
    with pytest.raises(ValueError):
        LBEClassifier(n_em_iter=0).fit(X, y)
    with pytest.raises(ValueError):
        LBEClassifier(C=-1.0).fit(X, y)


@pytest.mark.unit
def test_invalid_pusb_params_raises(rng):
    X, y = _data(rng)
    with pytest.raises(ValueError):
        PUSBClassifier(threshold=0.0).fit(X, y)
    with pytest.raises(ValueError):
        PUSBClassifier(C=-1.0).fit(X, y)


@pytest.mark.unit
def test_invalid_lbe_class_prior_raises(rng):
    X, y = _data(rng)
    with pytest.raises(ValueError):
        LBEClassifier().fit(X, y, class_prior=0.0)
    with pytest.raises(ValueError):
        LBEClassifier().fit(X, y, class_prior=1.0)


# ── edge ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_edge_single_em_iteration(rng):
    X, y = _data(rng)
    model = LBEClassifier(n_em_iter=1).fit(X, y)
    assert model.predict(X).shape == (len(X),)


@pytest.mark.unit
def test_edge_pusb_all_positive_raises(rng):
    X = rng.normal(size=(30, 3))
    y = np.ones(30, dtype=int)
    with pytest.raises(Exception):
        PUSBClassifier().fit(X, y)


# ── determ ─────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("cls", [PUSBClassifier, LBEClassifier])
def test_deterministic_predictions_across_runs(rng, cls):
    X, y = _data(rng)
    m1 = cls().fit(X, y)
    m2 = cls().fit(X, y)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))
