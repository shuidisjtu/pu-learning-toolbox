# ruff: noqa: N806

import numpy as np
import pytest

from pu_toolbox.estimators.bias_aware import LBEClassifier, PUSBClassifier


def _data(rng):
    p = rng.normal(1.0, 0.7, size=(30, 3))
    u = rng.normal(0.0, 1.0, size=(80, 3))
    return np.vstack([p, u]), np.r_[np.ones(len(p), dtype=int), np.zeros(len(u), dtype=int)]


@pytest.mark.unit
@pytest.mark.parametrize("cls", [PUSBClassifier, LBEClassifier])
def test_bias_aware_estimators_expose_classifier_api(rng, cls):
    X, y = _data(rng)
    estimator = cls().fit(X, y)
    assert estimator.predict(X).shape == (len(X),)
    assert estimator.decision_function(X).shape == (len(X),)
    assert np.isfinite(estimator.predict_proba(X)).all()


@pytest.mark.unit
def test_lbe_propensity_is_bounded(rng):
    X, y = _data(rng)
    model = LBEClassifier(n_em_iter=3).fit(X, y)
    propensity = model.predict_label_proba(X)
    assert np.all((propensity >= 0.0) & (propensity <= 1.0))
