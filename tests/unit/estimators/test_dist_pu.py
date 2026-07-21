# ruff: noqa: E402, N806

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.risk import DistPUClassifier


@pytest.mark.unit
def test_dist_pu_fit_predict_and_alignment(rng):
    p = rng.normal(1.0, 0.7, size=(20, 3)).astype(np.float32)
    u = rng.normal(0.0, 1.0, size=(40, 3)).astype(np.float32)
    X = np.vstack([p, u])
    y = np.r_[np.ones(len(p), dtype=int), np.zeros(len(u), dtype=int)]
    model = DistPUClassifier(0.3, hidden_dim=8, epochs=2, random_state=3).fit(X, y)
    assert model.predict(X).shape == (len(X),)
    assert model.predict_proba(X).shape == (len(X), 2)
    assert np.isfinite(model.loss_history_).all()
