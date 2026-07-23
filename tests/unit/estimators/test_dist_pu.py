# ruff: noqa: E402, N806

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.risk import DistPUClassifier


def _data(rng):
    p = rng.normal(1.0, 0.7, size=(20, 3)).astype(np.float32)
    u = rng.normal(0.0, 1.0, size=(40, 3)).astype(np.float32)
    X = np.vstack([p, u])
    y = np.r_[np.ones(len(p), dtype=int), np.zeros(len(u), dtype=int)]
    return X, y


# ── basic ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_basic_fit_predict_output_shapes(rng):
    X, y = _data(rng)
    model = DistPUClassifier(0.3, hidden_dim=8, epochs=2, random_state=3).fit(X, y)
    assert model.predict(X).shape == (len(X),)
    assert model.predict_proba(X).shape == (len(X), 2)
    assert np.isfinite(model.loss_history_).all()


# ── param validation ───────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("kwargs", [
    {"class_prior": 0.0},
    {"class_prior": 1.0},
    {"class_prior": -0.5},
])
def test_invalid_class_prior_raises(rng, kwargs):
    X, y = _data(rng)
    pi = kwargs["class_prior"]
    with pytest.raises(ValueError):
        DistPUClassifier(pi, hidden_dim=8, epochs=1).fit(X, y)


@pytest.mark.unit
def test_invalid_epochs_raises(rng):
    X, y = _data(rng)
    with pytest.raises(ValueError):
        DistPUClassifier(0.3, hidden_dim=8, epochs=0).fit(X, y)


# ── edge ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_edge_single_epoch(rng):
    X, y = _data(rng)
    model = DistPUClassifier(0.3, hidden_dim=4, epochs=1, random_state=0).fit(X, y)
    assert len(model.loss_history_) == 1


@pytest.mark.unit
def test_edge_zero_mixup_weight(rng):
    X, y = _data(rng)
    model = DistPUClassifier(0.3, hidden_dim=4, epochs=2, mixup_weight=0.0, random_state=0).fit(X, y)
    assert model.predict(X).shape == (len(X),)


# ── determ ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_deterministic_loss_with_seed(rng):
    X, y = _data(rng)
    m1 = DistPUClassifier(0.3, hidden_dim=8, epochs=3, random_state=42).fit(X, y)
    m2 = DistPUClassifier(0.3, hidden_dim=8, epochs=3, random_state=42).fit(X, y)
    np.testing.assert_allclose(m1.loss_history_, m2.loss_history_)
