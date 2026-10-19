# ruff: noqa: E402, N806

import json

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
    assert model.training_provenance_["implementation_path"] == "clean_room_single_stage_full_batch"
    assert model.training_provenance_["batch_size_parameter_effective"] is False
    assert model.training_provenance_["configured_batch_size_not_used"] == "128"
    assert model.training_provenance_["optimizer_steps_per_epoch"] == 1
    assert model.training_provenance_["optimizer_steps"] == 2
    assert model.training_provenance_["upstream_training_recipe_reproduced"] is False
    json.dumps(model.training_provenance_)


@pytest.mark.unit
def test_training_provenance_records_calibrated_view_without_changing_prior(rng):
    X, y = _data(rng)
    model = DistPUClassifier(0.3, hidden_dim=8, epochs=1, random_state=3).fit(
        X, y, os_or_ts="ts"
    )
    assert model.calibration_applied_ is True
    assert model.training_provenance_["run_view"] == "ts"
    assert model.training_provenance_["calibration_applied"] is True
    assert model.training_provenance_["class_prior_used"] == pytest.approx(0.3)
    assert model.training_provenance_["mixup_pool"] == "entire_training_matrix"


@pytest.mark.unit
def test_author_histogram_objective_is_explicit_opt_in_and_trains(rng):
    X, y = _data(rng)
    model = DistPUClassifier(
        0.3,
        hidden_dim=8,
        epochs=2,
        mixup_weight=0.0,
        distribution_loss="author_histogram",
        random_state=3,
    ).fit(X, y, os_or_ts="ts")
    assert model.training_provenance_["distribution_loss"] == "author_histogram"
    assert model.training_provenance_["upstream_training_recipe_reproduced"] is False
    assert np.isfinite(model.loss_history_).all()
    assert model.predict_proba(X).shape == (len(X), 2)


@pytest.mark.unit
def test_invalid_distribution_loss_mode_fails_before_training(rng):
    X, y = _data(rng)
    with pytest.raises(ValueError, match="distribution_loss must be"):
        DistPUClassifier(0.3, hidden_dim=8, epochs=1, distribution_loss="unknown").fit(X, y)


# ── param validation ───────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"class_prior": 0.0},
        {"class_prior": 1.0},
        {"class_prior": -0.5},
    ],
)
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
    model = DistPUClassifier(
        0.3,
        hidden_dim=4,
        epochs=2,
        mixup_weight=0.0,
        random_state=0,
    ).fit(X, y)
    assert model.predict(X).shape == (len(X),)
