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
    assert model.training_provenance_["configured_batch_size_not_used"] == "256"
    assert model.training_provenance_["optimizer_steps_per_epoch"] == 1
    assert model.training_provenance_["optimizer_steps"] == 2
    assert model.training_provenance_["upstream_training_recipe_reproduced"] is False
    json.dumps(model.training_provenance_)


@pytest.mark.unit
def test_training_provenance_records_calibrated_view_without_changing_prior(rng):
    X, y = _data(rng)
    model = DistPUClassifier(0.3, hidden_dim=8, epochs=1, random_state=3).fit(X, y, os_or_ts="ts")
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
def test_author_staged_recipe_replays_two_stage_minibatch_and_scheduler_cadence(rng):
    X, y = _data(rng)
    checkpoints = []

    def callback(epoch, fitted):
        checkpoints.append(
            (
                epoch,
                fitted.checkpoint_stage_,
                fitted.checkpoint_stage_epoch_,
                fitted.optimizer_steps_,
            )
        )

    model = DistPUClassifier(
        0.3,
        hidden_dim=8,
        batch_size=8,
        distribution_loss="author_histogram",
        training_recipe="author_staged",
        warmup_epochs=2,
        pu_epochs=3,
        random_state=11,
    ).fit(X, y, epoch_callback=callback)

    batches_per_epoch = int(np.ceil(len(X) / 8))
    assert model.epochs_completed_ == 5
    assert model.optimizer_steps_ == batches_per_epoch * 5
    assert [item[:3] for item in checkpoints] == [
        (1, "warmup", 1),
        (2, "warmup", 2),
        (3, "dist_pu", 1),
        (4, "dist_pu", 2),
        (5, "dist_pu", 3),
    ]
    assert model.training_provenance_["scheduler_step_counts"] == {
        "warmup": 2,
        "dist_pu": 3 * batches_per_epoch,
    }
    assert model.training_provenance_["expected_scheduler_step_counts"] == {
        "warmup": 2,
        "dist_pu": 3 * batches_per_epoch,
    }
    assert model.training_provenance_["batch_size_parameter_effective"] is True
    assert model.training_provenance_["upstream_training_recipe_reproduced"] is False
    assert model.training_provenance_["formal_admission"] is False
    assert model.training_provenance_["data_device_policy"].startswith("CPU-resident")
    assert model.pseudo_labels_.shape == (len(X),)
    assert torch.isfinite(model.pseudo_labels_).all()
    assert np.isfinite(model.loss_history_).all()
    json.dumps(model.training_provenance_)
    assert model.predict_proba(X).shape == (len(X), 2)


@pytest.mark.unit
def test_author_staged_checkpoints_preserve_stage_identity(tmp_path, rng):
    from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer

    X, y = _data(rng)
    estimator = DistPUClassifier(
        0.3,
        hidden_dim=8,
        batch_size=8,
        distribution_loss="author_histogram",
        training_recipe="author_staged",
        warmup_epochs=1,
        pu_epochs=2,
        random_state=3,
    )
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(estimator, X, y)
    assert len(trajectory.checkpoints) == estimator.checkpoint_epoch_count == 3
    assert [row.training_context["stage"] for row in trajectory.checkpoints] == [
        "warmup",
        "dist_pu",
        "dist_pu",
    ]
    assert [row.training_context["stage_epoch"] for row in trajectory.checkpoints] == [1, 1, 2]
    assert [row.epoch_label for row in trajectory.checkpoints] == [1, 2, 3]


@pytest.mark.unit
def test_author_staged_ts_calibration_changes_only_marginal_role_objective(rng):
    X, y = _data(rng)
    common = dict(
        hidden_dim=8,
        batch_size=8,
        distribution_loss="author_histogram",
        training_recipe="author_staged",
        warmup_epochs=1,
        pu_epochs=1,
        random_state=8,
    )
    os_model = DistPUClassifier(0.3, **common).fit(X, y, os_or_ts="os")
    ts_model = DistPUClassifier(0.3, **common).fit(X, y, os_or_ts="ts")
    assert os_model.training_provenance_["calibration_applied"] is False
    assert ts_model.training_provenance_["calibration_applied"] is True
    assert (
        os_model.training_provenance_["mixup_pool"] == ts_model.training_provenance_["mixup_pool"]
    )
    assert (
        os_model.training_provenance_["class_prior_used"]
        == ts_model.training_provenance_["class_prior_used"]
    )
    assert not np.array_equal(os_model._decision_function(X), ts_model._decision_function(X))


@pytest.mark.unit
def test_determ_author_staged_same_seed_reproduces_cpu_training(rng):
    X, y = _data(rng)
    params = dict(
        hidden_dim=8,
        batch_size=8,
        distribution_loss="author_histogram",
        training_recipe="author_staged",
        warmup_epochs=1,
        pu_epochs=2,
        random_state=18,
    )
    first = DistPUClassifier(0.3, **params).fit(X, y)
    second = DistPUClassifier(0.3, **params).fit(X, y)
    np.testing.assert_array_equal(first._decision_function(X), second._decision_function(X))
    assert first.loss_history_ == second.loss_history_


@pytest.mark.unit
def test_author_staged_recipe_requires_author_histogram_loss(rng):
    X, y = _data(rng)
    with pytest.raises(ValueError, match="requires distribution_loss='author_histogram'"):
        DistPUClassifier(
            0.3,
            hidden_dim=8,
            training_recipe="author_staged",
            warmup_epochs=1,
            pu_epochs=1,
        ).fit(X, y)


@pytest.mark.unit
def test_author_staged_interruption_invalidates_previous_fit(rng):
    from pu_toolbox.core.exceptions import NotFittedError

    X, y = _data(rng)
    model = DistPUClassifier(
        0.3,
        hidden_dim=8,
        batch_size=8,
        distribution_loss="author_histogram",
        training_recipe="author_staged",
        warmup_epochs=1,
        pu_epochs=1,
        random_state=4,
    ).fit(X, y)

    def interrupt(*_args):
        raise KeyboardInterrupt("stop staged test fit")

    with pytest.raises(KeyboardInterrupt, match="stop staged test fit"):
        model.fit(X, y, epoch_callback=interrupt)
    with pytest.raises(NotFittedError):
        model.predict(X[:1])


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
