"""Issue 121: actual Adam LR, branch sizes and honest training provenance."""

import copy
import json
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox import GradPUClassifier  # noqa: E402
from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


def data():
    return np.random.RandomState(5).normal(size=(12, 3)).astype(np.float32), np.r_[
        np.ones(5, int), np.zeros(7, int)
    ]


def model(**kwargs):
    params = dict(hidden_dim=4, batch_size=4, max_epochs=3, random_state=3, device="cpu")
    params.update(kwargs)
    return GradPUClassifier(**params)


@pytest.mark.parametrize(
    "schedule,expected",
    [
        ("constant", [1.0, 1.0, 1.0, 1.0]),
        ("linear", [1.0, 0.7, 0.4, 0.1]),
        ("cosine", [1.0, 0.775, 0.325, 0.1]),
    ],
)
def test_basic_actual_adam_rates_not_only_logged_schedule(monkeypatch, schedule, expected):
    features, labels = data()
    seen = []
    step = torch.optim.Adam.step

    def capture(optimizer, *args, **kwargs):
        seen.append(optimizer.param_groups[0]["lr"])
        return step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", capture)
    estimator = model(max_epochs=4, learning_rate=0.002, lr_schedule=schedule).fit(features, labels)
    expected_lr = np.array(expected) * 0.002
    np.testing.assert_allclose(seen, np.repeat(expected_lr, 2), rtol=1e-14)
    np.testing.assert_allclose(estimator.history_["learning_rate"], expected_lr, rtol=1e-14)
    recipe = estimator.training_provenance_
    assert recipe["learning_rate"]["epoch_used_rates"] == estimator.history_["learning_rate"]
    assert recipe["learning_rate"]["actual_last_ratio"] == pytest.approx(expected[-1])
    assert recipe["batch"]["optimizer_steps"] == estimator.optimizer_steps_ == 8
    assert not recipe["learning_rate"]["curve_is_specified_by_paper"]
    assert not recipe["formal_admission"] and recipe["formal_candidate_pool"] is None
    json.dumps(recipe, allow_nan=False)


@pytest.mark.parametrize("view", ["os", "ts"])
@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_basic_batch_semantics_match_real_objective_calls_and_short_tail(monkeypatch, view, alpha):
    import pu_toolbox.estimators.deep.grad_pu as module

    features, labels = data()
    seen = []
    objective = module.gradpu_objective

    def capture(p, u, **kwargs):
        mixed = kwargs["interpolated_inputs"]
        seen.append((len(p), len(u), 0 if mixed is None else len(mixed)))
        return objective(p, u, **kwargs)

    monkeypatch.setattr(module, "gradpu_objective", capture)
    estimator = model(max_epochs=1, alpha=alpha).fit(features, labels, os_or_ts=view)
    risk_u = [8, 4] if view == "ts" else [4, 3]
    assert seen == [
        (4, risk_u[0], (max(4, risk_u[0]) if alpha else 0)),
        (1, risk_u[1], (max(1, risk_u[1]) if alpha else 0)),
    ]
    batch = estimator.training_provenance_["batch"]
    assert batch["steps_per_epoch"] == 2 and batch["optimizer_steps"] == 2
    assert batch["observed_max_rows"] == {
        "positive": 4,
        "native_unlabeled": 4,
        "risk_unlabeled": risk_u[0],
        "physical_P_plus_U": 8,
        "GP_interpolation": risk_u[0] if alpha else 0,
    }
    assert estimator.training_provenance_["calibration_applied"] is (view == "ts")
    assert not estimator.training_provenance_["TS_is_paper_experiment_replay"]
    assert not estimator.training_provenance_["assumptions"]["general_SAR_theoretical_guarantee"]


def test_edge_shorter_branch_cycles_and_steps_are_not_pooled_total_batch():
    features = np.random.RandomState(9).normal(size=(11, 3)).astype(np.float32)
    labels = np.r_[np.ones(2, int), np.zeros(9, int)]
    estimator = model(max_epochs=1).fit(features, labels, os_or_ts="ts")
    batch = estimator.training_provenance_["batch"]
    assert batch["steps_per_epoch"] == batch["optimizer_steps"] == 3
    assert batch["observed_max_rows"]["positive"] == 2
    assert batch["observed_max_rows"]["native_unlabeled"] == 4
    assert batch["observed_max_rows"]["risk_unlabeled"] == 6
    assert batch["observed_max_rows"]["GP_interpolation"] == 6


def test_edge_numpy_scalar_learning_rate_keeps_training_facts_strict_json():
    features, labels = data()
    fitted = model(
        max_epochs=1, learning_rate=np.float32(0.001), lr_final_ratio=np.float32(0.1)
    ).fit(features, labels)
    json.dumps(fitted.training_provenance_, allow_nan=False)


@pytest.mark.parametrize("schedule", ["constant", "linear", "cosine"])
def test_edge_single_epoch_keeps_initial_rate_not_false_terminal_decay(schedule):
    features, labels = data()
    fitted = model(max_epochs=1, lr_schedule=schedule).fit(features, labels)
    assert fitted.history_["learning_rate"] == [0.001]
    assert fitted.training_provenance_["learning_rate"]["actual_last_ratio"] == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr_schedule": "step"},
        {"lr_schedule": []},
        {"lr_final_ratio": 0},
        {"lr_final_ratio": 1.1},
        {"lr_final_ratio": float("nan")},
        {"lr_final_ratio": float("inf")},
        {"lr_final_ratio": True},
        {"lr_final_ratio": "0.1"},
        {"lr_schedule": "linear", "lr_final_ratio": np.nextafter(0.0, 1.0)},
    ],
)
def test_param_invalid_policy_invalidates_previous_fit(kwargs):
    features, labels = data()
    estimator = model().fit(features, labels)
    estimator.set_params(**kwargs)
    with pytest.raises(ValueError, match="lr_schedule|lr_final_ratio"):
        estimator.fit(features, labels)
    with pytest.raises(NotFittedError):
        estimator.predict(features)


def test_determ_clone_pickle_checkpoint_and_diagnostics_do_not_change_update_count(tmp_path):
    features, labels = data()
    template = model(lr_schedule="cosine")
    estimator = clone(template)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        estimator, features, labels, os_or_ts="ts"
    )
    other = clone(template).fit(features, labels, os_or_ts="ts")
    assert estimator.training_provenance_ == other.training_provenance_
    np.testing.assert_array_equal(
        estimator.decision_function(features), other.decision_function(features)
    )
    restored = pickle.loads(pickle.dumps(estimator))
    assert restored.training_provenance_ == copy.deepcopy(estimator.training_provenance_)
    np.testing.assert_array_equal(
        restored.decision_function(features), estimator.decision_function(features)
    )
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore().decision_function(features),
        estimator.decision_function(features),
    )
    assert len(trajectory.checkpoints) == 3 and estimator.optimizer_steps_ == 6
