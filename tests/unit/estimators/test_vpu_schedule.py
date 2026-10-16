"""VPU author-reset timing, independent budget axes and preserved defaults."""

import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402
from pu_toolbox.estimators.risk.vpu import VPUClassifier  # noqa: E402

pytestmark = pytest.mark.unit


def data():
    return np.random.RandomState(4).normal(size=(16, 3)).astype("float32"), np.r_[
        np.ones(6), np.zeros(10)
    ]


def model(**kwargs):
    params = dict(
        hidden_dim=4,
        depth=1,
        max_epochs=41,
        batch_size=8,
        random_state=0,
        device="cpu",
        iterations_per_epoch=1,
    )
    params.update(kwargs)
    return VPUClassifier(**params)


def test_basic_source_halves_before_one_based_epochs_20_and_40_and_recreates_adam(monkeypatch):
    adam = torch.optim.Adam
    optimizers, first_step_states = [], []

    def factory(*args, **kwargs):
        optimizer = adam(*args, **kwargs)
        optimizers.append(optimizer)
        step = optimizer.step

        def record(*args, **kwargs):
            if not optimizer.state:
                first_step_states.append(optimizer.param_groups[0]["lr"])
            return step(*args, **kwargs)

        optimizer.step = record
        return optimizer

    monkeypatch.setattr(torch.optim, "Adam", factory)
    fitted = model(optimizer_schedule="source_reset_halving").fit(*data())
    rates = fitted.history_["learning_rate"]
    assert rates == [3e-4] * 19 + [1.5e-4] * 20 + [7.5e-5] * 2
    assert fitted.history_["optimizer_generation"] == [0] * 19 + [1] * 20 + [2] * 2
    assert len(optimizers) == 3 and len({id(item) for item in optimizers}) == 3
    assert first_step_states == [3e-4, 1.5e-4, 7.5e-5]
    assert fitted.optimizer_resets_ == 2 and fitted.optimizer_steps_ == 41


def test_basic_constant_schedule_keeps_adam_and_adds_diagnostic_history():
    fitted = model().fit(*data())
    assert fitted.optimizer_resets_ == 0
    assert fitted.history_["learning_rate"] == [3e-4] * 41
    assert fitted.history_["optimizer_generation"] == [0] * 41
    assert fitted.optimizer_steps_ == 41


@pytest.mark.parametrize("view", ["os", "ts"])
def test_edge_fixed_iteration_budget_is_independent_of_pool_size(view):
    fitted = model(max_epochs=2, iterations_per_epoch=3).fit(*data(), os_or_ts=view)
    assert fitted.optimizer_steps_ == 6
    assert fitted.history_["optimizer_steps"] == [3, 6]
    assert fitted.n_loss_unlabeled_ == (16 if view == "ts" else 10)


@pytest.mark.parametrize("view,steps", [("os", 2), ("ts", 4)])
def test_basic_none_iteration_budget_preserves_data_derived_steps(view, steps):
    fitted = model(max_epochs=2, batch_size=5, iterations_per_epoch=None).fit(
        *data(), os_or_ts=view
    )
    assert fitted.optimizer_steps_ == 2 * steps


@pytest.mark.parametrize(
    "name,value",
    [
        ("optimizer_schedule", "step_lr"),
        ("optimizer_schedule", None),
        ("optimizer_schedule", []),
        ("iterations_per_epoch", 0),
        ("iterations_per_epoch", -1),
        ("iterations_per_epoch", True),
        ("iterations_per_epoch", 1.5),
    ],
)
def test_param_bad_schedule_or_step_budget_fails_and_invalidates_refit(name, value):
    fitted = model(max_epochs=1).fit(*data())
    fitted.set_params(**{name: value})
    with pytest.raises(ValueError):
        fitted.fit(*data())
    with pytest.raises(NotFittedError):
        fitted.predict(data()[0])


def test_edge_source_schedule_does_not_change_first_nineteen_epochs():
    constant = model(max_epochs=19).fit(*data())
    source = model(max_epochs=19, optimizer_schedule="source_reset_halving").fit(*data())
    np.testing.assert_array_equal(
        constant.decision_function(data()[0]), source.decision_function(data()[0])
    )
    assert source.optimizer_resets_ == 0 and source.history_ == constant.history_


def test_determ_clone_pickle_repeat_and_prior_do_not_change_schedule():
    first = clone(model(optimizer_schedule="source_reset_halving")).fit(*data())
    second = model(optimizer_schedule="source_reset_halving").fit(*data(), class_prior=0.7)
    assert first.history_ == second.history_
    np.testing.assert_array_equal(
        first.decision_function(data()[0]), second.decision_function(data()[0])
    )
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(
        first.decision_function(data()[0]), restored.decision_function(data()[0])
    )


def test_basic_machine_source_review_pins_defaults_without_admission():
    root = Path(__file__).resolve().parents[3]
    receipt = json.loads(
        (root / "docs/research/pu_survey/data/vpu_schedule_source_20261015.json").read_text(
            encoding="utf-8"
        )
    )
    signature = inspect.signature(VPUClassifier)
    for name, value in receipt["toolbox_defaults"].items():
        assert signature.parameters[name].default == value
    assert receipt["source"]["commit"] == "603bcc1e628795f57a5ac87e5b0b977273b7cf91"
    assert receipt["schedule_semantics"]["reset_adam_moments"] is True
    assert not receipt["formal_admission"] and receipt["candidate_protocol_ref"] is None
