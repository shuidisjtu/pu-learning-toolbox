"""Reject invalid Self-PU knobs before seeding, allocating or updating models."""

# ruff: noqa: N803, N806

import numpy as np
import pytest

from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.estimators.deep.self_pu import SelfPUClassifier

pytestmark = pytest.mark.unit
REAL_PARAMETERS = (
    "class_prior",
    "max_trust_ratio",
    "pace_1",
    "pace_2",
    "meta_step_size",
    "reweight_gamma",
    "distillation_alpha",
    "ema_decay",
    "student_loss_weight",
    "teacher_loss_weight",
    "learning_rate",
    "weight_decay",
    "threshold",
)
INTEGER_PARAMETERS = (
    "hidden_dim",
    "max_epochs",
    "batch_size",
    "warmup_epochs",
    "self_paced_start",
    "self_paced_end",
    "distill_start",
)


def data():
    rng = np.random.RandomState(37)
    features = rng.normal(size=(12, 3)).astype(np.float32)
    labels = np.r_[np.ones(4, dtype=int), np.zeros(8, dtype=int)]
    validation = rng.normal(size=(6, 3)).astype(np.float32)
    return features, labels, (validation, np.array([0, 1, 0, 1, 0, 1]))


def estimator():
    return SelfPUClassifier(
        0.4,
        hidden_dim=4,
        max_epochs=2,
        warmup_epochs=0,
        self_paced_start=0,
        self_paced_end=1,
        distill_start=1,
        batch_size=4,
        random_state=5,
        device="cpu",
    )


def assert_early_rejection(monkeypatch, changes, match):
    torch = pytest.importorskip("torch")
    features, labels, validation = data()
    model = estimator().set_params(**changes)
    torch_before, numpy_before = torch.get_rng_state().clone(), np.random.get_state()

    def fail(*_args):
        raise AssertionError("invalid parameters must not construct a model")

    monkeypatch.setattr(model, "_make_model", fail)
    with pytest.raises(ValueError, match=match):
        model.fit(features, labels, validation_data=validation)

    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0] and numpy_after[2:] == numpy_before[2:]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert model._is_fitted is False


@pytest.mark.parametrize("name", REAL_PARAMETERS)
@pytest.mark.parametrize(
    "value",
    [np.nan, np.inf, -np.inf, True, np.bool_(False), "0.1", None, np.array(0.1), 1 + 0j, 10**400],
    ids=[
        "nan",
        "inf",
        "negative_inf",
        "bool",
        "np_bool",
        "string",
        "none",
        "array",
        "complex",
        "overflow",
    ],
)
def test_param_invalid_real_values_fail_before_rng_or_model(monkeypatch, name, value):
    assert_early_rejection(monkeypatch, {name: value}, name)


@pytest.mark.parametrize("name", INTEGER_PARAMETERS)
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(False), 2.0, "2", np.nan, np.inf, None, np.array(2), -1],
    ids=["bool", "np_bool", "float", "string", "nan", "inf", "none", "array", "negative"],
)
def test_param_noninteger_or_negative_budgets_fail_before_rng(monkeypatch, name, value):
    assert_early_rejection(monkeypatch, {name: value}, name)


@pytest.mark.parametrize(
    "name,value",
    [
        ("class_prior", 0),
        ("class_prior", 1),
        ("max_trust_ratio", 0.51),
        ("pace_1", -0.1),
        ("pace_2", 0.51),
        ("reweight_gamma", 1.01),
        ("ema_decay", 1),
        ("threshold", 0),
        ("threshold", 1),
        ("meta_step_size", 0),
        ("learning_rate", 0),
        ("weight_decay", -0.1),
        ("distillation_alpha", -0.1),
        ("student_loss_weight", -0.1),
        ("teacher_loss_weight", -0.1),
        ("hidden_dim", 0),
        ("max_epochs", 0),
        ("batch_size", 0),
    ],
)
def test_edge_out_of_range_parameters_are_named_before_training(monkeypatch, name, value):
    assert_early_rejection(monkeypatch, {name: value}, name)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"warmup_epochs": 1}, "Require"),
        ({"self_paced_end": 3}, "Require"),
        ({"distill_start": 3}, "distill_start"),
        ({"self_paced_start": 1, "distill_start": 0}, "distill_start"),
        ({"require_validation": "False"}, "require_validation"),
        ({"require_validation": 1}, "require_validation"),
        ({"random_state": True}, "random_state"),
        ({"random_state": 2.0}, "random_state"),
        ({"random_state": -1}, "random_state"),
        ({"random_state": 2**32}, "random_state"),
    ],
)
def test_param_schedule_boolean_and_seed_contracts_are_checked_early(monkeypatch, changes, match):
    assert_early_rejection(monkeypatch, changes, match)


@pytest.mark.parametrize(
    "name", ["student_loss_weight", "teacher_loss_weight", "distillation_alpha"]
)
def test_edge_nan_loss_weight_refit_cannot_leave_a_fitted_nan_predictor(name):
    pytest.importorskip("torch")
    features, labels, validation = data()
    model = estimator().fit(features, labels, validation_data=validation)
    before = model.predict_proba(features)
    model.set_params(**{name: np.nan})
    with pytest.raises(ValueError, match=name):
        model.fit(features, labels, validation_data=validation)
    with pytest.raises(NotFittedError):
        model.predict_proba(features)
    model.set_params(**{name: 10.0 if name == "distillation_alpha" else 1.0})
    model.fit(features, labels, validation_data=validation)
    np.testing.assert_array_equal(model.predict_proba(features), before)


def test_basic_determ_numpy_scalars_match_equivalent_python_parameters():
    pytest.importorskip("torch")
    features, labels, validation = data()
    reference, numpy_model = estimator(), estimator()
    python_parameters = reference.get_params()
    numpy_model.set_params(
        **{name: np.float64(python_parameters[name]) for name in REAL_PARAMETERS},
        **{name: np.int64(python_parameters[name]) for name in INTEGER_PARAMETERS},
        require_validation=np.bool_(True),
        random_state=np.int64(python_parameters["random_state"]),
    )
    reference.fit(features, labels, validation_data=validation)
    numpy_model.fit(features, labels, validation_data=validation)
    np.testing.assert_array_equal(
        numpy_model.predict_proba(features), reference.predict_proba(features)
    )
    assert numpy_model.history_ == reference.history_


@pytest.mark.parametrize(
    "changes",
    [
        {
            "student_loss_weight": 0.0,
            "teacher_loss_weight": 0.0,
            "distillation_alpha": 0.0,
            "weight_decay": 0.0,
            "max_trust_ratio": 0.0,
            "pace_1": 0.0,
            "pace_2": 0.0,
            "reweight_gamma": 1.0,
            "ema_decay": 0.0,
            "random_state": 0,
        },
        {
            "max_trust_ratio": 0.5,
            "pace_1": 0.5,
            "pace_2": 0.5,
            "reweight_gamma": 0.0,
            "ema_decay": np.nextafter(1.0, 0.0),
            "random_state": 2**32 - 1,
        },
    ],
)
def test_basic_edge_valid_closed_boundaries_still_train_finite_models(changes):
    pytest.importorskip("torch")
    features, labels, validation = data()
    model = estimator().set_params(**changes).fit(features, labels, validation_data=validation)
    assert np.isfinite(model.predict_proba(features)).all()
    assert np.isfinite(model.history_["train_risk"]).all()
