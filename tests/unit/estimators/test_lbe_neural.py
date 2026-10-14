"""Neural LBE is separate, deterministic, OS-only and safe to refit/serialize."""

import copy
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox import NeuralLBEClassifier  # noqa: E402
from pu_toolbox.core.exceptions import NotFittedError, ValidationError  # noqa: E402
from pu_toolbox.estimators.deep import lbe_neural  # noqa: E402
from pu_toolbox.registry import get_algorithm, register_all_builtin_methods  # noqa: E402

pytestmark = pytest.mark.unit


def data():
    return np.random.RandomState(7).normal(size=(12, 3)), np.r_[np.ones(4), np.zeros(8)]


def model(**kwargs):
    params = dict(
        hidden_dim=4,
        pretrain_steps=1,
        n_em_iter=2,
        m_steps=2,
        batch_size=4,
        sample_fraction=0.5,
        device="cpu",
    )
    params.update(kwargs)
    return NeuralLBEClassifier(**params)


def test_basic_fit_clone_pickle_probabilities_and_registry_are_separate():
    x, labels = data()
    fitted = clone(model()).fit(x, labels)
    assert fitted.n_iter_ == 2 and fitted.optimizer_steps_ == 6
    assert fitted.pretraining_optimizer_steps_ == 2 and fitted.em_optimizer_steps_ == 4
    assert fitted.history_["optimizer_steps"] == [4, 6]
    assert fitted.training_view_ == "os" and not fitted.calibration_applied_
    np.testing.assert_allclose(fitted.predict_proba(x).sum(axis=1), 1)
    assert np.all(fitted.predict_label_proba(x) <= fitted.predict_proba(x)[:, 1])
    assert np.all(fitted._latent_positive_probability_[:4] == 1)
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(fitted.predict_label_proba(x), restored.predict_label_proba(x))
    register_all_builtin_methods()
    assert get_algorithm("lbe_neural") is NeuralLBEClassifier
    assert get_algorithm("neural_lbe") is NeuralLBEClassifier
    assert get_algorithm("lbe") is not NeuralLBEClassifier


@pytest.mark.parametrize(
    "name,value",
    [
        ("n_em_iter", 0),
        ("m_steps", 1.2),
        ("pretrain_steps", -1),
        ("hidden_dim", True),
        ("batch_size", 0),
        ("sample_fraction", 0),
        ("sample_fraction", 1.1),
        ("learning_rate", float("nan")),
        ("weight_decay", -1),
        ("random_state", True),
    ],
)
def test_param_invalid_settings_are_rejected(name, value):
    with pytest.raises(ValueError):
        model(**{name: value}).fit(*data())


@pytest.mark.parametrize(
    "method", ["predict", "predict_proba", "predict_label_proba", "decision_function"]
)
def test_param_prediction_requires_successful_fit(method):
    with pytest.raises(NotFittedError):
        getattr(model(), method)(data()[0])


def test_edge_failed_refit_invalidates_both_probability_paths():
    x, labels = data()
    fitted = model().fit(x, labels)
    fitted.set_params(n_em_iter=0)
    with pytest.raises(ValueError):
        fitted.fit(x, labels)
    for method in (fitted.predict_proba, fitted.predict_label_proba):
        with pytest.raises(NotFittedError):
            method(x)


def test_param_ts_weights_and_missing_pu_roles_are_refused():
    x, labels = data()
    with pytest.raises(NotImplementedError, match="TS"):
        model().fit(x, labels, os_or_ts="ts")
    with pytest.raises(NotImplementedError, match="sample_weight"):
        model().fit(x, labels, sample_weight=np.ones(len(x)))
    for bad in (np.zeros(len(x)), np.ones(len(x))):
        with pytest.raises((ValueError, ValidationError)):
            model().fit(x, bad)


def test_determ_seed_prior_unused_and_global_rng_preserved():
    x, labels = data()
    before = torch.random.get_rng_state().clone()
    first = model().fit(x, labels)
    torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
    second = model().fit(x, labels, class_prior=0.8)
    np.testing.assert_array_equal(first.decision_function(x), second.decision_function(x))
    assert first.history_ == second.history_


def test_edge_chunk_accumulation_preserves_mlp_update_objective():
    x, labels = data()
    first = model(batch_size=3, sample_fraction=1).fit(x, labels)
    second = model(batch_size=12, sample_fraction=1).fit(x, labels)
    np.testing.assert_allclose(
        first.decision_function(x), second.decision_function(x), rtol=1e-5, atol=1e-6
    )
    assert first.optimizer_steps_ == second.optimizer_steps_ == 6


def test_basic_estep_count_fixed_m_round_and_callback_budget(monkeypatch):
    calls = []
    original = lbe_neural.lbe_latent_positive_posterior

    def count(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(lbe_neural, "lbe_latent_positive_posterior", count)
    events = []
    fitted = model(m_steps=3).fit(
        *data(),
        epoch_callback=lambda epoch, item: events.append(
            (epoch, item.optimizer_steps_, item.checkpoint_stage_)
        ),
    )
    assert len(calls) == 3  # Once per E round plus the final diagnostic, not per M step.
    assert events == [(0, 5, "em"), (1, 8, "em")]
    assert fitted.history_["sample_size"] == [6, 6]


def test_edge_tiny_sample_empty_prediction_and_unmodified_network_templates():
    x, labels = data()
    network = torch.nn.Linear(3, 1)
    propensity = torch.nn.Linear(3, 1)
    before = copy.deepcopy((network.state_dict(), propensity.state_dict()))
    fitted = model(
        model=network, propensity_model=propensity, sample_fraction=0.01, pretrain_steps=0
    ).fit(x[:3], np.array([1, 1, 0]))
    assert fitted.history_["sample_size"] == [1, 1]
    assert fitted.predict_proba(np.empty((0, 3))).shape == (0, 2)
    for actual, expected in zip(
        (network.state_dict(), propensity.state_dict()), before, strict=True
    ):
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_param_bad_network_outputs_and_float_overflow_are_refused():
    x, labels = data()
    with pytest.raises(ValueError, match="one raw logit"):
        model(model=torch.nn.Linear(3, 2)).fit(x, labels)
    with pytest.raises(ValueError, match="float32"):
        model().fit(np.full_like(x, 1e100), labels)


def test_basic_source_receipt_matches_constructor_and_optimizer_budget():
    root = Path(__file__).resolve().parents[3]
    receipt = json.loads(
        (root / "docs/research/pu_survey/data/lbe_neural_engineering_20261014.json").read_text(
            encoding="utf-8"
        )
    )
    signature = inspect.signature(NeuralLBEClassifier)
    for name, value in receipt["constructor_defaults"].items():
        assert signature.parameters[name].default == value
    defaults = receipt["constructor_defaults"]
    budget = receipt["default_schedule"]
    assert (
        budget["total_optimizer_steps"]
        == 2 * defaults["pretrain_steps"] + defaults["n_em_iter"] * defaults["m_steps"]
        == 840
    )
    assert budget["checkpoint_epoch_count"] == NeuralLBEClassifier().checkpoint_epoch_count == 20
    assert not receipt["formal_admission"] and receipt["protocol_binding"] is None
    assert not receipt["calibration"]["calibration_applied"]


@pytest.mark.parametrize("prior", [0, 1, float("nan"), True, [0.5]])
def test_param_unused_prior_still_requires_valid_scalar(prior):
    with pytest.raises(ValueError, match="class_prior"):
        model().fit(*data(), class_prior=prior)


def test_basic_ledger_view_gate_and_public_pipeline_access():
    from pu_toolbox.experiment.method_ledger import load_ledger
    from pu_toolbox.experiment.training_views import resolve_training_view
    from pu_toolbox.workflows import PUPipeline

    ledger = load_ledger()
    assert (
        resolve_training_view(
            ledger, "lbe_neural", None, is_oracle=False, estimator_class=NeuralLBEClassifier
        )
        == "os"
    )
    with pytest.raises(ValueError, match="calibration_hooked=false"):
        resolve_training_view(
            ledger, "lbe_neural", "ts", is_oracle=False, estimator_class=NeuralLBEClassifier
        )
    pipeline = PUPipeline(
        classifier="lbe_neural",
        prior_estimator=None,
        cv=2,
        device="cpu",
        classifier_params={
            "hidden_dim": 4,
            "pretrain_steps": 1,
            "n_em_iter": 2,
            "m_steps": 2,
            "batch_size": 4,
        },
    )
    report = pipeline.fit_evaluate(*data())
    assert report.final_model.predict(data()[0]).shape == (12,)


def test_determ_cpu_fit_does_not_seed_unrelated_cuda_devices(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU fit must not seed every CUDA generator")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    model(device="cpu").fit(*data())
