"""Inference restores per-module modes, including mixed states and exceptions."""

import copy
import pickle
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import (  # noqa: E402
    GenPUClassifier,
    GradPUClassifier,
    HolisticPUClassifier,
    LaGAMClassifier,
    NeuralLBEClassifier,
    PULDAClassifier,
    PULNSClassifier,
    RobustPUClassifier,
    SplitPUClassifier,
    VPUClassifier,
)
from pu_toolbox.estimators.deep.tedn import TEDnClassifier  # noqa: E402

pytestmark = pytest.mark.unit
METHODS = (
    "pulda",
    "gradpu",
    "robust_pu",
    "split_pu",
    "genpu",
    "holistic_pu",
    "pulns",
    "lagam",
    "tedn",
    "vpu",
    "lbe_neural",
)


def fitted_model(method):
    features = np.random.RandomState(5).normal(size=(12, 3)).astype(np.float32)
    labels = np.r_[np.ones(4, int), np.zeros(8, int)]
    torch.manual_seed(11)
    encoder = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.Dropout(0.2),
        torch.nn.BatchNorm1d(3) if method != "gradpu" else torch.nn.Identity(),
    )
    kwargs = dict(encoder=encoder, hidden_dim=4, random_state=3, device="cpu")
    if method == "lagam_resident":
        kwargs["encoder"] = None
    if method == "pulda":
        model = PULDAClassifier(
            class_prior=0.4,
            warmup_epochs=1,
            pu_epochs=1,
            positive_batch_size=2,
            unlabeled_batch_size=4,
            **kwargs,
        )
    elif method == "robust_pu":
        model = RobustPUClassifier(
            class_prior=0.4,
            pretrain_epochs=1,
            episodes=1,
            inner_epochs=1,
            batch_size=4,
            **kwargs,
        )
    elif method == "split_pu":
        model = SplitPUClassifier(
            class_prior=0.4,
            teacher_epochs=1,
            split_epochs=1,
            student_epochs=1,
            rounds=1,
            batch_size=4,
            encoder_feature_layer="0",
            **kwargs,
        )
    elif method == "genpu":
        model = GenPUClassifier(
            class_prior=0.4,
            latent_dim=2,
            max_epochs=1,
            classifier_epochs=1,
            batch_size=4,
            **kwargs,
        )
    elif method == "holistic_pu":
        model = HolisticPUClassifier(warmup_epochs=3, max_epochs=1, batch_size=4, **kwargs)
    elif method in {"pulns", "lagam", "lagam_resident"}:
        if method == "pulns":
            model = PULNSClassifier(
                pretrain_epochs=1, episodes=1, classifier_epochs=1, batch_size=4, **kwargs
            )
        else:
            model = LaGAMClassifier(
                warmup_epochs=1,
                max_epochs=2,
                batch_size=4,
                support_batch_size=4,
                num_clusters=2,
                **kwargs,
            )
        support = np.random.RandomState(19).normal(size=(8, 3)).astype(np.float32)
        return model.fit(features, labels, support_data=(support, np.tile([0, 1], 4))), features
    elif method == "vpu":
        model = VPUClassifier(max_epochs=1, batch_size=4, iterations_per_epoch=1, **kwargs)
    elif method == "lbe_neural":
        model = NeuralLBEClassifier(
            pretrain_steps=1, n_em_iter=1, m_steps=1, sample_fraction=0.5, batch_size=4, **kwargs
        )
    elif method == "tedn":
        import pu_toolbox.estimators.deep.tedn as implementation

        original = implementation.best_bin_estimate
        model = TEDnClassifier(
            input_sampling="ts", warm_start_epochs=1, max_epochs=1, batch_size=4, **kwargs
        )
        # Isolate inference-mode behavior from small-sample mixture estimation.
        with patch.object(
            implementation,
            "best_bin_estimate",
            lambda *args, **options: replace(original(*args, **options), alpha_u=0.2),
        ):
            return model.fit(features, labels), features
    else:
        model = GradPUClassifier(max_epochs=1, batch_size=4, **kwargs)
    return model.fit(features, labels), features


def set_mixed_modes(model, root_training):
    model.model_.train(root_training)
    for index, layer in enumerate(model.model_.modules()):
        if index % 2:
            layer.training = not root_training
    return [layer.training for layer in model.model_.modules()]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("root_training", [True, False])
def test_basic_prediction_is_eval_only_and_restores_all_children(method, root_training):
    model, features = fitted_model(method)
    expected = model.decision_function(features)
    before = set_mixed_modes(model, root_training)
    weights = copy.deepcopy(model.model_.state_dict())
    requires_grad = [p.requires_grad for p in model.model_.parameters()]
    rng = torch.get_rng_state().clone()
    calls = []
    handle = model.model_.register_forward_pre_hook(
        lambda module, args: calls.append([layer.training for layer in module.modules()])
    )
    try:
        actual = model.decision_function(features)
    finally:
        handle.remove()
    np.testing.assert_array_equal(actual, expected)
    assert calls and all(not any(flags) for flags in calls)
    assert [layer.training for layer in model.model_.modules()] == before
    assert [p.requires_grad for p in model.model_.parameters()] == requires_grad
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in model.model_.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("failure_call", [1, 2])
def test_edge_forward_failure_restores_modes_and_propagates_same_exception(method, failure_call):
    model, features = fitted_model(method)
    before = set_mixed_modes(model, True)
    sentinel = RuntimeError("injected inference failure")
    calls = []

    def fail(module, args):
        calls.append(len(args[0]))
        if len(calls) == failure_call:
            raise sentinel

    handle = model.model_.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError) as error:
            model.decision_function(features)
        assert error.value is sentinel
    finally:
        handle.remove()
    assert len(calls) == failure_call
    assert [layer.training for layer in model.model_.modules()] == before


@pytest.mark.parametrize("method", METHODS)
def test_edge_eval_transition_failure_also_restores_partial_mode_changes(method, monkeypatch):
    model, features = fitted_model(method)
    before = set_mixed_modes(model, True)
    dropout = next(layer for layer in model.model_.modules() if isinstance(layer, torch.nn.Dropout))
    original = dropout.train
    sentinel = RuntimeError("injected eval transition failure")

    def fail(mode=True):
        original(mode)
        raise sentinel

    monkeypatch.setattr(dropout, "train", fail)
    with pytest.raises(RuntimeError) as error:
        model.decision_function(features)
    assert error.value is sentinel
    assert [layer.training for layer in model.model_.modules()] == before


@pytest.mark.parametrize("method", METHODS)
def test_param_empty_or_invalid_prediction_keeps_modes_unchanged(method):
    model, features = fitted_model(method)
    before = set_mixed_modes(model, False)
    assert model.decision_function(features[:0]).shape == (0,)
    with pytest.raises(ValueError, match="shape|feature"):
        model.decision_function(features[:, :2])
    assert [layer.training for layer in model.model_.modules()] == before


@pytest.mark.parametrize("method", METHODS)
def test_determ_pickle_retains_mixed_modes_and_predictions(method):
    model, features = fitted_model(method)
    before = set_mixed_modes(model, True)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        restored.decision_function(features), model.decision_function(features)
    )
    assert [layer.training for layer in restored.model_.modules()] == before
    assert [layer.training for layer in model.model_.modules()] == before


@pytest.mark.parametrize("fault", ["none", "forward", "eval"])
def test_edge_lagam_resident_scoring_restores_modes_on_all_exits(monkeypatch, fault):
    model, features = fitted_model("lagam_resident")
    assert model.data_device_policy_ == "resident"
    expected = model.decision_function(features)
    before = set_mixed_modes(model, True)
    sentinel = RuntimeError("resident inference failure")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise sentinel

    handle = None
    if fault == "forward":
        handle = model.model_.register_forward_pre_hook(fail)
    elif fault == "eval":
        monkeypatch.setattr(model.model_[1], "train", fail)
    try:
        if fault == "none":
            np.testing.assert_array_equal(model.decision_function(features), expected)
        else:
            with pytest.raises(RuntimeError) as error:
                model.decision_function(features)
            assert error.value is sentinel and calls == [1]
    finally:
        if handle is not None:
            handle.remove()
    assert [layer.training for layer in model.model_.modules()] == before
    assert model.decision_function(features[:0]).shape == (0,)
    assert [layer.training for layer in model.model_.modules()] == before


@pytest.mark.parametrize("fault", ["none", "forward", "eval"])
def test_edge_neural_lbe_label_probability_restores_both_networks(monkeypatch, fault):
    model, features = fitted_model("lbe_neural")
    expected = model.predict_label_proba(features)
    set_mixed_modes(model, True)
    model.propensity_model_.train(False)
    for index, layer in enumerate(model.propensity_model_.modules()):
        if index % 2:
            layer.training = True
    before = [
        (layer, layer.training)
        for network in (model.model_, model.propensity_model_)
        for layer in network.modules()
    ]
    sentinel = RuntimeError("propensity inference failure")

    def fail(*args, **kwargs):
        raise sentinel

    handle = None
    if fault == "forward":
        handle = model.propensity_model_.register_forward_pre_hook(fail)
    elif fault == "eval":
        dropout = next(
            layer
            for layer in model.propensity_model_.modules()
            if isinstance(layer, torch.nn.Dropout)
        )
        monkeypatch.setattr(dropout, "train", fail)
    try:
        if fault == "none":
            np.testing.assert_array_equal(model.predict_label_proba(features), expected)
        else:
            with pytest.raises(RuntimeError) as error:
                model.predict_label_proba(features)
            assert error.value is sentinel
    finally:
        if handle is not None:
            handle.remove()
    assert all(layer.training == mode for layer, mode in before)
    assert model.predict_label_proba(features[:0]).shape == (0,)
    assert all(layer.training == mode for layer, mode in before)
