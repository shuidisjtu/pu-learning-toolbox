"""CVIR input ownership, inference modes and pre-training parameter boundaries."""

# ruff: noqa: E402, N803, N806

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.risk.cvir import CVIRClassifier, select_cvir_negatives

pytestmark = pytest.mark.unit


def data(images=False):
    shape = (1, 4, 4) if images else (3,)
    X = np.random.RandomState(71).normal(size=(19, *shape)).astype(np.float32)
    return X, np.r_[np.ones(6, dtype=int), np.zeros(13, dtype=int)]


def model(**overrides):
    params = dict(
        unlabeled_positive_prior=0.3,
        hidden_dim=4,
        warm_start_epochs=1,
        max_epochs=2,
        batch_size=4,
        random_state=17,
        device="cpu",
    )
    params.update(overrides)
    return CVIRClassifier(**params)


@pytest.mark.parametrize("policy", ["resident", "cpu_stream"])
def test_basic_prediction_preserves_individual_module_modes_and_empty_input(policy):
    X, y = data()
    fitted = model(data_device_policy=policy).fit(X, y)
    modules = list(fitted.model_.modules())
    modules[0].training = True
    modules[-1].training = False
    before = [module.training for module in modules]
    first = fitted.decision_function(X)
    assert [module.training for module in modules] == before
    np.testing.assert_array_equal(first, fitted.decision_function(X))
    assert fitted.decision_function(np.empty((0, 3), dtype=np.float32)).shape == (0,)
    assert [module.training for module in modules] == before


@pytest.mark.parametrize("failure", ["eval", "forward"])
def test_edge_partial_mode_transition_or_forward_failure_restores_every_layer(failure):
    X, y = data()
    fitted = model().fit(X, y)
    modules = list(fitted.model_.modules())
    for index, module in enumerate(modules):
        module.training = index % 2 == 0
    before = [module.training for module in modules]

    def fail(*_args, **_kwargs):
        if failure == "eval":
            modules[-1].training = not modules[-1].training
        raise RuntimeError("injected CVIR failure")

    setattr(fitted.model_, failure, fail)
    original = X.copy()
    with pytest.raises(RuntimeError, match="injected CVIR failure"):
        fitted.decision_function(X)
    np.testing.assert_array_equal(X, original)
    assert [module.training for module in modules] == before


def test_edge_empty_prediction_does_not_touch_network():
    X, y = data()
    fitted = model().fit(X, y)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("empty inference touched network")

    fitted.model_.eval = forbidden
    assert fitted.decision_function(X[:0]).shape == (0,)


def test_basic_cnn_scores_use_eval_without_altering_batchnorm_dropout_or_buffers():
    X, y = data(images=True)
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.25),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )
    fitted = model(encoder=encoder).fit(X, y)
    modules = list(fitted.model_.modules())
    for index, module in enumerate(modules):
        module.training = index % 2 == 0
    modes = [module.training for module in modules]
    state = copy.deepcopy(fitted.model_.state_dict())
    first = fitted.decision_function(X)
    np.testing.assert_array_equal(first, fitted.decision_function(X))
    assert [module.training for module in modules] == modes
    for name, value in fitted.model_.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("images", [False, True])
def test_basic_determ_inplace_encoder_keeps_raw_inputs_and_matches_noninplace_training(images):
    X, y = data(images)
    template = torch.nn.Sequential(
        torch.nn.ReLU(), torch.nn.Flatten(), torch.nn.Linear(int(np.prod(X.shape[1:])), 4)
    )
    inplace = copy.deepcopy(template)
    inplace[0].inplace = True
    original = X.copy()
    reference = model(encoder=template).fit(X, y)
    actual = model(encoder=inplace).fit(X, y)
    np.testing.assert_array_equal(X, original)
    first = actual.decision_function(X)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(first, actual.decision_function(X))
    assert reference.history_ == actual.history_
    np.testing.assert_array_equal(
        reference.selected_negative_indices_, actual.selected_negative_indices_
    )
    for name, value in reference.model_.state_dict().items():
        torch.testing.assert_close(actual.model_.state_dict()[name], value, rtol=0, atol=0)
    np.testing.assert_array_equal(first, reference.decision_function(X))


def test_edge_mutation_then_scoring_failure_keeps_caller_input_unchanged():
    X, y = data()
    fitted = model().fit(X, y)
    original = X.copy()

    def fail(inputs):
        inputs.mul_(0)
        raise RuntimeError("injected mutating forward")

    fitted.model_.forward = fail
    with pytest.raises(RuntimeError, match="injected mutating forward"):
        fitted.decision_function(X)
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("value", [True, np.bool_(False), "0.001", [], np.nan, np.inf, 0, -1])
def test_param_invalid_learning_rate_rejected_before_rng_or_network(monkeypatch, value):
    X, y = data()
    state = torch.get_rng_state().clone()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid parameters reached model construction")

    monkeypatch.setattr(torch.nn, "Linear", forbidden)
    with pytest.raises(ValueError, match="learning_rate"):
        model(learning_rate=value).fit(X, y)
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)


def test_param_noncallable_callback_is_rejected_before_rng():
    state = torch.get_rng_state().clone()
    with pytest.raises(TypeError, match="epoch_callback"):
        model().fit(*data(), epoch_callback=3)
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)


@pytest.mark.parametrize("value", [True, np.bool_(True), "0.3", [], np.nan, np.inf, 0, 1])
def test_param_public_selector_requires_a_finite_real_fraction(value):
    with pytest.raises(ValueError, match="unlabeled_positive_prior"):
        select_cvir_negatives(np.arange(4.0), value)


@pytest.mark.parametrize("images", [False, True])
def test_param_complex_features_are_not_silently_cast_to_real(images):
    X, y = data(images)
    X = X.astype(np.complex64) + 1j
    kwargs = {"encoder": torch.nn.Sequential(torch.nn.Flatten())} if images else {}
    with pytest.raises(ValueError, match="real numeric"):
        model(**kwargs).fit(X, y)
