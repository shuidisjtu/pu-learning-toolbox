"""PAN inference is observational, including mixed child train/eval modes."""

# Match public feature naming.
# ruff: noqa: N806

import copy
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import PANClassifier  # noqa: E402

pytestmark = pytest.mark.unit
APIS = ("decision_function", "predict", "predict_proba", "score_samples")


def problem(images):
    rng = np.random.RandomState(5)
    shape = (24, 3, 4, 4) if images else (24, 3)
    X = rng.normal(size=shape).astype("float32")
    y = np.r_[np.ones(8, int), np.zeros(16, int)]
    torch.manual_seed(11)
    if images:
        params = dict(
            encoder=torch.nn.Sequential(
                torch.nn.Conv2d(3, 3, 1),
                torch.nn.BatchNorm2d(3),
                torch.nn.Dropout2d(0.3),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
            )
        )
    else:
        params = dict(
            model=torch.nn.Sequential(
                torch.nn.Linear(3, 4),
                torch.nn.BatchNorm1d(4),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(4, 1),
            )
        )
    fitted = PANClassifier(**params, max_epochs=1, batch_size=8, random_state=3, device="cpu").fit(
        X, y
    )
    return fitted, X


def mixed_modes(fitted, parent_training):
    fitted.model_.train(parent_training)
    for layer in fitted.model_.modules():
        if isinstance(
            layer, torch.nn.modules.batchnorm._BatchNorm | torch.nn.Dropout | torch.nn.Dropout2d
        ):
            layer.train(not parent_training)
    fitted.discriminator_.train()


def flags(network):
    return [layer.training for layer in network.modules()]


def assert_state(network, state):
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("images", [False, True])
@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("parent_training", [False, True])
def test_basic_determ_prediction_preserves_every_layer_mode_weights_buffers_and_rng(
    images, api, parent_training
):
    fitted, X = problem(images)
    expected = getattr(fitted, api)(X)
    mixed_modes(fitted, parent_training)
    before = flags(fitted.model_), flags(fitted.discriminator_)
    state = copy.deepcopy(fitted.model_.state_dict())
    d_state = copy.deepcopy(fitted.discriminator_.state_dict())
    rng = torch.get_rng_state().clone()
    for _ in range(2):
        np.testing.assert_array_equal(getattr(fitted, api)(X), expected)
        assert (flags(fitted.model_), flags(fitted.discriminator_)) == before
        assert_state(fitted.model_, state)
        assert_state(fitted.discriminator_, d_state)
        assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("images", [False, True])
@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_edge_forward_failure_restores_mixed_modes_and_preserves_exception(
    monkeypatch, images, api, failure_type
):
    fitted, X = problem(images)
    mixed_modes(fitted, True)
    before = flags(fitted.model_), flags(fitted.discriminator_)
    failure = failure_type("PAN inference interrupted")
    state = copy.deepcopy(fitted.model_.state_dict())
    original = fitted.model_.forward
    calls = []

    def interrupted(batch):
        calls.append(1)
        if len(calls) == 2:
            raise failure
        return original(batch)

    monkeypatch.setattr(fitted.model_, "forward", interrupted)
    with pytest.raises(failure_type) as caught:
        getattr(fitted, api)(X)
    assert caught.value is failure and len(calls) == 2
    assert (flags(fitted.model_), flags(fitted.discriminator_)) == before
    assert_state(fitted.model_, state)


@pytest.mark.parametrize("images", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_param_empty_or_invalid_shape_does_not_mutate_modes(images, empty):
    fitted, X = problem(images)
    mixed_modes(fitted, False)
    before = flags(fitted.model_)
    if empty:
        assert fitted.decision_function(X[:0]).shape == (0,)
    else:
        with pytest.raises(ValueError, match="input shape"):
            fitted.decision_function(X[:, :2])
    assert flags(fitted.model_) == before


@pytest.mark.parametrize("images", [False, True])
def test_determ_pickle_preserves_mixed_modes_before_and_after_prediction(images):
    fitted, X = problem(images)
    mixed_modes(fitted, True)
    restored = pickle.loads(pickle.dumps(fitted))
    before = flags(restored.model_)
    restored.predict_proba(X)
    assert flags(restored.model_) == before == flags(fitted.model_)
