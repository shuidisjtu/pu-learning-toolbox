"""Extra candidate lifecycle guards without changing objectives or support budgets."""

import inspect
import pickle
from dataclasses import replace

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox import (  # noqa: E402
    GenPUClassifier,
    HolisticPUClassifier,
    LaGAMClassifier,
    PULNSClassifier,
)
from pu_toolbox.core.base import BasePUClassifier  # noqa: E402
from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402
from pu_toolbox.estimators.deep.tedn import TEDnClassifier  # noqa: E402

pytestmark = pytest.mark.unit
METHODS = ("genpu", "holistic_pu", "lagam", "pulns", "tedn")


def case(method, monkeypatch):
    features = np.random.RandomState(8).normal(size=(24, 3)).astype(np.float32)
    labels = np.r_[np.ones(8, int), np.zeros(16, int)]
    common = dict(hidden_dim=4, batch_size=8, random_state=3, device="cpu")
    kwargs = {}
    if method == "genpu":
        model = GenPUClassifier(
            class_prior=0.4, latent_dim=2, max_epochs=1, classifier_epochs=1, **common
        )
    elif method == "holistic_pu":
        model = HolisticPUClassifier(warmup_epochs=3, max_epochs=1, **common)
    elif method in {"pulns", "lagam"}:
        support = np.random.RandomState(19).normal(size=(8, 3)).astype(np.float32)
        kwargs["support_data"] = support, np.tile([0, 1], 4)
        model = (
            PULNSClassifier(pretrain_epochs=1, episodes=1, classifier_epochs=1, **common)
            if method == "pulns"
            else LaGAMClassifier(
                warmup_epochs=1, max_epochs=2, num_clusters=2, support_batch_size=4, **common
            )
        )
    else:
        import pu_toolbox.estimators.deep.tedn as implementation

        original = implementation.best_bin_estimate
        # Controlled alpha isolates lifecycle; not a BBE accuracy claim.
        monkeypatch.setattr(
            implementation,
            "best_bin_estimate",
            lambda *args, **options: replace(original(*args, **options), alpha_u=0.2),
        )
        model = TEDnClassifier(input_sampling="ts", warm_start_epochs=1, max_epochs=1, **common)
    return model, features, labels, kwargs


def assert_unfitted(model, features):
    assert model.get_pu_metadata()["is_fitted"] is False
    for name in ("predict", "decision_function", "score_samples"):
        with pytest.raises(NotFittedError):
            getattr(model, name)(features)
    if type(model).predict_proba is not BasePUClassifier.predict_proba:
        with pytest.raises(NotFittedError):
            model.predict_proba(features)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("keyword", ["y_true", "support_labels", "test_data"])
def test_param_unknown_roles_invalidate_previous_fit(monkeypatch, method, keyword):
    model, features, labels, kwargs = case(method, monkeypatch)
    model.fit(features, labels, **kwargs)
    with pytest.raises(TypeError, match=keyword):
        model.fit(features, labels, **kwargs, **{keyword: labels})
    assert_unfitted(model, features)


@pytest.mark.parametrize("method", METHODS)
def test_param_missing_required_arguments_invalidate_previous_fit(monkeypatch, method):
    model, features, labels, kwargs = case(method, monkeypatch)
    model.fit(features, labels, **kwargs)
    with pytest.raises(TypeError):
        model.fit(features, **kwargs)
    assert_unfitted(model, features)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, KeyboardInterrupt, SystemExit])
def test_edge_interrupted_fit_does_not_expose_partial_model(monkeypatch, method, error_type):
    model, features, labels, kwargs = case(method, monkeypatch)
    model.fit(features, labels, **kwargs)
    sentinel = error_type("injected training interruption")
    calls = []

    def fail(*args):
        calls.append(1)
        raise sentinel

    if method == "pulns":
        monkeypatch.setattr(model, "_train_selected", fail)
    else:
        kwargs["epoch_callback"] = fail
    with pytest.raises(error_type) as caught:
        model.fit(features, labels, **kwargs)
    assert caught.value is sentinel and calls == [1]
    assert_unfitted(model, features)
    if method == "pulns":
        monkeypatch.undo()
    assert_unfitted(pickle.loads(pickle.dumps(model)), features)


@pytest.mark.parametrize("method", METHODS)
def test_basic_determ_guarded_success_preserves_signature_cost_and_prediction(monkeypatch, method):
    model, features, labels, kwargs = case(method, monkeypatch)
    reference = clone(model)
    original = type(model).fit.__wrapped__
    assert inspect.signature(type(model).fit) == inspect.signature(original)
    calls, reference_calls = [], []

    def record(target):
        return lambda epoch, fitted: target.append(
            (epoch, fitted._is_fitted, fitted.optimizer_steps_)
        )

    current_kwargs, reference_kwargs = dict(kwargs), dict(kwargs)
    if method != "pulns":
        current_kwargs["epoch_callback"] = record(calls)
        reference_kwargs["epoch_callback"] = record(reference_calls)
    assert model.fit(features, labels, **current_kwargs) is model
    assert original(reference, features, labels, **reference_kwargs) is reference
    assert calls == reference_calls
    np.testing.assert_array_equal(
        model.decision_function(features), reference.decision_function(features)
    )
    assert model.history_ == reference.history_
    assert model.optimizer_steps_ == reference.optimizer_steps_
    for name, value in model.model_.state_dict().items():
        torch.testing.assert_close(value, reference.model_.state_dict()[name], rtol=0, atol=0)
    expected = model.decision_function(features).copy()
    with pytest.raises(TypeError):
        model.fit(features, labels, y_true=labels, **kwargs)
    assert_unfitted(model, features)
    assert model.fit(features, labels, **kwargs) is model
    np.testing.assert_array_equal(model.decision_function(features), expected)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.decision_function(features), expected)


@pytest.mark.parametrize("method", METHODS)
def test_basic_determ_cnn_guard_preserves_success_and_rejects_failed_refit(monkeypatch, method):
    model, _, labels, kwargs = case(method, monkeypatch)
    features = np.random.RandomState(5).normal(size=(24, 1, 4, 4)).astype(np.float32)
    torch.manual_seed(11)
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(1, 3, 1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )
    model.set_params(encoder=encoder)
    if method in {"lagam", "pulns"}:
        support = np.random.RandomState(19).normal(size=(8, 1, 4, 4)).astype(np.float32)
        kwargs["support_data"] = support, np.tile([0, 1], 4)
    reference = clone(model)
    model.fit(features, labels, **kwargs)
    type(reference).fit.__wrapped__(reference, features, labels, **kwargs)
    np.testing.assert_array_equal(
        model.decision_function(features), reference.decision_function(features)
    )
    assert model.optimizer_steps_ == reference.optimizer_steps_
    with pytest.raises(TypeError, match="y_true"):
        model.fit(features, labels, y_true=labels, **kwargs)
    assert_unfitted(model, features)
