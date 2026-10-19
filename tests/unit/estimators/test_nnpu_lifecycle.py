"""nnPU failed-refit invalidation and side-effect-free inference."""

# ruff: noqa: N803, N806

import inspect
import pickle

import numpy as np
import pytest
from sklearn.base import clone

from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier

pytestmark = pytest.mark.unit


def data():
    features = np.random.RandomState(21).normal(size=(18, 3)).astype(np.float32)
    return features, np.r_[np.ones(6, dtype=int), np.zeros(12, dtype=int)]


def estimator(**kwargs):
    return NonNegativePUClassifier(
        max_epochs=2, batch_size=3, class_prior=0.4, random_state=7, device="cpu", **kwargs
    )


def assert_unfitted(model, features):
    assert model.get_pu_metadata()["is_fitted"] is False
    for method in ("predict", "decision_function", "score_samples"):
        with pytest.raises(NotFittedError):
            getattr(model, method)(features)


@pytest.mark.parametrize("kwargs", [{"class_prior": -0.1}, {"y_true": None}])
def test_param_failed_refit_including_argument_binding_blocks_old_predictions(kwargs):
    pytest.importorskip("torch")
    features, labels = data()
    fitted = estimator().fit(features, labels)
    before = fitted.decision_function(features)

    with pytest.raises((ValueError, TypeError)):
        fitted.fit(features, labels, **kwargs)

    assert_unfitted(fitted, features)
    fitted.fit(features, labels)
    np.testing.assert_array_equal(fitted.decision_function(features), before)


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, KeyboardInterrupt, SystemExit])
def test_edge_callback_failure_invalidates_partial_fit_and_propagates_once(error_type):
    pytest.importorskip("torch")
    features, labels = data()
    fitted = estimator().fit(features, labels)
    calls = []
    failure = error_type("nnPU callback interrupted")

    def interrupt(epoch, current):
        calls.append(epoch)
        assert current._is_fitted is False
        raise failure

    with pytest.raises(error_type) as caught:
        fitted.fit(features, labels, epoch_callback=interrupt)

    assert caught.value is failure
    assert calls == [0]
    assert_unfitted(fitted, features)


def test_basic_determ_guard_preserves_success_scores_history_signature_clone_and_pickle():
    pytest.importorskip("torch")
    features, labels = data()
    guarded, reference = estimator(), estimator()
    original = NonNegativePUClassifier.fit.__wrapped__
    guarded.fit(features, labels)
    original(reference, features, labels)

    np.testing.assert_array_equal(
        guarded.decision_function(features), reference.decision_function(features)
    )
    assert guarded.history_ == reference.history_
    assert inspect.signature(NonNegativePUClassifier.fit) == inspect.signature(original)
    cloned = clone(guarded)
    assert cloned.get_params() == guarded.get_params()
    assert_unfitted(cloned, features)
    restored = pickle.loads(pickle.dumps(guarded))
    np.testing.assert_array_equal(
        restored.decision_function(features), guarded.decision_function(features)
    )


def test_basic_determ_prediction_restores_mixed_bn_dropout_modes_and_buffers():
    torch = pytest.importorskip("torch")
    features, labels = data()
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(4, 1),
    )
    fitted = estimator(model=model).fit(features, labels)
    fitted.model_.train()
    fitted.model_[2].eval()
    modes = [module.training for module in fitted.model_.modules()]
    buffers = {name: value.clone() for name, value in fitted.model_.named_buffers()}

    first = fitted.decision_function(features)
    second = fitted.score_samples(features)

    np.testing.assert_array_equal(first, second)
    assert [module.training for module in fitted.model_.modules()] == modes
    for name, value in fitted.model_.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)


@pytest.mark.parametrize("failure_stage", ["eval", "forward"])
def test_edge_prediction_failure_restores_modes_even_after_partial_eval(monkeypatch, failure_stage):
    torch = pytest.importorskip("torch")
    features, labels = data()
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(), torch.nn.Linear(4, 1))
    fitted = estimator(model=model).fit(features, labels)
    fitted.model_.train()
    fitted.model_[1].eval()
    before = [module.training for module in fitted.model_.modules()]
    failure = RuntimeError("nnPU prediction interrupted")

    def fail(*_args):
        fitted.model_[0].training = False
        raise failure

    monkeypatch.setattr(fitted.model_, failure_stage, fail)
    with pytest.raises(RuntimeError) as caught:
        fitted.decision_function(features)

    assert caught.value is failure
    assert [module.training for module in fitted.model_.modules()] == before


def test_edge_empty_prediction_does_not_call_model_or_change_modes(monkeypatch):
    pytest.importorskip("torch")
    features, labels = data()
    fitted = estimator().fit(features, labels)
    before = fitted.model_.training

    def fail():
        raise AssertionError("empty prediction must not evaluate the model")

    monkeypatch.setattr(fitted.model_, "eval", fail)
    assert fitted.decision_function(features[:0]).shape == (0,)
    assert fitted.model_.training is before
