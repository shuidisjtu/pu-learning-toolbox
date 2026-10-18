"""Audited five-method fit lifecycle, not a change to their PU objectives."""

# Match public feature naming.
# ruff: noqa: N803, N806

import inspect
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox import (  # noqa: E402
    GradPUClassifier,
    PUExtraTreesClassifier,
    PULDAClassifier,
    RobustPUClassifier,
    SplitPUClassifier,
)
from pu_toolbox.core.base import BasePUClassifier  # noqa: E402
from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402

pytestmark = pytest.mark.unit
METHODS = ("pulda", "puet", "gradpu", "robust_pu", "split_pu")
STAGES = [
    ("pulda", 0, "warmup"),
    ("pulda", 1, "pu_mixup"),
    ("robust_pu", 0, "pretrain"),
    ("robust_pu", 1, "self_paced"),
    ("split_pu", 0, "teacher"),
    ("split_pu", 1, "splitter"),
    ("split_pu", 2, "student"),
    ("gradpu", 0, "pu"),
]


def data():
    rng = np.random.RandomState(8)
    X = np.r_[rng.normal(1, 0.5, (8, 3)), rng.normal(-0.5, 0.7, (16, 3))]
    return X.astype("float32"), np.r_[np.ones(8), np.zeros(16)]


def model(method):
    common = dict(random_state=17)
    if method == "puet":
        return PUExtraTreesClassifier(0.4, n_estimators=2, max_depth=2, **common)
    common.update(hidden_dim=4, device="cpu")
    if method == "pulda":
        return PULDAClassifier(
            0.4,
            warmup_epochs=1,
            pu_epochs=1,
            positive_batch_size=4,
            unlabeled_batch_size=8,
            **common,
        )
    common["batch_size"] = 8
    if method == "gradpu":
        return GradPUClassifier(max_epochs=1, **common)
    if method == "robust_pu":
        return RobustPUClassifier(0.4, pretrain_epochs=1, episodes=1, **common)
    return SplitPUClassifier(
        0.4, teacher_epochs=1, split_epochs=1, student_epochs=1, rounds=1, **common
    )


def assert_unfitted(fitted, X):
    for name in ("predict", "decision_function", "score_samples"):
        with pytest.raises(NotFittedError):
            getattr(fitted, name)(X)
    if type(fitted).predict_proba is not BasePUClassifier.predict_proba:
        with pytest.raises(NotFittedError):
            fitted.predict_proba(X)
    assert fitted.get_pu_metadata()["is_fitted"] is False


@pytest.mark.parametrize("method", METHODS)
def test_param_unsupported_sample_weight_invalidates_previous_fit(method):
    X, y = data()
    fitted = model(method).fit(X, y)
    assert fitted.predict(X).shape == (len(X),)
    with pytest.raises(NotImplementedError, match="sample_weight"):
        fitted.fit(X, y, sample_weight=np.ones(len(X)))
    assert_unfitted(fitted, X)


@pytest.mark.parametrize("method", METHODS)
def test_param_invalid_prior_invalidates_previous_fit(method):
    X, y = data()
    fitted = model(method).fit(X, y)
    with pytest.raises(ValueError):
        fitted.fit(X, y, class_prior=-0.1)
    assert_unfitted(fitted, X)


@pytest.mark.parametrize("method", METHODS)
def test_param_unknown_keyword_cannot_leave_previous_model_usable(method):
    X, y = data()
    fitted = model(method).fit(X, y)
    with pytest.raises(TypeError, match="y_true"):
        fitted.fit(X, y, y_true=y)
    assert_unfitted(fitted, X)


@pytest.mark.parametrize("method,position,stage", STAGES)
@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, KeyboardInterrupt, SystemExit])
def test_edge_callback_exception_invalidates_partial_stage_without_retry(
    method, position, stage, error_type
):
    X, y = data()
    fitted = model(method).fit(X, y)
    seen = []
    failure = error_type("candidate callback interrupted")

    def callback(epoch, current):
        seen.append((epoch, current.checkpoint_stage_))
        if epoch == position:
            raise failure

    with pytest.raises(error_type) as caught:
        fitted.fit(X, y, epoch_callback=callback)
    assert caught.value is failure
    assert seen[-1] == (position, stage)
    assert len(seen) == position + 1
    assert_unfitted(fitted, X)


@pytest.mark.parametrize("method", METHODS)
def test_basic_determ_guarded_success_matches_unwrapped_training_and_cost(method):
    X, y = data()
    guarded, reference = model(method), model(method)
    calls, reference_calls = [], []

    def record(target):
        def callback(epoch, current):
            target.append((epoch, current._is_fitted, current.optimizer_steps_))

        return callback

    kwargs = {} if method == "puet" else {"epoch_callback": record(calls)}
    assert guarded.fit(X, y, **kwargs) is guarded
    baseline_kwargs = {} if method == "puet" else {"epoch_callback": record(reference_calls)}
    original = type(reference).fit.__wrapped__
    assert original(reference, X, y, **baseline_kwargs) is reference
    np.testing.assert_array_equal(guarded.decision_function(X), reference.decision_function(X))
    assert calls == reference_calls
    if hasattr(guarded, "optimizer_steps_"):
        assert guarded.optimizer_steps_ == reference.optimizer_steps_
    assert clone(guarded).get_params() == guarded.get_params()
    assert not clone(guarded)._is_fitted
    assert inspect.signature(type(guarded).fit) == inspect.signature(original)
    assert type(guarded).fit.__name__ == "fit"


@pytest.mark.parametrize("method", METHODS)
def test_basic_failed_refit_can_be_followed_by_new_success(method):
    X, y = data()
    fitted = model(method).fit(X, y)
    expected = fitted.decision_function(X).copy()
    with pytest.raises(NotImplementedError):
        fitted.fit(X, y, sample_weight=np.ones(len(X)))
    assert_unfitted(fitted, X)
    assert fitted.fit(X, y) is fitted
    np.testing.assert_array_equal(fitted.decision_function(X), expected)


@pytest.mark.parametrize("method", METHODS)
def test_param_nonfinite_input_cannot_reuse_previous_model(method):
    X, y = data()
    fitted = model(method).fit(X, y)
    broken = X.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError):
        fitted.fit(broken, y)
    assert_unfitted(fitted, X)


def test_edge_puet_tree_failure_invalidates_without_replacing_old_diagnostics(monkeypatch):
    X, y = data()
    fitted = model("puet").fit(X, y)
    previous_trees = fitted.trees_
    build = fitted._build_tree
    calls = []
    failure = RuntimeError("second tree failed")

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise failure
        return build(*args, **kwargs)

    monkeypatch.setattr(fitted, "_build_tree", interrupted)
    with pytest.raises(RuntimeError) as caught:
        fitted.fit(X, y)
    assert caught.value is failure and len(calls) == 2
    assert fitted.trees_ is previous_trees  # Diagnostic retention is not rollback.
    assert_unfitted(fitted, X)


@pytest.mark.parametrize("method", METHODS)
def test_determ_pickle_preserves_success_and_failed_fit_state(method):
    X, y = data()
    fitted = model(method).fit(X, y)
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.decision_function(X), fitted.decision_function(X))
    with pytest.raises(NotImplementedError):
        restored.fit(X, y, sample_weight=np.ones(len(X)))
    assert_unfitted(pickle.loads(pickle.dumps(restored)), X)


def image_model(method):
    torch.manual_seed(11)
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )
    return model(method).set_params(encoder=encoder)


@pytest.mark.parametrize("method", [name for name in METHODS if name != "puet"])
def test_basic_determ_cnn_success_is_unchanged_and_failed_refit_rejects_prediction(method):
    X = np.random.RandomState(5).normal(size=(24, 3, 4, 4)).astype("float32")
    _, y = data()
    guarded, reference = image_model(method), image_model(method)
    guarded.fit(X, y, os_or_ts="ts")
    type(reference).fit.__wrapped__(reference, X, y, os_or_ts="ts")
    np.testing.assert_array_equal(guarded.decision_function(X), reference.decision_function(X))
    assert guarded.optimizer_steps_ == reference.optimizer_steps_
    with pytest.raises(NotImplementedError):
        guarded.fit(X, y, sample_weight=np.ones(len(X)), os_or_ts="ts")
    assert_unfitted(guarded, X)


@pytest.mark.parametrize("method,position,stage", STAGES)
def test_edge_cnn_partial_stage_failure_invalidates_model(method, position, stage):
    X = np.random.RandomState(5).normal(size=(24, 3, 4, 4)).astype("float32")
    _, y = data()
    fitted = image_model(method)
    failure = RuntimeError("CNN callback failed")
    seen = []

    def callback(epoch, current):
        seen.append((epoch, current.checkpoint_stage_))
        if epoch == position:
            raise failure

    with pytest.raises(RuntimeError) as caught:
        fitted.fit(X, y, epoch_callback=callback, os_or_ts="ts")
    assert caught.value is failure
    assert seen[-1] == (position, stage) and len(seen) == position + 1
    assert_unfitted(fitted, X)
