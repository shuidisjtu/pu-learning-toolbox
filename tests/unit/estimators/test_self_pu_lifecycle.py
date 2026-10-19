"""Self-PU public fitted state and teacher inference isolation."""

# ruff: noqa: N803, N806

import inspect
import pickle
from contextlib import nullcontext

import numpy as np
import pytest
from sklearn.base import clone

from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.estimators.deep.self_pu import SelfPUClassifier, _pu_validation_risk

pytestmark = pytest.mark.unit


def data():
    rng = np.random.RandomState(29)
    features = rng.normal(size=(18, 4)).astype(np.float32)
    labels = np.r_[np.ones(6, dtype=int), np.zeros(12, dtype=int)]
    validation = rng.normal(size=(8, 4)).astype(np.float32)
    clean_labels = np.r_[np.ones(4, dtype=int), np.zeros(4, dtype=int)]
    return features, labels, validation, clean_labels


def estimator(**overrides):
    parameters = dict(
        hidden_dim=4,
        warmup_epochs=0,
        self_paced_start=0,
        self_paced_end=1,
        distill_start=1,
        max_epochs=2,
        batch_size=6,
        device="cpu",
        random_state=11,
    )
    parameters.update(overrides)
    return SelfPUClassifier(0.4, **parameters)


def assert_unfitted(model, features):
    assert model.get_pu_metadata()["is_fitted"] is False
    for name in ("predict", "decision_function", "score_samples", "predict_proba"):
        with pytest.raises(NotFittedError):
            getattr(model, name)(features)
    with pytest.raises(NotFittedError):
        model.get_training_checkpoint()


def test_edge_fresh_probability_and_empty_probability_are_not_fitted():
    features, _, _, _ = data()
    model = estimator()
    assert_unfitted(model, features)
    with pytest.raises(NotFittedError):
        model.predict_proba(features[:0])


@pytest.mark.parametrize(
    "failure",
    ["prior", "sample_weight", "clean_labels", "view", "argument_binding"],
)
def test_param_failed_refit_blocks_old_probabilities_and_training_checkpoint(failure):
    pytest.importorskip("torch")
    features, labels, validation, clean_labels = data()
    model = estimator().fit(features, labels, validation_data=(validation, clean_labels))
    before = model.predict_proba(features)
    kwargs = {"validation_data": (validation, clean_labels)}
    if failure == "prior":
        kwargs["class_prior"] = -0.1
    elif failure == "sample_weight":
        kwargs["sample_weight"] = np.ones(len(labels))
    elif failure == "clean_labels":
        kwargs["validation_data"] = validation, np.ones(len(validation))
    elif failure == "view":
        kwargs["os_or_ts"] = "invalid"
    else:
        kwargs["y_true"] = labels

    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        model.fit(features, labels, **kwargs)

    assert_unfitted(model, features)
    model.fit(features, labels, validation_data=(validation, clean_labels))
    np.testing.assert_array_equal(model.predict_proba(features), before)


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, KeyboardInterrupt, SystemExit])
def test_edge_interrupted_refit_propagates_once_and_revokes_old_teacher(error_type):
    pytest.importorskip("torch")
    features, labels, validation, clean_labels = data()
    model = estimator().fit(features, labels, validation_data=(validation, clean_labels))
    calls = []
    failure = error_type("Self-PU callback interrupted")

    def interrupt(epoch, current):
        calls.append(epoch)
        assert current._is_fitted is False
        raise failure

    with pytest.raises(error_type) as caught:
        model.fit(
            features, labels, validation_data=(validation, clean_labels), epoch_callback=interrupt
        )

    assert caught.value is failure
    assert calls == [1]
    assert_unfitted(model, features)


@pytest.mark.parametrize("view", ["os", "ts"])
@pytest.mark.parametrize("scope", ["untrusted_u", "paper_population"])
@pytest.mark.parametrize("selection", ["clean", "pu", "ablation"])
def test_basic_determ_success_preserves_both_views_scopes_and_selection_paths(
    view, scope, selection
):
    torch = pytest.importorskip("torch")
    features, labels, validation, clean_labels = data()
    kwargs = {"os_or_ts": view}
    if selection == "clean":
        kwargs["validation_data"] = validation, clean_labels
    elif selection == "pu":
        kwargs["pu_validation_data"] = validation, clean_labels
    guarded, reference = estimator(distillation_scope=scope), estimator(distillation_scope=scope)

    original = SelfPUClassifier.fit.__wrapped__
    for model, fit in (
        (guarded, guarded.fit),
        (reference, lambda *a, **k: original(reference, *a, **k)),
    ):
        context = (
            nullcontext()
            if selection == "clean"
            else pytest.warns(UserWarning, match="explicit Self-PU ablation")
        )
        with context:
            assert fit(features, labels, **kwargs) is model

    np.testing.assert_array_equal(
        guarded.predict_proba(features), reference.predict_proba(features)
    )
    assert guarded.history_ == reference.history_
    assert guarded.training_history_ == reference.training_history_
    assert guarded.teacher_selection_basis_ == reference.teacher_selection_basis_
    assert guarded.teacher_selection_metrics_ == reference.teacher_selection_metrics_
    for attribute in ("student_1_", "student_2_", "teacher_1_", "teacher_2_"):
        actual, expected = getattr(guarded, attribute), getattr(reference, attribute)
        for name, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)


def test_basic_signature_clone_and_pickle_remain_compatible():
    pytest.importorskip("torch")
    features, labels, validation, clean_labels = data()
    model = estimator().fit(features, labels, validation_data=(validation, clean_labels))
    original = SelfPUClassifier.fit.__wrapped__
    assert inspect.signature(SelfPUClassifier.fit) == inspect.signature(original)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert_unfitted(cloned, features)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict_proba(features), model.predict_proba(features))


@pytest.mark.parametrize("path", ["probability", "pu_validation"])
def test_basic_determ_scoring_preserves_mixed_modes_bn_buffers_and_rng(path):
    torch = pytest.importorskip("torch")
    features, labels, _, _ = data()
    backbone = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(4, 1),
    ).eval()
    with pytest.warns(UserWarning, match="explicit Self-PU ablation"):
        model = estimator(backbone=backbone).fit(features, labels)
    teacher = model.best_teacher_
    teacher.train()
    teacher[2].eval()
    modes = [module.training for module in teacher.modules()]
    buffers = {name: value.clone() for name, value in teacher.named_buffers()}
    rng_before = torch.get_rng_state().clone()

    def score():
        if path == "probability":
            return model.predict_proba(features)
        return _pu_validation_risk(teacher, torch.as_tensor(features), labels, 0.4)

    np.testing.assert_array_equal(score(), score())
    assert [module.training for module in teacher.modules()] == modes
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    for name, value in teacher.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)


@pytest.mark.parametrize("path", ["probability", "pu_validation"])
@pytest.mark.parametrize("stage", ["eval", "forward"])
def test_edge_scoring_failure_restores_modes_including_partial_eval(monkeypatch, path, stage):
    torch = pytest.importorskip("torch")
    features, labels, validation, clean_labels = data()
    model = estimator().fit(features, labels, validation_data=(validation, clean_labels))
    teacher = model.best_teacher_
    teacher.train()
    teacher[-1].eval()
    modes = [module.training for module in teacher.modules()]
    failure = RuntimeError("Self-PU teacher scoring interrupted")

    def fail(*_args):
        teacher[0].training = False
        raise failure

    monkeypatch.setattr(teacher, stage, fail)
    with pytest.raises(RuntimeError) as caught:
        if path == "probability":
            model.predict_proba(features)
        else:
            _pu_validation_risk(teacher, torch.as_tensor(features), labels, 0.4)

    assert caught.value is failure
    assert [module.training for module in teacher.modules()] == modes


def test_edge_empty_and_invalid_prediction_leave_teacher_untouched(monkeypatch):
    pytest.importorskip("torch")
    features, labels, validation, clean_labels = data()
    model = estimator().fit(features, labels, validation_data=(validation, clean_labels))
    teacher = model.best_teacher_
    modes = [module.training for module in teacher.modules()]

    def fail():
        raise AssertionError("empty or invalid prediction must not evaluate the teacher")

    monkeypatch.setattr(teacher, "eval", fail)
    assert model.predict(features[:0]).shape == (0,)
    assert model.decision_function(features[:0]).shape == (0,)
    assert model.predict_proba(features[:0]).shape == (0, 2)
    with pytest.raises(ValueError, match="sample shape"):
        model.predict_proba(np.empty((0, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.predict_proba(np.full((1, 4), np.nan, dtype=np.float32))
    assert [module.training for module in teacher.modules()] == modes
