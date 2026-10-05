"""Split-PU source-derived losses, stages and safe API boundaries."""

# ruff: noqa: N806

import copy
import importlib
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402
from pu_toolbox.estimators.deep.split_pu import (  # noqa: E402
    SplitPUClassifier,
    splitpu_js_loss,
)
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402
from pu_toolbox.experiment.method_ledger import load_ledger  # noqa: E402
from pu_toolbox.experiment.training_views import resolve_training_view  # noqa: E402

pytestmark = pytest.mark.unit


def _data():
    rng = np.random.RandomState(4)
    X = np.vstack([rng.normal(1, 0.5, (8, 3)), rng.normal(-0.5, 0.8, (16, 3))])
    return X.astype(np.float32), np.r_[np.ones(8, int), np.zeros(16, int)]


def _model(**kwargs):
    args = dict(
        class_prior=0.4,
        hidden_dim=8,
        teacher_epochs=1,
        split_epochs=1,
        student_epochs=1,
        rounds=2,
        batch_size=6,
        random_state=7,
        device="cpu",
    )
    args.update(kwargs)
    return SplitPUClassifier(**args)


@pytest.mark.math
def test_js_matches_manual_bernoulli_divergence_and_zero_for_equal_predictions():
    student = torch.tensor([-1.0, 0.5], requires_grad=True)
    teacher = torch.tensor([1.0, -0.5])
    actual = splitpu_js_loss(student, teacher)
    p = torch.sigmoid(teacher)
    q = torch.sigmoid(student)
    m = 0.7 * p + 0.3 * q
    kl_p = p * torch.log(p / m) + (1 - p) * torch.log((1 - p) / (1 - m))
    kl_q = q * torch.log(q / m) + (1 - q) * torch.log((1 - q) / (1 - m))
    expected = (0.7 * kl_p + 0.3 * kl_q).mean() / (-0.3 * np.log(0.3))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert student.grad is not None
    assert teacher.grad is None
    assert splitpu_js_loss(teacher, teacher) < 1e-6


def test_stages_prediction_determinism_and_checkpoint(tmp_path):
    X, y = _data()
    clf = _model()
    with pytest.raises(NotFittedError):
        clf.predict(X)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(clf, X, y)
    assert len(trajectory.checkpoints) == 4
    assert clf.n_easy_ > 0 and clf.n_hard_ > 0
    assert len(clf.history_["teacher_risk"]) == 1
    assert len(clf.history_["student_loss"]) == 2
    assert clf.history_["round_weights"] == [
        {"hard": 0.3, "feature": 0.3, "similarity": 0.1},
        {"hard": 0.01, "feature": 0.0, "similarity": 0.0},
    ]
    scores = clf.decision_function(X)
    assert np.isfinite(scores).all() and scores.shape == (len(X),)
    np.testing.assert_array_equal(_model().fit(X, y).decision_function(X), scores)
    # Snapshot replay uses one full matrix; fitted prediction is batched.
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore().decision_function(X), scores, atol=1e-7
    )
    restored = copy.deepcopy(clf.model_)
    restored.load_state_dict(clf.model_.state_dict())
    with torch.no_grad():
        np.testing.assert_allclose(
            restored(torch.as_tensor(X)).reshape(-1).numpy(), scores, atol=1e-7
        )


def test_scores_are_bounded_batches_with_cpu_pool_and_mode_restoration(monkeypatch):
    X, y = _data()
    module = importlib.import_module("pu_toolbox.estimators.deep.split_pu")
    original_scan, original_score = module._batched_score, module._score
    scans, sizes = [], []

    def scan(model, data, batch_size, device):
        scans.append((data.device.type, len(data)))
        return original_scan(model, data, batch_size, device)

    def score(model, batch):
        sizes.append(len(batch))
        return original_score(model, batch)

    monkeypatch.setattr(module, "_batched_score", scan)
    monkeypatch.setattr(module, "_score", score)
    fitted = _model().fit(X, y, os_or_ts="ts")
    assert scans and all(kind == "cpu" for kind, _ in scans)
    assert max(sizes) <= fitted.batch_size
    assert fitted.n_easy_ + fitted.n_hard_ == int((y == 0).sum())
    assert fitted.n_loss_unlabeled_ == len(y)
    assert fitted.model_.training
    first = fitted.decision_function(X)
    assert fitted.model_.training
    fitted.model_.eval()
    np.testing.assert_array_equal(fitted.decision_function(X), first)
    assert not fitted.model_.training
    assert fitted.decision_function(X[:0]).shape == (0,)
    assert max(sizes) <= fitted.batch_size


def test_single_u_cannot_support_two_nonempty_groups():
    X, y = _data()
    indices = np.r_[np.flatnonzero(y == 1), np.flatnonzero(y == 0)[:1]]
    with pytest.raises(ValueError, match="at least two original U"):
        _model().fit(X[indices], y[indices])


def test_prediction_rejects_float32_overflow():
    X, y = _data()
    fitted = _model().fit(X, y)
    huge = np.full(X.shape, np.finfo(np.float64).max)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="float32"):
        fitted.decision_function(huge)


def test_ts_calibrates_teacher_but_preserves_original_u_split():
    X, y = _data()
    assert (
        resolve_training_view(
            load_ledger(), "split_pu", None, is_oracle=False, estimator_class=SplitPUClassifier
        )
        == "ts"
    )
    os_model = _model().fit(X, y, os_or_ts="os")
    ts_model = _model().fit(X, y, os_or_ts="ts")
    assert not os_model.calibration_applied_
    assert ts_model.calibration_applied_
    assert ts_model.n_loss_unlabeled_ == len(y)
    assert ts_model.n_unlabeled_ == int((y == 0).sum())
    assert ts_model.n_easy_ + ts_model.n_hard_ == ts_model.n_unlabeled_
    assert os_model.history_["teacher_risk"] != ts_model.history_["teacher_risk"]
    with pytest.raises(ValueError, match="requested_view"):
        _model().fit(X, y, os_or_ts="unknown")


@pytest.mark.math
def test_ts_teacher_marginal_is_exact_p_u_union(monkeypatch):
    X, y = _data()
    module = importlib.import_module("pu_toolbox.estimators.deep.split_pu")
    original = module._nnpu_train_step
    inputs = []

    def capture(pos_loss, pos_as_negative, unl_loss, *, class_prior):
        inputs.append(
            (float(pos_loss.detach()), float(pos_as_negative.detach()), float(unl_loss.detach()))
        )
        return original(pos_loss, pos_as_negative, unl_loss, class_prior=class_prior)

    monkeypatch.setattr(module, "_nnpu_train_step", capture)
    _model(batch_size=32).fit(X, y, os_or_ts="os")
    _model(batch_size=32).fit(X, y, os_or_ts="ts")
    os_input, ts_input = inputs
    np.testing.assert_allclose(os_input[:2], ts_input[:2], rtol=0, atol=0)
    np.testing.assert_allclose(ts_input[2], (os_input[2] * 16 + os_input[1] * 8) / 24)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"class_prior": None}, "class_prior"),
        ({"class_prior": 1}, "class_prior"),
        ({"teacher_epochs": 0}, "teacher_epochs"),
        ({"rounds": 0}, "rounds"),
        ({"agreement_threshold": 1}, "agreement_threshold"),
        ({"noise_std": -1}, "noise_std"),
    ],
)
def test_invalid_parameters_fail(params, message):
    X, y = _data()
    with pytest.raises(ValueError, match=message):
        _model(**params).fit(X, y)


def test_rejects_weights_and_bad_input():
    X, y = _data()
    with pytest.raises(NotImplementedError, match="sample_weight"):
        _model().fit(X, y, sample_weight=np.ones(len(y)))
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _model().fit(bad, y)
    fitted = _model().fit(X, y)
    with pytest.raises(ValueError, match="feature shape"):
        fitted.predict(X[:, :2])


@pytest.mark.gpu
def test_cuda_smoke(monkeypatch, tmp_path):
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    X, y = _data()
    module = importlib.import_module("pu_toolbox.estimators.deep.split_pu")
    original = module._batched_score
    scans = []

    def capture(model, data, batch_size, device):
        scans.append(data.device.type)
        return original(model, data, batch_size, device)

    monkeypatch.setattr(module, "_batched_score", capture)
    fitted = _model(device="cuda")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, X, y, os_or_ts="ts")
    assert next(fitted.model_.parameters()).device.type == "cuda"
    scores = fitted.decision_function(X)
    assert np.isfinite(scores).all()
    assert scans and set(scans) == {"cpu"}
    assert fitted.n_loss_unlabeled_ == len(y)
    restored = trajectory.checkpoints[-1].restore(device="cpu")
    assert next(restored.model_.parameters()).device.type == "cpu"
    np.testing.assert_allclose(restored.decision_function(X), scores, atol=1e-6, rtol=1e-5)
