"""Split-PU source-derived losses, stages and safe API boundaries."""

# ruff: noqa: N806

import copy
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
    np.testing.assert_allclose(trajectory.checkpoints[-1].restore().decision_function(X), scores)
    restored = copy.deepcopy(clf.model_)
    restored.load_state_dict(clf.model_.state_dict())
    with torch.no_grad():
        np.testing.assert_allclose(restored(torch.as_tensor(X)).reshape(-1).numpy(), scores)


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
def test_cuda_smoke():
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    X, y = _data()
    fitted = _model(device="cuda").fit(X, y)
    assert next(fitted.model_.parameters()).device.type == "cuda"
    assert np.isfinite(fitted.decision_function(X)).all()
