"""Robust-PU source-derived self-paced rules and PU-only training gates."""

# ruff: noqa: N806

import copy
import importlib
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402
from pu_toolbox.estimators.deep.robust_pu import (  # noqa: E402
    RobustPUClassifier,
    _episode_weights,
    self_paced_weights,
)
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402
from pu_toolbox.experiment.method_ledger import load_ledger  # noqa: E402
from pu_toolbox.experiment.training_views import resolve_training_view  # noqa: E402
from pu_toolbox.registry import get_algorithm, register_all_builtin_methods  # noqa: E402
from pu_toolbox.workflows import PUPipeline  # noqa: E402

pytestmark = pytest.mark.unit


def _data():
    rng = np.random.RandomState(8)
    X = np.vstack((rng.normal(1, 0.5, (8, 3)), rng.normal(-0.5, 0.7, (16, 3))))
    return X.astype(np.float32), np.r_[np.ones(8, int), np.zeros(16, int)]


def _estimator(**kwargs):
    params = dict(
        class_prior=0.4,
        hidden_dim=8,
        pretrain_epochs=2,
        episodes=3,
        inner_epochs=1,
        batch_size=5,
        random_state=17,
        device="cpu",
    )
    params.update(kwargs)
    return RobustPUClassifier(**params)


@pytest.mark.math
def test_self_paced_rules_match_source_formulas_and_detach():
    losses = torch.tensor([0.0, 0.5, 2.0], requires_grad=True)
    torch.testing.assert_close(
        self_paced_weights(losses, 1.0, "hard"), torch.tensor([1.0, 1.0, 0.0])
    )
    torch.testing.assert_close(
        self_paced_weights(losses, 1.0, "linear"), torch.tensor([1.0, 0.5, 0.0])
    )
    welsch = self_paced_weights(losses, 2.0)
    torch.testing.assert_close(welsch, torch.exp(-losses.detach() / 4))
    assert not welsch.requires_grad


@pytest.mark.math
def test_pu_group_hardness_has_opposite_logit_directions():
    logits = torch.tensor([-2.0, 2.0, -2.0, 2.0], requires_grad=True)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    weights = _episode_weights(
        logits,
        labels,
        threshold_p=1.0,
        threshold_n=1.0,
        temper_p=1.0,
        temper_n=1.0,
        kind="welsch",
    )
    assert weights[1] > weights[0]  # easy positive: high positive logit
    assert weights[2] > weights[3]  # easy candidate negative: low positive logit
    assert not weights.requires_grad


def test_fit_predict_checkpoint_registry_and_determinism(tmp_path):
    X, y = _data()
    model = _estimator()
    with pytest.raises(NotFittedError):
        model.predict(X)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, X, y)
    scores = model.decision_function(X)
    assert scores.shape == (len(X),) and np.isfinite(scores).all()
    assert set(model.predict(X)) <= {0, 1}
    assert len(trajectory.checkpoints) == 5
    np.testing.assert_allclose(trajectory.checkpoints[-1].restore().decision_function(X), scores)
    np.testing.assert_array_equal(_estimator().fit(X, y).decision_function(X), scores)
    assert len(model.history_["pretrain_risk"]) == 2
    assert len(model.history_["episode_loss"]) == 3
    assert model.history_["threshold_p"][0] == 0.1
    register_all_builtin_methods()
    assert get_algorithm("robust_pu") is RobustPUClassifier
    assert get_algorithm("robust-pu") is RobustPUClassifier


def test_weights_remain_aligned_with_samples_across_shuffles():
    X, y = _data()
    model = _estimator(pretrain_epochs=0, phi=0.5, spl_type="linear").fit(X, y)
    assert all(0 <= value <= 1 for value in model.history_["positive_weight"])
    assert all(0 <= value <= 1 for value in model.history_["unlabeled_weight"])
    assert model.history_["positive_weight"] != model.history_["unlabeled_weight"]


def test_edge_single_batch_and_threshold_cap():
    """The degenerate end of the batch and threshold schedules.

    ``batch_size == len(X)`` collapses both the pretrain loop and the episode
    loop to a single step.  ``grow_steps=1`` reaches the threshold ceiling on
    the second episode, so the third one exercises the ``min(..., 1.0)``
    clamp instead of the linear ramp.  ``phi=0`` drops the cross-episode
    moving average entirely.
    """
    X, y = _data()
    model = _estimator(
        pretrain_epochs=1,
        episodes=3,
        batch_size=len(X),
        grow_steps=1,
        phi=0.0,
    ).fit(X, y)
    assert model.history_["threshold_p"] == [0.1, 2.0, 2.0]
    assert model.history_["threshold_n"] == [0.1, 2.0, 2.0]
    assert len(model.history_["episode_loss"]) == 3
    assert all(0 < value <= 1 for value in model.history_["positive_weight"])
    assert all(0 < value <= 1 for value in model.history_["unlabeled_weight"])
    scores = model.decision_function(X)
    assert scores.shape == (len(X),) and np.isfinite(scores).all()


def test_ts_calibrates_only_warmup_and_routes_from_ledger():
    X, y = _data()
    ledger = load_ledger()
    assert (
        resolve_training_view(
            ledger, "robust_pu", None, is_oracle=False, estimator_class=RobustPUClassifier
        )
        == "ts"
    )
    os_model = _estimator(pretrain_epochs=1, episodes=1).fit(X, y, os_or_ts="os")
    ts_model = _estimator(pretrain_epochs=1, episodes=1).fit(X, y, os_or_ts="ts")
    assert not os_model.calibration_applied_
    assert ts_model.calibration_applied_
    assert ts_model.n_loss_unlabeled_ == len(y)
    assert ts_model.n_unlabeled_ == int((y == 0).sum())
    assert os_model.history_["pretrain_risk"] != ts_model.history_["pretrain_risk"]
    with pytest.raises(ValueError, match="requires pretrain_epochs"):
        _estimator(pretrain_epochs=0).fit(X, y, os_or_ts="ts")
    with pytest.raises(ValueError, match="requested_view"):
        _estimator().fit(X, y, os_or_ts="unknown")


@pytest.mark.math
def test_ts_warmup_marginal_is_exact_p_u_union(monkeypatch):
    X, y = _data()
    module = importlib.import_module("pu_toolbox.estimators.deep.robust_pu")
    original = module._nnpu_train_step
    inputs = []

    def capture(pos_loss, pos_as_negative, unl_loss, *, class_prior):
        inputs.append(
            (float(pos_loss.detach()), float(pos_as_negative.detach()), float(unl_loss.detach()))
        )
        return original(pos_loss, pos_as_negative, unl_loss, class_prior=class_prior)

    monkeypatch.setattr(module, "_nnpu_train_step", capture)
    _estimator(pretrain_epochs=1, episodes=1, batch_size=32).fit(X, y, os_or_ts="os")
    _estimator(pretrain_epochs=1, episodes=1, batch_size=32).fit(X, y, os_or_ts="ts")
    os_input, ts_input = inputs
    np.testing.assert_allclose(os_input[:2], ts_input[:2], rtol=0, atol=0)
    np.testing.assert_allclose(ts_input[2], (os_input[2] * 16 + os_input[1] * 8) / 24)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"class_prior": None}, "class_prior"),
        ({"class_prior": 1.0}, "class_prior"),
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"episodes": 0}, "episodes"),
        ({"grow_steps": 0}, "grow_steps"),
        ({"temper_n": 0}, "temper_n"),
        ({"phi": 1}, "phi"),
        ({"spl_type": "other"}, "spl_type"),
    ],
)
def test_invalid_parameters_fail(kwargs, message):
    X, y = _data()
    with pytest.raises(ValueError, match=message):
        _estimator(**kwargs).fit(X, y)


def test_rejects_unsupported_weight_and_bad_model_and_input():
    X, y = _data()
    with pytest.raises(NotImplementedError, match="sample_weight"):
        _estimator().fit(X, y, sample_weight=np.ones(len(X)))
    with pytest.raises(ValueError, match="one raw score"):
        _estimator(model=torch.nn.Linear(3, 2)).fit(X, y)
    with pytest.raises(ValueError, match="finite"):
        broken = X.copy()
        broken[0, 0] = np.inf
        _estimator().fit(broken, y)
    fitted = _estimator().fit(X, y)
    with pytest.raises(ValueError, match="feature shape"):
        fitted.decision_function(X[:, :2])
    restored = copy.deepcopy(fitted.model_)
    restored.load_state_dict(fitted.model_.state_dict())
    with torch.no_grad():
        values = restored(torch.as_tensor(X)).reshape(-1).numpy()
    np.testing.assert_allclose(values, fitted.decision_function(X))


@pytest.mark.integration
def test_pipeline_resolves_robust_pu_with_prior():
    X, y = _data()
    report = PUPipeline(
        classifier=_estimator(pretrain_epochs=1, episodes=1, batch_size=8),
        prior_estimator=None,
        cv=2,
        random_state=3,
    ).fit_evaluate(X, y)
    assert isinstance(report.final_model, RobustPUClassifier)
    assert report.final_model.predict(X).shape == (len(X),)


@pytest.mark.gpu
def test_cuda_smoke():
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    X, y = _data()
    model = _estimator(device="cuda", pretrain_epochs=1, episodes=1).fit(X, y)
    assert next(model.model_.parameters()).device.type == "cuda"
    assert np.isfinite(model.decision_function(X)).all()
