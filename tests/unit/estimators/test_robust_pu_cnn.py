"""Robust-PU CNN stages, stable row weights and fold-safe inference snapshots."""

import copy
import os
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox import RobustPUClassifier  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


def data():
    return (
        np.random.RandomState(5).normal(size=(24, 3, 4, 4)).astype("float32"),
        np.r_[np.ones(8, int), np.zeros(16, int)],
    )


def encoder():
    torch.manual_seed(11)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 1),
        torch.nn.BatchNorm2d(3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )


def model(**kwargs):
    params = dict(
        class_prior=0.4,
        encoder=encoder(),
        hidden_dim=4,
        pretrain_epochs=1,
        episodes=1,
        inner_epochs=1,
        batch_size=8,
        random_state=3,
        device="cpu",
    )
    params.update(kwargs)
    return RobustPUClassifier(**params)


def test_basic_cnn_updates_both_stages_without_mutating_encoder_template_or_bn_probes():
    features, labels = data()
    source = encoder().requires_grad_(False)
    before = copy.deepcopy(source.state_dict())
    fitted = model(encoder=source).fit(features, labels, os_or_ts="ts")
    assert fitted.encoder_ is not source and fitted.encoder_[0].weight.requires_grad
    assert not torch.equal(fitted.encoder_[0].weight, before["0.weight"])
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not source[0].weight.requires_grad
    assert fitted.encoder_[1].num_batches_tracked.item() == 7  # 2 P/U pairs + 3 episode batches
    assert fitted.optimizer_steps_ == 5
    assert fitted.calibration_applied_ and fitted.n_loss_unlabeled_ == 24
    assert len(fitted.history_["pretrain_risk"]) == len(fitted.history_["episode_loss"]) == 1


def test_determ_clone_seed_pickle_and_stage_snapshots_roundtrip(tmp_path):
    features, labels = data()
    fitted = clone(model())
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, features, labels)
    expected = fitted.decision_function(features)
    np.testing.assert_array_equal(
        clone(model()).fit(features, labels).decision_function(features), expected
    )
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(fitted)).decision_function(features), expected
    )
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore().decision_function(features), expected
    )
    assert len(trajectory.checkpoints) == fitted.checkpoint_epoch_count == 2


def test_edge_spatial_shape_empty_prediction_batch_bounds_and_mode_restore():
    features, labels = data()
    fitted = model().fit(features, labels)
    assert fitted.decision_function(features[:0]).shape == (0,)
    fitted.model_.train()
    batches = []
    before = copy.deepcopy(fitted.encoder_.state_dict())
    handle = fitted.model_.register_forward_pre_hook(
        lambda module, args: batches.append(len(args[0]))
    )
    try:
        fitted.decision_function(features)
    finally:
        handle.remove()
    assert batches == [8, 8, 8] and fitted.model_.training
    for key, value in fitted.encoder_.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="shape"):
        fitted.predict(features[:, :, :2])
    with pytest.raises(ValueError, match="encoder"):
        model(encoder=None).fit(features, labels)


@pytest.mark.parametrize("bad", ["not_module", torch.nn.Flatten(start_dim=0)])
def test_param_invalid_encoder_contract_refused(bad):
    features, labels = data()
    with pytest.raises((TypeError, ValueError), match="encoder"):
        model(encoder=bad).fit(features, labels)


def test_param_cnn_ts_still_requires_actual_nnpu_warmup():
    features, labels = data()
    with pytest.raises(ValueError, match="pretrain_epochs"):
        model(pretrain_epochs=0).fit(features, labels, os_or_ts="ts")
    fitted = model(pretrain_epochs=0).fit(features, labels, os_or_ts="os")
    assert not fitted.calibration_applied_ and fitted.optimizer_steps_ == 3


@pytest.mark.gpu
def test_cuda_cnn_batch_residency_weights_and_cpu_checkpoint(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    import pu_toolbox.estimators.deep.robust_pu as implementation

    features, labels = data()
    observed = []
    original = implementation._episode_weights

    def capture(raw, labels, **kwargs):
        observed.append((raw.device.type, labels.device.type, len(raw)))
        return original(raw, labels, **kwargs)

    monkeypatch.setattr(implementation, "_episode_weights", capture)
    fitted = model(device="cuda")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        fitted, features, labels, os_or_ts="ts"
    )
    assert observed == [("cpu", "cpu", 24)]
    assert next(fitted.encoder_.parameters()).is_cuda
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(features),
        fitted.decision_function(features),
        atol=1e-5,
        rtol=1e-5,
    )
