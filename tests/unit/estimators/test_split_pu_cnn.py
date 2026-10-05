"""Split-PU real CNN feature distillation, independent stages and replay."""

import copy
import os
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox import SplitPUClassifier  # noqa: E402
from pu_toolbox.estimators.deep.split_pu import _features, _network, _score  # noqa: E402
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
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(3, 4, 1),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )


def model(**kwargs):
    params = dict(
        class_prior=0.4,
        encoder=encoder(),
        hidden_dim=4,
        teacher_epochs=1,
        split_epochs=1,
        student_epochs=1,
        rounds=2,
        batch_size=8,
        random_state=3,
        device="cpu",
    )
    params.update(kwargs)
    return SplitPUClassifier(**params)


def test_basic_cnn_stages_are_independent_trainable_copies_with_frozen_teacher():
    features, labels = data()
    template = encoder().requires_grad_(False)
    before = copy.deepcopy(template.state_dict())
    fitted = model(encoder=template).fit(features, labels, os_or_ts="ts")
    assert fitted.encoder_feature_layer_ == "3"  # first spatial pooling output
    stages = (fitted.teacher_encoder_, fitted.splitter_encoder_, fitted.encoder_)
    assert len({id(stage) for stage in (*stages, template)}) == 4
    for stage in stages:
        assert not torch.equal(stage[0].weight, before["0.weight"])
        assert not stage[0]._forward_hooks
    assert not fitted.teacher_encoder_[0].weight.requires_grad
    assert (
        fitted.splitter_encoder_[0].weight.requires_grad and fitted.encoder_[0].weight.requires_grad
    )
    assert not template[0].weight.requires_grad
    for key, value in template.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert fitted.n_easy_ + fitted.n_hard_ == 16
    assert fitted.n_loss_unlabeled_ == 24 and fitted.calibration_applied_


@pytest.mark.math
def test_low_feature_is_real_convolution_output_with_gradients_and_no_hook_leak():
    features, _ = data()
    network = _network(4, 4, encoder=encoder(), feature_layer="0").eval()
    batch = torch.as_tensor(features[:3])
    expected_low = network[0][0](batch).clone()
    expected_high = network[0](batch)
    low, high = _features(network, batch)
    torch.testing.assert_close(low, expected_low)
    torch.testing.assert_close(high, expected_high)
    assert low.shape == (3, 3, 4, 4) and high.shape == (3, 4)
    assert (low < 0).any()  # not silently overwritten by in-place ReLU
    (low.square().mean() + high.square().mean()).backward()
    assert network[0][0].weight.grad is not None
    assert network[0][0].weight.grad.abs().sum() > 0
    assert not network[0][0]._forward_hooks


def test_edge_singleton_bn_uses_running_stats_but_keeps_parameter_gradients():
    features, _ = data()
    source = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 1),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.BatchNorm2d(4),
        torch.nn.Flatten(),
    )
    network = _network(4, 4, encoder=source, feature_layer="0").train()
    norm = network[0][2]
    before = copy.deepcopy(norm.state_dict())
    batch = torch.as_tensor(features[:1])
    low, high = _features(network, batch)
    (_score(network, batch).sum() + low.square().mean() + high.square().mean()).backward()
    assert network.training and norm.training
    assert norm.weight.grad is not None and network[0][0].weight.grad is not None
    for key, value in norm.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    _score(network, torch.as_tensor(features[:2]))
    assert norm.num_batches_tracked.item() == 1
    bad = torch.nn.Sequential(torch.nn.BatchNorm2d(3, track_running_stats=False))
    with pytest.raises(ValueError, match="more than 1 value"):
        _score(bad, torch.ones(1, 3, 1, 1))
    assert bad.training and bad[0].training  # even a failed forward restores modes


def test_param_reused_feature_layer_rejected_and_hook_removed_on_failure():
    class ReusedConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 1)

        def forward(self, batch):
            result = self.conv(self.conv(batch))
            return result.mean(dim=(2, 3))

    features, _ = data()
    network = _network(3, 4, encoder=ReusedConv(), feature_layer="conv")
    with pytest.raises(ValueError, match="exactly once"):
        _features(network, torch.as_tensor(features[:2]))
    assert not network[0].conv._forward_hooks


def test_determ_seed_clone_pickle_and_all_stage_snapshots(tmp_path):
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
    assert len(trajectory.checkpoints) == fitted.checkpoint_epoch_count == 4
    for checkpoint in trajectory.checkpoints:
        scores = checkpoint.restore(device="cpu").decision_function(features)
        assert scores.shape == (24,) and np.isfinite(scores).all()
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(features),
        expected,
        atol=1e-7,
    )


def test_edge_prediction_shape_empty_and_bn_mode_isolation():
    features, labels = data()
    fitted = model().fit(features, labels)
    before = copy.deepcopy(fitted.encoder_.state_dict())
    assert fitted.model_.training
    assert fitted.decision_function(features[:0]).shape == (0,)
    assert fitted.decision_function(features).shape == (24,)
    assert fitted.model_.training
    for key, value in fitted.encoder_.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="feature shape"):
        fitted.predict(features[:, :, :2])
    with pytest.raises(ValueError, match="encoder"):
        model(encoder=None).fit(features, labels)


@pytest.mark.parametrize(
    "params",
    [
        {"encoder": "bad"},
        {"encoder_feature_layer": "absent"},
        {"encoder_feature_layer": ""},
        {"encoder": torch.nn.Flatten()},
        {"encoder": torch.nn.Flatten(start_dim=0), "encoder_feature_layer": "not_a_layer"},
    ],
)
def test_param_invalid_encoder_or_layer_is_refused(params):
    features, labels = data()
    with pytest.raises((TypeError, ValueError), match="encoder"):
        model(**params).fit(features, labels)


@pytest.mark.gpu
def test_cuda_cnn_ts_cpu_pool_and_checkpoint_replay(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    import pu_toolbox.estimators.deep.split_pu as implementation

    features, labels = data()
    observed = []
    original = implementation._batched_score

    def capture(network, pool, batch_size, device):
        observed.append(pool.device.type)
        return original(network, pool, batch_size, device)

    monkeypatch.setattr(implementation, "_batched_score", capture)
    fitted = model(device="cuda")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        fitted, features, labels, os_or_ts="ts"
    )
    assert next(fitted.encoder_.parameters()).is_cuda
    assert observed and set(observed) == {"cpu"}
    assert fitted.n_loss_unlabeled_ == 24 and fitted.n_easy_ + fitted.n_hard_ == 16
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(features),
        fitted.decision_function(features),
        atol=1e-5,
        rtol=1e-5,
    )
