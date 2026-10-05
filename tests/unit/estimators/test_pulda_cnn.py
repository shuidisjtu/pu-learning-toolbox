"""PULDA injected CNN training, batch residency and two-stage snapshots."""

import copy
import os
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox import PULDAClassifier  # noqa: E402
from pu_toolbox.core.exceptions import ValidationError  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


def images():
    values = np.random.RandomState(5).normal(size=(24, 3, 4, 4)).astype("float32")
    return values, np.r_[np.ones(8, int), np.zeros(16, int)]


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
        depth=1,
        warmup_epochs=1,
        pu_epochs=1,
        positive_batch_size=4,
        unlabeled_batch_size=8,
        random_state=3,
        device="cpu",
    )
    params.update(kwargs)
    return PULDAClassifier(**params)


def test_basic_cnn_weights_update_without_mutating_frozen_template_and_bn_probes():
    features, labels = images()
    source = encoder().requires_grad_(False)
    before = copy.deepcopy(source.state_dict())
    fitted = model(encoder=source).fit(features, labels, os_or_ts="ts")
    assert fitted.encoder_ is not source and fitted.encoder_[0].weight.requires_grad
    assert not torch.equal(fitted.encoder_[0].weight, before["0.weight"])
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not source[0].weight.requires_grad
    # Warmup: 2 optimization forwards; PU: 2 original + 2 mixed forwards.
    # Probe, pseudo-label refresh and prediction must not update BN buffers.
    assert fitted.encoder_[1].num_batches_tracked.item() == 6
    assert fitted.optimizer_steps_ == 4
    assert fitted.calibration_applied_ and fitted.n_loss_unlabeled_ == 24
    assert fitted.history_["phase"] == ["warmup", "pu_mixup"]
    assert fitted.pseudo_labels_.shape == (24,)


def test_determ_seed_clone_pickle_and_both_stage_classifier_checkpoint_recovery(tmp_path):
    features, labels = images()
    fitted = clone(model())
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, features, labels)
    expected = fitted.decision_function(features)
    np.testing.assert_array_equal(
        clone(model()).fit(features, labels).decision_function(features), expected
    )
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(fitted)).decision_function(features), expected
    )
    assert len(trajectory.checkpoints) == fitted.checkpoint_epoch_count == 2
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore().decision_function(features), expected
    )


def test_edge_empty_prediction_wrong_spatial_shape_and_no_silent_flattening():
    features, labels = images()
    fitted = model().fit(features, labels)
    assert fitted.predict_proba(features[:0]).shape == (0, 2)
    with pytest.raises(ValueError, match="input shape"):
        fitted.predict(features[:, :, :2])
    with pytest.raises(ValueError, match="encoder"):
        model(encoder=None).fit(features, labels)
    with pytest.raises(ValidationError, match="2-D or 4-D"):
        model().fit(features[:, 0], labels)


@pytest.mark.parametrize("bad", ["not_module", torch.nn.Flatten(start_dim=0)])
def test_param_encoder_module_and_feature_contract_refused(bad):
    features, labels = images()
    with pytest.raises((TypeError, ValueError), match="encoder"):
        model(encoder=bad).fit(features, labels)


def test_basic_prediction_batches_are_bounded_and_restore_training_mode():
    features, labels = images()
    fitted = model().fit(features, labels)
    fitted.model_.train()
    before = copy.deepcopy(fitted.encoder_.state_dict())
    batches = []
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


@pytest.mark.gpu
def test_gpu_training_keeps_image_storage_on_cpu_and_restores_cpu_snapshot(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    features, labels = images()
    original = PULDAClassifier._train_epoch
    residency = []

    def capture(self, data, targets, *args, **kwargs):
        residency.append((data.device.type, targets.device.type))
        return original(self, data, targets, *args, **kwargs)

    monkeypatch.setattr(PULDAClassifier, "_train_epoch", capture)
    fitted = model(device="cuda")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, features, labels)
    assert residency == [("cpu", "cpu")] * 2
    assert next(fitted.encoder_.parameters()).is_cuda
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(features),
        fitted.decision_function(features),
        atol=1e-5,
        rtol=1e-5,
    )
