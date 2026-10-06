"""Holistic-PU end-to-end images, stable U identities and stage snapshots."""

import copy
import os
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.holistic_pu import HolisticPUClassifier  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


def images():
    values = np.random.RandomState(7).normal(size=(27, 3, 4, 4)).astype("float32")
    values[:10] += 1
    return values, np.r_[np.ones(10, int), np.zeros(17, int)]


def encoder():
    torch.manual_seed(17)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.BatchNorm1d(4),
    )


def model(**kwargs):
    params = dict(
        encoder=encoder(),
        hidden_dim=4,
        warmup_epochs=3,
        max_epochs=1,
        batch_size=8,
        random_state=3,
        device="cpu",
    )
    params.update(kwargs)
    return HolisticPUClassifier(**params)


def test_basic_encoder_updates_in_both_stages_without_mutating_template():
    features, labels = images()
    source = encoder().requires_grad_(False)
    saved = copy.deepcopy(source.state_dict())
    at_stage_end = {}

    def capture(epoch, fitted):
        at_stage_end[fitted.checkpoint_stage_] = fitted.encoder_[0].weight.detach().clone()
        assert fitted.encoder_[0].weight.grad is not None
        assert torch.isfinite(fitted.encoder_[0].weight.grad).all()

    fitted = model(encoder=source).fit(features, labels, epoch_callback=capture)
    assert fitted.encoder_ is not source
    assert fitted.encoder_[0].weight.requires_grad
    assert not torch.equal(at_stage_end["warmup"], saved["0.weight"])
    assert not torch.equal(at_stage_end["pseudo_pn"], at_stage_end["warmup"])
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    assert not source[0].weight.requires_grad and source.training
    # 17 U rows: two full P/U batches and a singleton tail each warmup.
    # Only full batches update BN; scanning/probing must not update its buffers.
    assert fitted.encoder_[1].num_batches_tracked.item() == 16
    assert fitted.encoder_[5].num_batches_tracked.item() == 16
    assert fitted.optimizer_steps_ == 13
    assert fitted.prediction_trajectory_.shape == (17, 3)
    np.testing.assert_array_equal(fitted.pseudo_label_indices_, np.flatnonzero(labels == 0))
    assert fitted.training_view_ == "os" and not fitted.calibration_applied_


def test_determ_seed_clone_pickle_refit_and_all_stage_checkpoint_recovery(tmp_path):
    features, labels = images()
    fitted = clone(model())
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, features, labels)
    expected = fitted.decision_function(features)
    repeated = clone(model()).fit(features, labels, class_prior=0.4)
    np.testing.assert_array_equal(repeated.decision_function(features), expected)
    np.testing.assert_array_equal(repeated.pseudo_labels_, fitted.pseudo_labels_)
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(fitted)).decision_function(features), expected
    )
    assert fitted.encoder_[0].weight.data_ptr() != repeated.encoder_[0].weight.data_ptr()
    assert len(trajectory.checkpoints) == fitted.checkpoint_epoch_count == 4
    references = [checkpoint.reference() for checkpoint in trajectory.checkpoints]
    assert [ref["training_context"]["stage"] for ref in references] == ["warmup"] * 3 + [
        "pseudo_pn"
    ]
    assert [ref["training_context"]["optimizer_steps"] for ref in references] == [3, 6, 9, 13]
    assert all(ref["training_resume_supported"] is False for ref in references)
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(features), expected
    )
    first = trajectory.checkpoints[0].restore(device="cpu").decision_function(features)
    fitted.fit(features, labels)
    assert fitted.optimizer_steps_ == 13
    np.testing.assert_array_equal(fitted.decision_function(features), expected)
    np.testing.assert_array_equal(
        trajectory.checkpoints[0].restore(device="cpu").decision_function(features), first
    )


def test_edge_empty_and_changed_input_shape_fail_closed_without_flattening():
    features, labels = images()
    fitted = model().fit(features, labels)
    assert fitted.decision_function(features[:0]).shape == (0,)
    assert fitted.predict_proba(features[:0]).shape == (0, 2)
    with pytest.raises(ValueError, match="input shape"):
        fitted.predict(features[:, :, :2])
    with pytest.raises(ValueError, match="explicit encoder"):
        model(encoder=None).fit(features, labels)
    with pytest.raises(ValueError, match="2-D features or 4-D"):
        model().fit(features[:, 0], labels)


@pytest.mark.parametrize("invalid", ["not-module", torch.nn.Identity(), torch.nn.Flatten(0)])
def test_param_invalid_encoder_contract_keeps_estimator_unfitted(invalid):
    features, labels = images()
    fitted = model(encoder=invalid)
    with pytest.raises((TypeError, ValueError), match="encoder"):
        fitted.fit(features, labels)
    assert not fitted._is_fitted


def test_basic_batched_prediction_and_trajectory_scan_preserve_bn_and_layer_modes():
    features, labels = images()
    fitted = model().fit(features, labels)
    fitted.model_.train()
    fitted.encoder_[1].eval()
    modes = [layer.training for layer in fitted.model_.modules()]
    before = copy.deepcopy(fitted.encoder_.state_dict())
    sizes = []
    hook = fitted.encoder_.register_forward_pre_hook(
        lambda module, args: sizes.append(len(args[0]))
    )
    try:
        expected = fitted.decision_function(features)
        np.testing.assert_array_equal(fitted.decision_function(features), expected)
    finally:
        hook.remove()
    assert sizes == [8, 8, 8, 3] * 2
    assert [layer.training for layer in fitted.model_.modules()] == modes
    for key, value in fitted.encoder_.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_edge_forward_error_restores_prediction_modes_and_refuses_bad_batch_identity():
    features, labels = images()
    fitted = model().fit(features, labels)
    fitted.model_.train()
    fitted.encoder_[1].eval()
    modes = [layer.training for layer in fitted.model_.modules()]
    handle = fitted.encoder_.register_forward_hook(lambda module, args, result: result[:0])
    try:
        with pytest.raises(ValueError, match="one logit per input row"):
            fitted.decision_function(features)
    finally:
        handle.remove()
    assert [layer.training for layer in fitted.model_.modules()] == modes


@pytest.mark.gpu
def test_cuda_native_cnn_snapshot_roundtrip(tmp_path):
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    features, labels = images()
    fitted = model(device="cuda")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, features, labels)
    assert next(fitted.encoder_.parameters()).is_cuda
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(features),
        fitted.decision_function(features),
        atol=1e-5,
        rtol=1e-5,
    )
