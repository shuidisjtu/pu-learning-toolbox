"""GradPU raw-image gradients, explicit no-BN encoders and inference snapshots."""

import copy
import os
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox import GradPUClassifier, build_encoder  # noqa: E402
from pu_toolbox.estimators.deep.grad_pu import gradpu_objective  # noqa: E402
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
        torch.nn.Conv2d(3, 3, 1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
    )


def model(**kwargs):
    params = dict(
        encoder=encoder(), hidden_dim=4, max_epochs=2, batch_size=8, random_state=3, device="cpu"
    )
    params.update(kwargs)
    return GradPUClassifier(**params)


def test_basic_cnn_updates_encoder_and_interpolates_original_images(monkeypatch):
    features, labels = data()
    source = encoder().requires_grad_(False)
    before = copy.deepcopy(source.state_dict())
    seen = []
    original = torch.autograd.grad

    def track(outputs, inputs, *args, **kwargs):
        seen.append((tuple(inputs.shape), inputs.requires_grad, kwargs.get("create_graph")))
        return original(outputs, inputs, *args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", track)
    fitted = model(encoder=source).fit(features, labels, os_or_ts="ts")
    assert seen == [((16, 3, 4, 4), True, True)] * 4
    assert fitted.encoder_ is not source and fitted.encoder_[0].weight.requires_grad
    assert not torch.equal(fitted.encoder_[0].weight, before["0.weight"])
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not source[0].weight.requires_grad
    assert fitted.optimizer_steps_ == 4 and fitted.n_loss_unlabeled_ == 24
    assert fitted.calibration_applied_
    assert all(loss > 0 for loss in fitted.history_["gradient_penalty"])


@pytest.mark.math
def test_input_gradient_norm_matches_linear_image_equation_and_backpropagates_conv_weights():
    conv = torch.nn.Conv2d(3, 1, 1, bias=False).double()
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).reshape(1, 3, 1, 1))
    images = torch.zeros((2, 3, 4, 4), dtype=torch.float64, requires_grad=True)
    raw = conv(images).mean(dim=(1, 2, 3))
    _, parts = gradpu_objective(
        raw, raw, beta=0.0, alpha=1.0, interpolated_inputs=images, raw_interpolated=raw
    )
    torch.testing.assert_close(
        parts["gradient_penalty"], torch.tensor(14 / 16, dtype=torch.float64)
    )
    parts["gradient_penalty"].backward()
    torch.testing.assert_close(
        conv.weight.grad.reshape(-1), torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64) / 16
    )


def test_determ_clone_seed_pickle_and_epoch_weights_roundtrip(tmp_path):
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
    assert len(trajectory.checkpoints) == 2 and np.all(np.abs(expected) <= 1)


def test_edge_empty_prediction_space_shape_no_flatten_and_batch_bounds():
    features, labels = data()
    fitted = model().fit(features, labels)
    assert fitted.decision_function(features[:0]).shape == (0,)
    fitted.model_.train()
    fitted.decision_function(features)
    assert fitted.model_.training
    with pytest.raises(ValueError, match="shape"):
        fitted.predict(features[:, :, :2])
    with pytest.raises(ValueError, match="encoder"):
        model(encoder=None).fit(features, labels)


@pytest.mark.parametrize("bad", ["not_module", torch.nn.Flatten(start_dim=0)])
def test_param_invalid_encoder_contract_is_rejected(bad):
    features, labels = data()
    with pytest.raises((TypeError, ValueError), match="encoder"):
        model(encoder=bad).fit(features, labels)


def test_basic_explicit_no_bn_backbone_does_not_rewrite_default():
    normal = build_encoder("cnn", backbone="cnn13", in_channels=3)
    no_bn = build_encoder("cnn", backbone="cnn13_no_bn", in_channels=3)
    assert any(isinstance(layer, torch.nn.BatchNorm2d) for layer in normal.modules())
    assert not any(
        isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) for layer in no_bn.modules()
    )
    features, labels = data()
    with pytest.raises(ValueError, match="BatchNorm"):
        model(encoder=normal).fit(features, labels)


@pytest.mark.gpu
def test_cuda_second_order_cnn_training_and_cpu_checkpoint(tmp_path):
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    features, labels = data()
    fitted = model(max_epochs=1, device="cuda")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, features, labels)
    assert next(fitted.encoder_.parameters()).is_cuda
    assert fitted.history_["gradient_penalty"][0] > 0
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(features),
        fitted.decision_function(features),
        atol=1e-5,
        rtol=1e-5,
    )
