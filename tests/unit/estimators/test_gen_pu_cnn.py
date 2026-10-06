"""Pixel generators and independent CNN score paths, not author replay."""

import copy
import os
import pickle
from contextlib import nullcontext

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.gen_pu import GenPUClassifier  # noqa: E402

pytestmark = pytest.mark.unit


def problem(channels=3):
    features = np.random.RandomState(7).uniform(-0.9, 0.9, (24, channels, 4, 5)).astype("float32")
    labels = np.r_[np.ones(8, int), np.zeros(16, int)]
    torch.manual_seed(17)
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(channels, 4, 1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.BatchNorm1d(4),
        torch.nn.Dropout(0.2),
    ).requires_grad_(False)
    return features, labels, encoder


def model(encoder, **overrides):
    params = dict(
        class_prior=0.4,
        encoder=encoder,
        hidden_dim=4,
        latent_dim=2,
        max_epochs=1,
        classifier_epochs=2,
        batch_size=8,
        random_state=2,
        device="cpu",
    )
    return GenPUClassifier(**(params | overrides))


def assert_state(before, network):
    for key, value in network.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_basic_six_network_updates_fresh_pn_and_encoder_template_isolation(monkeypatch):
    features, labels, encoder = problem()
    template = copy.deepcopy(encoder.state_dict())
    fitted = model(encoder)
    original_init, original_step = torch.optim.Adam.__init__, torch.optim.Adam.step
    optimizers, updates, gan_state = [], [], []

    def initialize(optimizer, params, *args, **kwargs):
        original_init(optimizer, params, *args, **kwargs)
        optimizers.append(optimizer)
        if len(optimizers) == 6:
            assert_state(template, fitted.encoder_)  # fresh, not a trained discriminator
            gan_state.extend(copy.deepcopy(net.state_dict()) for net in networks()[:5])

    def networks():
        return [
            fitted.positive_discriminator_,
            fitted.negative_discriminator_,
            fitted.unlabeled_discriminator_,
            fitted.positive_generator_,
            fitted.negative_generator_,
            fitted.model_,
        ]

    def step(optimizer, *args, **kwargs):
        index = next(i for i, item in enumerate(optimizers) if item is optimizer)
        params = optimizer.param_groups[0]["params"]
        before = [parameter.detach().clone() for parameter in params]
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in params)
        if index == 5:
            assert fitted.model_.training and fitted.encoder_.training
        result = original_step(optimizer, *args, **kwargs)
        assert any(not torch.equal(a, b) for a, b in zip(before, params, strict=True))
        updates.append(index)
        return result

    monkeypatch.setattr(torch.optim.Adam, "__init__", initialize)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    fitted.fit(features, labels, os_or_ts="ts")
    assert [updates.count(i) for i in range(6)] == [3, 3, 3, 3, 3, 6]
    assert fitted.stage_optimizer_steps_ == dict(
        positive_discriminator=3,
        negative_discriminator=3,
        unlabeled_discriminator=3,
        positive_generator=3,
        negative_generator=3,
        synthetic_pn=6,
    )
    assert fitted.optimizer_steps_ == sum(fitted.stage_optimizer_steps_.values()) == 21
    assert fitted.history_["optimizer_steps"] == [18, 21]
    encoders = [net[0] for net in networks()[:3]] + [fitted.encoder_]
    assert len({next(net.parameters()).data_ptr() for net in encoders + [encoder]}) == 5
    assert all(not torch.equal(net[0].weight, template["0.weight"]) for net in encoders)
    assert all(net[1].num_batches_tracked > 0 for net in encoders)
    for state, net in zip(gan_state, networks()[:5], strict=True):
        assert_state(state, net)  # final PN cannot update GAN parameters or BN
    assert_state(template, encoder)
    assert encoder.training and all(not p.requires_grad for p in encoder.parameters())


@pytest.mark.parametrize("fail", [False, True])
def test_basic_edge_generator_gradients_freeze_bn_dropout_and_restore_mixed_state(fail):
    features, _, encoder = problem()
    network = torch.nn.Sequential(copy.deepcopy(encoder), torch.nn.Linear(4, 1))
    network.requires_grad_(True).train()
    network[0][6].eval()
    next(network.parameters()).requires_grad_(False)
    state = copy.deepcopy(network.state_dict())
    modes = [layer.training for layer in network.modules()]
    flags = [p.requires_grad for p in network.parameters()]
    inputs = torch.tensor(features[:4], requires_grad=True)
    with (
        pytest.raises(RuntimeError, match="injected") if fail else nullcontext(),
        GenPUClassifier._frozen_discriminators([network]),
    ):
        assert all(not layer.training for layer in network.modules())
        assert all(not p.requires_grad for p in network.parameters())
        GenPUClassifier._logits(network, inputs).sum().backward()
        assert inputs.grad is not None and torch.count_nonzero(inputs.grad)
        assert all(p.grad is None for p in network.parameters())
        if fail:
            raise RuntimeError("injected G failure")
    assert_state(state, network)
    assert [layer.training for layer in network.modules()] == modes
    assert [p.requires_grad for p in network.parameters()] == flags


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("output", ["identity", "tanh"])
def test_basic_param_pixel_generator_coordinates_and_explicit_output_domain(channels, output):
    features, labels, encoder = problem(channels)
    original = features.copy()
    fitted = model(encoder, generator_output=output).fit(features, labels)
    for generator in (fitted.positive_generator_, fitted.negative_generator_):
        assert not any(isinstance(layer, torch.nn.Conv2d) for layer in generator.modules())
        samples = fitted._generate(generator, torch.zeros(3, fitted.latent_dim))
        assert samples.shape == (3, channels, 4, 5)
        assert isinstance(generator[-1], torch.nn.Unflatten)
        assert any(isinstance(layer, torch.nn.Tanh) for layer in generator) == (output == "tanh")
        if output == "tanh":
            assert samples.abs().max() <= 1
    assert fitted.generator_variant_ == f"dense_pixel_{output}_NCHW"
    np.testing.assert_array_equal(original, features)


def test_basic_determ_clone_refit_pickle_and_inference_mode_shape_preservation():
    features, labels, encoder = problem()
    fitted = clone(model(encoder)).fit(features, labels, os_or_ts="ts")
    repeated = model(encoder).fit(features, labels, os_or_ts="ts")
    scores = fitted.decision_function(features)
    np.testing.assert_array_equal(scores, repeated.decision_function(features))
    np.testing.assert_array_equal(
        scores, pickle.loads(pickle.dumps(fitted)).decision_function(features)
    )
    fitted.model_.train()
    fitted.encoder_[6].eval()
    modes = [layer.training for layer in fitted.model_.modules()]
    before = copy.deepcopy(fitted.model_.state_dict())
    np.testing.assert_array_equal(scores, fitted.decision_function(features))
    assert [layer.training for layer in fitted.model_.modules()] == modes
    assert_state(before, fitted.model_)
    assert fitted.decision_function(features[:0]).shape == (0,)
    assert fitted.predict_proba(features[:0]).shape == (0, 2)
    with pytest.raises(ValueError, match="input shape"):
        fitted.decision_function(features[:, :, :, :3])
    fitted.fit(features, labels, os_or_ts="ts")
    np.testing.assert_array_equal(scores, fitted.decision_function(features))
    assert fitted.optimizer_steps_ == 21


def test_basic_determ_checkpoint_pn_only_full_gan_cost_and_cpu_snapshot_isolation(tmp_path):
    from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer

    features, labels, encoder = problem()
    fitted = model(encoder, max_epochs=2)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        fitted, features, labels, os_or_ts="ts"
    )
    contexts = [checkpoint.reference()["training_context"] for checkpoint in trajectory.checkpoints]
    assert contexts == [
        dict(stage="synthetic_pn", stage_epoch=1, round_index=None, optimizer_steps=33),
        dict(stage="synthetic_pn", stage_epoch=2, round_index=None, optimizer_steps=36),
    ]
    assert fitted.checkpoint_epoch_count == 2 and fitted.calibration_applied_
    restored = trajectory.checkpoints[-1].restore(device="cpu")
    np.testing.assert_array_equal(
        restored.decision_function(features), fitted.decision_function(features)
    )
    assert all(p.device.type == "cpu" for p in restored.model_.parameters())
    before = restored.decision_function(features)
    with torch.no_grad():
        next(fitted.model_.parameters()).add_(100)
    np.testing.assert_array_equal(
        before, trajectory.checkpoints[-1].restore().decision_function(features)
    )


def test_basic_calibration_changes_only_real_du_pool_and_actual_pool_step_count(monkeypatch):
    features, labels, encoder = problem()
    original_rng = np.random.RandomState
    seen = []

    class RecordingRng:
        def __init__(self, seed):
            self.rng = original_rng(seed)

        def randint(self, *args):
            return self.rng.randint(*args)

        def normal(self, *args, **kwargs):
            return self.rng.normal(*args, **kwargs)

        def choice(self, values, size):
            seen.append(np.array(values))
            return self.rng.choice(values, size)

    monkeypatch.setattr(np.random, "RandomState", RecordingRng)
    fitted = model(encoder).fit(features, labels, os_or_ts="ts")
    assert all(np.array_equal(pool, np.flatnonzero(labels == 1)) for pool in seen[::2])
    assert all(np.array_equal(np.sort(pool), np.arange(len(labels))) for pool in seen[1::2])
    assert fitted.n_loss_unlabeled_ == 24 and fitted.optimizer_steps_ == 21
    seen.clear()
    fitted.fit(features, labels, os_or_ts="os")
    assert all(np.array_equal(pool, np.flatnonzero(labels == 0)) for pool in seen[1::2])
    assert fitted.n_loss_unlabeled_ == 16 and fitted.optimizer_steps_ == 14
    assert not fitted.calibration_applied_


@pytest.mark.parametrize("fault", ["missing", "type", "spatial", "batch", "nan"])
def test_param_edge_bad_encoder_never_flattens_real_images_or_marks_fitted(fault):
    features, labels, encoder = problem()
    if fault == "missing":
        encoder = None
    elif fault == "type":
        encoder = "cnn"
    elif fault == "spatial":
        encoder = torch.nn.Conv2d(3, 4, 1)
    elif fault == "batch":

        class BadBatch(torch.nn.Module):
            def forward(self, values):
                return values.new_zeros((len(values) + 1, 4))

        encoder = BadBatch()
    elif fault == "nan":
        encoder[0].weight.data.fill_(float("nan"))
    fitted = model(encoder)
    with pytest.raises((TypeError, ValueError)):
        fitted.fit(features, labels)
    assert not fitted._is_fitted


@pytest.mark.parametrize("fault", ["range", "output", "dimension", "weight", "prior"])
def test_param_edge_invalid_image_recipe_fails_before_training(fault):
    features, labels, encoder = problem()
    kwargs, fit_kwargs = {}, {}
    if fault == "range":
        features[0, 0, 0, 0] = 1.01
        kwargs["generator_output"] = "tanh"
    elif fault == "output":
        kwargs["generator_output"] = ["tanh"]
    elif fault == "dimension":
        features = features[:, 0]
    elif fault == "weight":
        fit_kwargs["sample_weight"] = np.ones(len(labels))
    else:
        kwargs["class_prior"] = None
    fitted = model(encoder, **kwargs)
    with pytest.raises((ValueError, NotImplementedError)):
        fitted.fit(features, labels, **fit_kwargs)
    assert not fitted._is_fitted
    assert not hasattr(fitted, "positive_generator_")


def test_edge_singleton_bn_training_and_prediction_remain_finite():
    features, labels, encoder = problem()
    fitted = model(encoder, batch_size=1, classifier_epochs=1).fit(features[:9], labels[:9])
    assert np.isfinite(fitted.predict_proba(features[:1])).all()
    assert fitted.optimizer_steps_ == 6
    assert fitted.encoder_[5].num_batches_tracked == 0


def test_edge_generator_failure_restores_discriminators_without_retry_or_success(monkeypatch):
    features, labels, encoder = problem()
    fitted = model(encoder)
    generate = GenPUClassifier._generate
    calls = []

    def fail(self, network, latent):
        calls.append(network)
        if len(calls) == 3:  # first generator call in frozen-D update
            raise TypeError("injected generator error")
        return generate(self, network, latent)

    monkeypatch.setattr(GenPUClassifier, "_generate", fail)
    with pytest.raises(TypeError, match="injected generator error"):
        fitted.fit(features, labels)
    assert len(calls) == 3 and fitted.optimizer_steps_ == 3
    assert not fitted._is_fitted and not hasattr(fitted, "model_")
    for net in (
        fitted.positive_discriminator_,
        fitted.negative_discriminator_,
        fitted.unlabeled_discriminator_,
    ):
        assert all(p.requires_grad for p in net.parameters()) and net.training


@pytest.mark.gpu
def test_basic_cuda_native_image_smoke_requires_real_device_when_requested(tmp_path):
    if not torch.cuda.is_available():
        if os.getenv("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 but CUDA unavailable")
        pytest.skip("CUDA unavailable; CPU evidence is not GPU evidence")
    from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer

    features, labels, encoder = problem()
    fitted = model(encoder, device="cuda")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, features, labels)
    assert all(p.is_cuda for p in fitted.model_.parameters())
    assert all(p.is_cuda for p in fitted.positive_generator_.parameters())
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(features),
        fitted.decision_function(features),
        rtol=1e-5,
        atol=1e-5,
    )
