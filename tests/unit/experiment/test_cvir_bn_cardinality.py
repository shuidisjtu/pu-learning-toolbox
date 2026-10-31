"""CVIR shared ResNet BN safety depends on N*spatial, not N alone."""

# ruff: noqa: E402

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.experiment.candidate_image import build_candidate_image_classifier
from pu_toolbox.experiment.candidate_image_runner import CandidateImageCheckpointTrainer
from pu_toolbox.experiment.cvir_adapter import cvir_risk_prior
from pu_toolbox.experiment.feature_adapter import encoder_state_sha256
from pu_toolbox.experiment.image import fit_survey_image_preprocessing

pytestmark = pytest.mark.unit


def setup(size, **parameters):
    raw = np.random.RandomState(17).randint(0, 256, (12, 1, size, size)).astype(np.uint8)
    pixels, preprocessing = fit_survey_image_preprocessing(
        raw, dataset="mnist", train_augmentation="none"
    )
    model, record = build_candidate_image_classifier("cvir", preprocessing, seed=3, device="cpu")
    model.set_params(
        **(dict(max_epochs=1, warm_start_epochs=1, batch_size=3, hidden_dim=4) | parameters)
    )
    return model, record, pixels, np.r_[np.ones(4, int), np.zeros(8, int)]


def trainer(record, path):
    return CandidateImageCheckpointTrainer(
        record,
        checkpoint_dir=path,
        checkpoint_bytes_per_component=64 * 2**20,
        population_prior_source="synthetic population",
    )


@pytest.mark.parametrize("size,values", [(8, 1), (16, 4), (28, 16), (32, 16), (96, 9)])
def test_basic_determ_shape_evidence_preserves_template_bn_buffers_and_global_rng(size, values):
    before = torch.random.get_rng_state().clone()
    model, record, _, _ = setup(size)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert record["minimum_bn_values_per_single_image"] == values
    assert record["bn_shape_probe_scope"].startswith("private_CPU_eval")
    assert encoder_state_sha256(model.encoder) == record["encoder_initial_state_sha256"]
    for module in model.encoder.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert module.training and int(module.num_batches_tracked) == 0
            torch.testing.assert_close(module.running_mean, torch.zeros_like(module.running_mean))


@pytest.mark.parametrize("view", ["os", "ts"])
def test_basic_determ_safe_singleton_training_keeps_native_bn_and_fit_exactly(tmp_path, view):
    model, record, pixels, labels = setup(28)
    alpha = cvir_risk_prior(0.5, n_samples=len(labels), n_labeled=4, os_or_ts=view)
    plain = (
        copy.deepcopy(model)
        .set_params(unlabeled_positive_prior=alpha, prior_source="synthetic population")
        .fit(pixels, labels, os_or_ts=view)
    )
    trajectory = trainer(record, tmp_path).fit(
        model, pixels, labels, class_prior=0.5, os_or_ts=view
    )
    assert model.history_ == plain.history_ and model.optimizer_steps_ == plain.optimizer_steps_
    for key, value in model.model_.state_dict().items():
        torch.testing.assert_close(value, plain.model_.state_dict()[key], rtol=0, atol=0)
    for module in model.encoder_.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert int(module.num_batches_tracked) > 0
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(pixels),
        model.decision_function(pixels),
    )


def test_edge_one_value_per_channel_still_rejected_before_fit(tmp_path, monkeypatch):
    model, record, pixels, labels = setup(8)
    monkeypatch.setattr(type(model), "fit", lambda *a, **k: pytest.fail("unsafe BN fit"))
    with pytest.raises(ValueError, match="BN"):
        trainer(record, tmp_path).fit(model, pixels, labels, class_prior=0.5, os_or_ts="os")
    assert not list(tmp_path.rglob("*.pt"))


@pytest.mark.parametrize("value", [None, 0, True])
def test_param_missing_or_invalid_bn_evidence_rejected_before_fit(tmp_path, monkeypatch, value):
    model, record, pixels, labels = setup(28)
    record["minimum_bn_values_per_single_image"] = value
    monkeypatch.setattr(type(model), "fit", lambda *a, **k: pytest.fail("invalid BN evidence fit"))
    with pytest.raises(ValueError):
        trainer(record, tmp_path).fit(model, pixels, labels, class_prior=0.5, os_or_ts="os")
