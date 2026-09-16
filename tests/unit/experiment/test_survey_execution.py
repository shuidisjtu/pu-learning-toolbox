# ruff: noqa: N803, N806
"""Image assembly, cache integrity and shared score blueprint tests."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart
from pu_toolbox.experiment.feature_adapter import _encoder_state_sha256
from pu_toolbox.experiment.strategies import SARLBEAGenerator
from pu_toolbox.experiment.survey_execution import (
    PilotOracleMLP,
    SourceSpaceGenerator,
    assemble_model,
    cached_adapter,
    prepare_image_bundle,
    validate_score_model,
)
from pu_toolbox.experiment.survey_protocol import ROLES, load_protocol, resolve_unit

pytestmark = pytest.mark.unit


def image_bundle():
    rng = np.random.RandomState(9)
    parts = {}
    for index, role in enumerate(ROLES):
        X = rng.randint(0, 256, size=(4, 3, 32, 32), dtype=np.uint8)
        parts[role] = DatasetPart(
            X=X,
            labels=np.array([1, 1, 0, 0]),
            view="clean",
            indices=np.arange(index * 4, index * 4 + 4),
            for_selection=role != "test",
        )
    return DatasetBundle(**parts)


def tiny_encoder():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        return nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())


def test_basic_image_statistics_and_encoder_are_train_only():
    source = image_bundle()
    prepared, encoder, manifest = prepare_image_bundle(source, load_protocol(), 0)
    assert prepared.train.X.dtype == np.float32
    assert manifest["normalization"]["source"] == "train_only_channel_statistics"
    assert manifest["augmentation"]["train"]["name"] == "none"
    assert manifest["backbone"]["weights"] is None
    assert _encoder_state_sha256(encoder) == manifest["encoder_state_sha256"]


def test_determ_encoder_seed_and_train_statistics_ignore_test_changes():
    source = image_bundle()
    _, first, manifest = prepare_image_bundle(source, load_protocol(), 0)
    changed = replace(source, test=replace(source.test, X=np.zeros_like(source.test.X)))
    _, second, other = prepare_image_bundle(changed, load_protocol(), 0)
    assert manifest == other
    assert _encoder_state_sha256(first) == _encoder_state_sha256(second)


def test_basic_adapter_cache_reuses_features_and_keeps_labels(tmp_path):
    source = image_bundle()
    encoder = tiny_encoder()
    before = _encoder_state_sha256(encoder)
    first, one = cached_adapter(source, encoder, {"test_spec": True}, cache_dir=tmp_path)
    second, two = cached_adapter(source, encoder, {"test_spec": True}, cache_dir=tmp_path)
    assert not one["cache_hit"] and two["cache_hit"]
    assert one["representation_sha256"] == two["representation_sha256"]
    assert _encoder_state_sha256(encoder) == before
    for role in ROLES:
        np.testing.assert_array_equal(getattr(first, role).X, getattr(second, role).X)
        np.testing.assert_array_equal(getattr(second, role).labels, getattr(source, role).labels)


def test_edge_cache_keys_do_not_depend_on_labels(tmp_path):
    source = image_bundle()
    _, first = cached_adapter(source, tiny_encoder(), {"test_spec": True}, cache_dir=tmp_path)
    changed = replace(source, train=replace(source.train, labels=1 - source.train.labels))
    adapted, other = cached_adapter(
        changed, tiny_encoder(), {"test_spec": True}, cache_dir=tmp_path
    )
    assert first["cache_key"] == other["cache_key"]
    assert other["cache_hit"]
    np.testing.assert_array_equal(adapted.train.labels, changed.train.labels)


def test_param_damaged_cache_rejected(tmp_path):
    source = image_bundle()
    _, manifest = cached_adapter(source, tiny_encoder(), {"test_spec": True}, cache_dir=tmp_path)
    path = tmp_path / manifest["cache_key"] / "adapter.json"
    payload = json.loads(path.read_text())
    payload["feature_sha256"]["train"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="feature hash"):
        cached_adapter(source, tiny_encoder(), {"test_spec": True}, cache_dir=tmp_path)


def test_determ_cache_changes_when_input_or_encoder_changes(tmp_path):
    source = image_bundle()
    _, first = cached_adapter(source, tiny_encoder(), {"test_spec": True}, cache_dir=tmp_path)
    encoder = tiny_encoder()
    with torch.no_grad():
        encoder[0].weight.add_(1)
    _, other = cached_adapter(source, encoder, {"test_spec": True}, cache_dir=tmp_path)
    assert first["cache_key"] != other["cache_key"]


def test_determ_sar_marking_is_generated_in_original_space(tmp_path):
    source = image_bundle()
    adapted, _ = cached_adapter(source, tiny_encoder(), {"test_spec": True}, cache_dir=tmp_path)
    wrapper = SourceSpaceGenerator(SARLBEAGenerator(), source, adapted)
    labels, metadata = wrapper.generate(adapted.train.X, source.train.labels, 0.5, 3)
    reference, _ = SARLBEAGenerator().generate(source.train.X, source.train.labels, 0.5, 3)
    np.testing.assert_array_equal(labels, reference)
    assert metadata["sampling_space"] == "original_input"
    with pytest.raises(ValueError, match="unbound"):
        wrapper.generate(adapted.test.X, source.test.labels, 0.5, 3)


def test_param_non_protocol_image_size_rejected():
    source = image_bundle()
    source = DatasetBundle(
        **{
            role: replace(getattr(source, role), X=getattr(source, role).X[:, :, :8, :8])
            for role in ROLES
        }
    )
    with pytest.raises(ValueError, match="input size"):
        prepare_image_bundle(source, load_protocol(), 0)


def test_basic_neural_methods_share_mlp_blueprint():
    protocol = load_protocol()
    for method in ("nnpu", "self_pu"):
        row = resolve_unit(protocol, "imdb", method)
        model = assemble_model(protocol, row, 384, seed=0, params={}, class_prior=0.3, device="cpu")
        network = model.model if method == "nnpu" else model.backbone
        validate_score_model(network, "mlp128")
        assert network[1].in_features == 384


def test_param_wrong_network_blueprint_rejected():
    with pytest.raises(ValueError, match="blueprint"):
        validate_score_model(nn.Linear(3, 1), "mlp128")
    with pytest.raises(ValueError, match="blueprint"):
        validate_score_model(nn.Linear(3, 1), "resnet18_linear")


def test_determ_oracle_reproducible_and_cloneable():
    from sklearn.base import clone

    rng = np.random.RandomState(3)
    X, y = rng.normal(size=(8, 3)), np.array([0, 1] * 4)
    first = PilotOracleMLP(max_epochs=2, batch_size=4, random_state=0).fit(X, y)
    second = clone(first).fit(X, y)
    np.testing.assert_array_equal(first.decision_function(X), second.decision_function(X))
    assert len(first.loss_history_) == 2
    validate_score_model(first.model_, "mlp128")


def test_edge_oracle_rejects_single_class_and_zero_budget():
    X = np.zeros((4, 3))
    with pytest.raises(ValueError, match="both real"):
        PilotOracleMLP(max_epochs=1).fit(X, np.ones(4))
    with pytest.raises(ValueError, match="positive"):
        PilotOracleMLP(max_epochs=0).fit(X, np.array([0, 1, 0, 1]))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_basic_real_resnet_adapter_gpu_cache_smoke(tmp_path):
    prepared, encoder, image = prepare_image_bundle(image_bundle(), load_protocol(), 0)
    before = _encoder_state_sha256(encoder)
    first, one = cached_adapter(
        prepared, encoder, image, cache_dir=tmp_path, device="cuda", batch_size=2
    )
    second, two = cached_adapter(
        prepared, encoder, image, cache_dir=tmp_path, device="cuda", batch_size=2
    )
    assert all(parameter.device.type == "cuda" for parameter in encoder.parameters())
    assert _encoder_state_sha256(encoder) == before
    assert one["device"] == "cuda" and one["encoder_mode"] == "eval_no_grad"
    assert not one["cache_hit"] and two["cache_hit"]
    assert one["feature_dimension"] == 512
    assert one["representation_sha256"] == two["representation_sha256"]
    for role in ROLES:
        features = getattr(first, role).X
        assert features.shape == (4, 512) and np.isfinite(features).all()
        np.testing.assert_array_equal(features, getattr(second, role).X)
        np.testing.assert_array_equal(getattr(first, role).labels, getattr(prepared, role).labels)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_determ_oracle_gpu_seed_training_and_prediction():
    from sklearn.base import clone

    rng = np.random.RandomState(3)
    X, y = rng.normal(size=(8, 3)), np.array([0, 1] * 4)
    first = PilotOracleMLP(max_epochs=2, batch_size=4, random_state=0, device="cuda").fit(X, y)
    second = clone(first).fit(X, y)
    assert all(parameter.device.type == "cuda" for parameter in first.model_.parameters())
    assert len(first.loss_history_) == 2 and np.isfinite(first.loss_history_).all()
    scores = first.decision_function(X)
    assert scores.shape == (8,) and np.isfinite(scores).all()
    np.testing.assert_array_equal(scores, second.decision_function(X))
    validate_score_model(first.model_, "mlp128")
