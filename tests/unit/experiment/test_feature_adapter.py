# ruff: noqa: E402, N803, N806

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pu_toolbox.experiment import (
    DatasetBundle,
    DatasetPart,
    LeaderboardRunSpec,
    adapt_image_bundle_to_features,
    partition_fair_leaderboard_runs,
)

pytestmark = pytest.mark.unit


def _bundle():
    rng = np.random.default_rng(4)
    parts = {}
    offset = 0
    for role, size in (("train", 8), ("pu_val", 4), ("clean_val", 4), ("test", 4)):
        parts[role] = DatasetPart(
            X=rng.normal(size=(size, 1, 4, 4)).astype(np.float32),
            labels=np.arange(size) % 2,
            view="clean",
            indices=np.arange(offset, offset + size),
            for_selection=role != "test",
        )
        offset += size
    return DatasetBundle(**parts)


def _encoder():
    return nn.Sequential(nn.Flatten(), nn.Linear(16, 3, bias=False))


def _run(method, path="native_cnn", **overrides):
    values = {
        "method": method,
        "dataset": "mnist",
        "training_path": path,
        "adaptation_level": (
            "benchmark-adapted" if path == "cnn_feature_adapter" else "source-faithful"
        ),
        "split_sha256": "a" * 64,
        "representation_sha256": ("b" if path == "native_cnn" else "c") * 64,
        "max_epochs": 100,
        "batch_size_candidates": (64, 128),
        "tuning_candidate_count": 5,
        "seeds": (0, 1, 2, 3, 4),
    }
    values.update(overrides)
    return LeaderboardRunSpec(**values)


def test_basic_adapter_preserves_roles_and_returns_2d_features():
    source = _bundle()
    adapted, manifest = adapt_image_bundle_to_features(
        source,
        _encoder(),
        feature_version="mnist-resnet18-fold-v1",
        backbone_manifest={"name": "tiny-test-encoder", "weights": None},
        batch_size=3,
    )

    assert adapted.train.X.shape == (8, 3)
    assert adapted.test.X.shape == (4, 3)
    np.testing.assert_array_equal(adapted.train.indices, source.train.indices)
    assert adapted.test.for_selection is False
    assert manifest["training_path"] == "cnn_feature_adapter"
    assert manifest["adaptation_level"] == "benchmark-adapted"
    assert len(manifest["encoder_state_sha256"]) == 64
    assert len(manifest["feature_sha256"]) == 4


def test_determ_same_encoder_and_bundle_produce_same_feature_hashes():
    torch.manual_seed(8)
    encoder_a = _encoder()
    torch.manual_seed(8)
    encoder_b = _encoder()
    kwargs = {
        "feature_version": "v1",
        "backbone_manifest": {"name": "tiny", "revision": "abc"},
    }

    features_a, manifest_a = adapt_image_bundle_to_features(_bundle(), encoder_a, **kwargs)
    features_b, manifest_b = adapt_image_bundle_to_features(_bundle(), encoder_b, **kwargs)

    np.testing.assert_array_equal(features_a.test.X, features_b.test.X)
    assert manifest_a == manifest_b


def test_param_train_fitted_encoder_requires_exact_train_indices():
    bundle = _bundle()
    _, manifest = adapt_image_bundle_to_features(
        bundle,
        _encoder(),
        feature_version="fold-v1",
        backbone_manifest={"name": "tiny"},
        encoder_fit_scope="train_partition_only",
        encoder_fit_indices=bundle.train.indices[::-1],
    )
    assert manifest["encoder_fit_scope"] == "train_partition_only"
    assert len(manifest["encoder_fit_indices_sha256"]) == 64

    with pytest.raises(ValueError, match="exactly match"):
        adapt_image_bundle_to_features(
            bundle,
            _encoder(),
            feature_version="fold-v1",
            backbone_manifest={"name": "tiny"},
            encoder_fit_scope="train_partition_only",
            encoder_fit_indices=np.r_[bundle.train.indices, bundle.test.indices[0]],
        )


def test_edge_adapter_rejects_non_2d_encoder_outputs():
    bundle = _bundle()
    with pytest.raises(ValueError, match="2-D tensor"):
        adapt_image_bundle_to_features(
            bundle,
            nn.Conv2d(1, 2, 1),
            feature_version="bad-v1",
            backbone_manifest={"name": "conv-only"},
        )


def test_basic_fairness_gate_partitions_native_and_adapter_rankings():
    runs = [
        _run("nnpu"),
        _run("wconpu"),
        _run("upu", "cnn_feature_adapter"),
        _run("kldce", "cnn_feature_adapter"),
    ]
    groups = partition_fair_leaderboard_runs(runs)

    assert set(groups) == {"mnist/native_cnn", "mnist/cnn_feature_adapter"}
    assert groups["mnist/native_cnn"]["methods"] == ["nnpu", "wconpu"]
    assert groups["mnist/cnn_feature_adapter"]["methods"] == ["kldce", "upu"]
    assert all(len(group["fairness_sha256"]) == 64 for group in groups.values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split_sha256", "d" * 64),
        ("max_epochs", 50),
        ("batch_size_candidates", (32,)),
        ("tuning_candidate_count", 7),
        ("seeds", (0, 1)),
    ],
)
def test_param_fairness_gate_rejects_shared_protocol_mismatch(field, value):
    with pytest.raises(ValueError, match=field):
        partition_fair_leaderboard_runs([_run("a"), _run("b", **{field: value})])


def test_edge_fairness_gate_rejects_path_mislabel_and_representation_drift():
    with pytest.raises(ValueError, match="benchmark-adapted"):
        partition_fair_leaderboard_runs(
            [
                _run(
                    "upu",
                    "cnn_feature_adapter",
                    adaptation_level="source-faithful",
                )
            ]
        )
    with pytest.raises(ValueError, match="representation_sha256"):
        partition_fair_leaderboard_runs([_run("a"), _run("b", representation_sha256="d" * 64)])
