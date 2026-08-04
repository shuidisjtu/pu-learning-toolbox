# ruff: noqa: E402, N803, N806

"""Tests for WConPU visual backbones and tensor augmentations."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from benchmarks.deep_pu.runner import _build_estimator
from pu_toolbox.estimators.deep import (
    WeightedContrastivePUClassifier,
    build_wconpu_augmentation,
    build_wconpu_backbone,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "expected_features"),
    [("cnn13", 16), ("resnet18", 512), ("resnet50", 2048)],
)
def test_param_backbones_return_flat_features(name, expected_features):
    kwargs = {"base_channels": 2} if name == "cnn13" else {"small_input_stem": True}
    model = build_wconpu_backbone(name, **kwargs).eval()
    with torch.no_grad():
        result = model(torch.rand(2, 3, 32, 32))
    assert result.shape == (2, expected_features)


@pytest.mark.unit
def test_determ_augmentations_preserve_batch_shape_and_seed():
    images = torch.linspace(0, 1, 4 * 3 * 16 * 16).reshape(4, 3, 16, 16)
    for name in ("simaugment", "randaugment"):
        augmentation = build_wconpu_augmentation(name, image_size=16)
        torch.manual_seed(9)
        first = augmentation(images)
        torch.manual_seed(9)
        second = augmentation(images)
        assert first.shape == images.shape
        assert torch.isfinite(first).all()
        torch.testing.assert_close(first, second)


@pytest.mark.unit
def test_basic_wconpu_trains_on_nchw_images_with_cosine_scheduler():
    rng = np.random.RandomState(4)
    X = rng.rand(8, 3, 16, 16).astype(np.float32)
    y_pu = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    estimator = WeightedContrastivePUClassifier(
        0.25,
        encoder=build_wconpu_backbone("cnn13", base_channels=2),
        weak_augmentation=build_wconpu_augmentation("simaugment", image_size=16),
        strong_augmentation=build_wconpu_augmentation("randaugment", image_size=16),
        hidden_dim=8,
        embedding_dim=4,
        queue_size=8,
        batch_size=4,
        max_epochs=1,
        scheduler="cosine_annealing",
        random_state=3,
    ).fit(X, y_pu)
    assert estimator.predict_proba(X).shape == (8, 2)
    assert estimator.final_learning_rate_ == pytest.approx(0.0)


@pytest.mark.unit
def test_edge_visual_builders_and_runner_reject_bad_config():
    with pytest.raises(ValueError, match="name"):
        build_wconpu_backbone("unknown")
    with pytest.raises(ValueError, match="crop_scale"):
        build_wconpu_augmentation("simaugment", image_size=16, crop_scale=(0.8, 0.2))
    with pytest.raises(ValueError, match="backbone.name"):
        _build_estimator(
            "weighted_contrastive_pu",
            {"parameters": {"vision": {"backbone": {}}}},
            class_prior=0.4,
            seed=0,
        )


@pytest.mark.unit
def test_basic_runner_builds_complete_visual_pipeline():
    estimator = _build_estimator(
        "weighted_contrastive_pu",
        {
            "parameters": {
                "vision": {
                    "backbone": {"name": "cnn13", "base_channels": 2},
                    "weak_augmentation": {"name": "simaugment", "image_size": 16},
                    "strong_augmentation": {"name": "randaugment", "image_size": 16},
                },
                "max_epochs": 1,
            }
        },
        class_prior=0.4,
        seed=5,
    )
    assert estimator.encoder is not None
    assert estimator.weak_augmentation is not None
    assert estimator.strong_augmentation is not None
