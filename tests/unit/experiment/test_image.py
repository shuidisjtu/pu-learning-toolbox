import numpy as np
import pytest

from pu_toolbox.experiment import (
    build_survey_image_augmentation,
    build_survey_image_encoder,
    fit_survey_image_preprocessing,
    transform_survey_images,
)

pytestmark = pytest.mark.unit


def _uint8_images(samples=6, channels=1, size=8):
    values = np.arange(samples * channels * size * size, dtype=np.uint16) % 256
    return values.astype(np.uint8).reshape(samples, channels, size, size)


def test_basic_train_fit_records_complete_resnet18_provenance():
    prepared, spec = fit_survey_image_preprocessing(
        _uint8_images(), dataset="mnist", train_augmentation="simaugment"
    )
    manifest = spec.to_manifest()

    assert prepared.dtype == np.float32
    assert 0 <= prepared.min() <= prepared.max() <= 1
    assert manifest["input_layout"] == "NCHW"
    assert manifest["backbone"]["name"] == "resnet18"
    assert manifest["backbone"]["initialization"] == "random"
    assert manifest["backbone"]["first_layer"]["in_channels"] == 1
    assert manifest["augmentation"]["train"]["name"] == "simaugment"
    assert manifest["augmentation"]["test"] == {"name": "none"}
    assert len(manifest["configuration_sha256"]) == 64


@pytest.mark.parametrize(
    ("dataset", "channels"),
    [("f-mnist", 1), ("cifar10", 3), ("adni", 1)],
)
def test_param_image_catalog_locks_expected_channels(dataset, channels):
    images = np.linspace(0, 1, 4 * channels * 8 * 8, dtype=np.float32).reshape(4, channels, 8, 8)
    _, spec = fit_survey_image_preprocessing(images, dataset=dataset, train_augmentation="none")
    assert spec.in_channels == channels
    assert spec.scaling_policy == "identity_unit_interval"


def test_determ_same_train_data_produces_same_stats_and_hashes():
    images = _uint8_images()
    first_array, first = fit_survey_image_preprocessing(images, dataset="mnist")
    second_array, second = fit_survey_image_preprocessing(images.copy(), dataset="mnist")

    np.testing.assert_array_equal(first_array, second_array)
    assert first == second
    assert first.train_data_sha256 == second.train_data_sha256
    assert first.configuration_sha256 == second.configuration_sha256


def test_basic_transform_reuses_scaling_and_blocks_eval_augmentation():
    train = _uint8_images()
    _, spec = fit_survey_image_preprocessing(train, dataset="mnist")
    transformed = transform_survey_images(train[:2], spec, role="test")

    np.testing.assert_allclose(transformed, train[:2].astype(np.float32) / 255)
    assert build_survey_image_augmentation(spec, role="pu_val") is None
    assert build_survey_image_augmentation(spec, role="clean_val") is None
    assert build_survey_image_augmentation(spec, role="test") is None


def test_basic_builders_use_gray_resnet_and_train_only_augmentation():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    _, spec = fit_survey_image_preprocessing(_uint8_images(), dataset="mnist")

    encoder = build_survey_image_encoder(spec).eval()
    augmentation = build_survey_image_augmentation(spec, role="train")
    with torch.no_grad():
        features = encoder(torch.from_numpy(_uint8_images(2).astype(np.float32) / 255))
    assert features.shape == (2, 512)
    assert augmentation is not None


def test_edge_rejects_shape_range_scaling_and_degenerate_train_stats():
    with pytest.raises(ValueError, match="NCHW"):
        fit_survey_image_preprocessing(np.zeros((2, 8, 8)), dataset="mnist")
    with pytest.raises(ValueError, match="3 channel"):
        fit_survey_image_preprocessing(_uint8_images(), dataset="cifar10")
    with pytest.raises(ValueError, match="non-zero"):
        fit_survey_image_preprocessing(np.zeros((3, 1, 8, 8), dtype=np.uint8), dataset="mnist")

    _, spec = fit_survey_image_preprocessing(_uint8_images(), dataset="mnist")
    with pytest.raises(ValueError, match="scaling differs"):
        transform_survey_images(
            _uint8_images(2).astype(np.float32) / 255,
            spec,
            role="test",
        )
    with pytest.raises(ValueError, match="role"):
        transform_survey_images(_uint8_images(2), spec, role="validation")
    with pytest.raises(ValueError, match="positive integer"):
        fit_survey_image_preprocessing(_uint8_images(), dataset="mnist", randaugment_num_ops=1.5)
