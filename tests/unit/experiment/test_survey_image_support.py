"""The clean-support part travels the same image path as the four dataset roles.

``prepare_image_bundle`` rewrites ``train``/``pu_val``/``clean_val``/``test`` and
stops there, while the support set is loaded from its own artifact.  Left alone it
reaches a native CNN as uint8 pixels in ``[0, 255]`` and a feature-adapter head as
4-D images where the training rows are 2-D, so these tests pin the conversion
rather than the two symptoms: the support rows must land exactly where the same
pixels land as a role, and nothing about their membership may move.
"""

import numpy as np
import pytest
import torch
from torch import nn

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart
from pu_toolbox.experiment.feature_adapter import (
    adapt_image_bundle_to_features,
    encoder_state_sha256,
)
from pu_toolbox.experiment.survey_execution import (
    prepare_image_bundle,
    prepare_image_support,
)
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATHS, ROLES, load_protocol

pytestmark = pytest.mark.unit

#: The v3 matrix is the one the support units run under; ``load_protocol()`` still
#: defaults to v2, whose image spec has no per-dataset block and would fit every
#: dataset at CIFAR-10's 32x32x3.
PROTOCOL = load_protocol(PROTOCOL_PATHS["survey-v3.0"])

#: ``(dataset, in_channels, size)``: the two image datasets this task adds.
IMAGES = (("cifar10", 3, 32), ("mnist", 1, 28))

SUPPORT_LABELS = np.array([1, 0, 1, 0, 1, 0])
SUPPORT_INDICES = np.arange(100, 106)


def support_part(channels=3, size=32, seed=5):
    rng = np.random.RandomState(seed)
    return DatasetPart(
        X=rng.randint(0, 256, size=(6, channels, size, size), dtype=np.uint8),
        labels=SUPPORT_LABELS.copy(),
        view="clean",
        indices=SUPPORT_INDICES.copy(),
        for_selection=False,
    )


def image_bundle(channels=3, size=32):
    rng = np.random.RandomState(9)
    parts = {}
    for index, role in enumerate(ROLES):
        parts[role] = DatasetPart(
            X=rng.randint(0, 256, size=(4, channels, size, size), dtype=np.uint8),
            labels=np.array([1, 1, 0, 0]),
            view="clean",
            indices=np.arange(index * 4, index * 4 + 4),
            for_selection=role != "test",
        )
    return DatasetBundle(**parts)


def tiny_encoder(channels=3):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        return nn.Sequential(nn.Conv2d(channels, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())


def prepared(dataset, channels, size, *, support_as_test=False):
    """The four-role bundle, with the support pixels optionally standing in as ``test``.

    Using the support rows as a role is what makes "the same pixels land in the
    same place" checkable: the role goes through the production preprocessing and
    the support part has to arrive at the same array.
    """
    source = image_bundle(channels, size)
    if support_as_test:
        support = support_part(channels, size)
        source = DatasetBundle(
            train=source.train,
            pu_val=source.pu_val,
            clean_val=source.clean_val,
            test=DatasetPart(
                X=support.X,
                labels=support.labels,
                view="clean",
                indices=support.indices,
                for_selection=False,
            ),
        )
    return prepare_image_bundle(source, PROTOCOL, 0, dataset=dataset, with_preprocessing=True)


def test_basic_native_cnn_support_lands_where_the_roles_do():
    bundle, _, _, fitted = prepared("cifar10", 3, 32, support_as_test=True)
    support = support_part()
    out = prepare_image_support(support, training_path="native_cnn", preprocessing=fitted)
    assert out.X.dtype == np.float32
    assert out.X.shape == support.X.shape
    np.testing.assert_array_equal(out.X, bundle.test.X)
    assert out.X.min() >= 0.0 and out.X.max() <= 1.0


def test_param_adapter_support_lands_in_the_frozen_feature_space():
    for dataset, channels, size in IMAGES:
        bundle, _, metadata, fitted = prepared(dataset, channels, size, support_as_test=True)
        encoder = tiny_encoder(channels)
        adapted, _ = adapt_image_bundle_to_features(
            bundle, encoder, feature_version="test-v1", backbone_manifest=metadata
        )
        out = prepare_image_support(
            support_part(channels, size),
            training_path="cnn_feature_adapter",
            preprocessing=fitted,
            encoder=encoder,
        )
        assert out.X.ndim == 2 and out.X.dtype == np.float32
        np.testing.assert_array_equal(out.X, adapted.test.X)
        np.testing.assert_array_equal(out.labels, SUPPORT_LABELS)
        np.testing.assert_array_equal(out.indices, SUPPORT_INDICES)


def test_param_membership_survives_every_image_training_path():
    for dataset, channels, size in IMAGES:
        _, encoder, _, fitted = prepared(dataset, channels, size)
        for training_path in ("native_cnn", "cnn_feature_adapter"):
            out = prepare_image_support(
                support_part(channels, size),
                training_path=training_path,
                preprocessing=fitted,
                encoder=encoder,
            )
            assert out.view == "clean"
            assert out.for_selection is False
            np.testing.assert_array_equal(out.labels, SUPPORT_LABELS)
            np.testing.assert_array_equal(out.indices, SUPPORT_INDICES)


def test_param_refuses_a_training_path_that_has_no_image_support_assembly():
    _, _, _, fitted = prepared("cifar10", 3, 32)
    with pytest.raises(ValueError, match="native_cnn and cnn_feature_adapter"):
        prepare_image_support(support_part(), training_path="native_2d", preprocessing=fitted)


def test_edge_refuses_inputs_that_are_not_images():
    _, _, _, fitted = prepared("cifar10", 3, 32)
    flat = DatasetPart(
        np.zeros((6, 57), dtype=np.float32), SUPPORT_LABELS, "clean", SUPPORT_INDICES
    )
    with pytest.raises(ValueError, match="non-empty NCHW images"):
        prepare_image_support(flat, training_path="native_cnn", preprocessing=fitted)
    empty = DatasetPart(
        np.zeros((0, 3, 32, 32), dtype=np.uint8),
        np.zeros(0, dtype=int),
        "clean",
        np.zeros(0, dtype=int),
    )
    with pytest.raises(ValueError, match="non-empty NCHW images"):
        prepare_image_support(empty, training_path="native_cnn", preprocessing=fitted)


def test_edge_adapter_path_refuses_to_run_without_the_frozen_encoder():
    _, _, _, fitted = prepared("cifar10", 3, 32)
    with pytest.raises(ValueError, match="requires the frozen encoder"):
        prepare_image_support(
            support_part(), training_path="cnn_feature_adapter", preprocessing=fitted
        )


def test_determ_support_conversion_is_repeatable_and_leaves_the_encoder_alone():
    _, _, _, fitted = prepared("cifar10", 3, 32)
    encoder = tiny_encoder()
    before = encoder_state_sha256(encoder)
    first = prepare_image_support(
        support_part(),
        training_path="cnn_feature_adapter",
        preprocessing=fitted,
        encoder=encoder,
    )
    second = prepare_image_support(
        support_part(),
        training_path="cnn_feature_adapter",
        preprocessing=fitted,
        encoder=encoder,
    )
    np.testing.assert_array_equal(first.X, second.X)
    assert encoder_state_sha256(encoder) == before
