# ruff: noqa: N803, S101

"""Vision module pickling (E2/E3 regression: local classes cannot be pickled)."""

import pickle

import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.vision import (  # noqa: E402
    build_wconpu_augmentation,
    build_wconpu_backbone,
)

pytestmark = pytest.mark.unit


def test_channel_normalize_pickle_roundtrip():
    model = build_wconpu_backbone("cnn13", in_channels=3)
    restored = pickle.loads(pickle.dumps(model))
    model.eval()
    restored.eval()
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(model(x), restored(x))


def test_augmentation_pickle_roundtrip():
    transform = build_wconpu_augmentation("simaugment", image_size=32)
    restored = pickle.loads(pickle.dumps(transform))
    x = torch.randn(2, 3, 32, 32)
    assert restored(x).shape == x.shape
