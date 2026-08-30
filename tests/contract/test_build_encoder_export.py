# ruff: noqa: N802, N803, N806
"""Public export contract for build_encoder (dual_architecture_plan.md §5 阶段 1)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import build_encoder  # noqa: E402
from pu_toolbox.estimators.deep import build_encoder as deep_export  # noqa: E402
from pu_toolbox.estimators.deep.vision import build_wconpu_backbone  # noqa: E402


@pytest.mark.contract
def test_exported_from_package_root():
    assert build_encoder is deep_export


@pytest.mark.contract
def test_mlp_returns_none():
    assert build_encoder("mlp", in_channels=3) is None


@pytest.mark.contract
def test_invalid_architecture_raises():
    with pytest.raises(ValueError, match="architecture"):
        build_encoder("lstm", in_channels=3)


@pytest.mark.contract
def test_cnn_matches_wconpu_backbone_structure():
    encoder = build_encoder("cnn", backbone="cnn13", in_channels=3)
    reference = build_wconpu_backbone("cnn13", in_channels=3)
    assert isinstance(encoder, torch.nn.Sequential)
    assert isinstance(reference, torch.nn.Sequential)
    x = torch.randn(2, 3, 8, 8)
    assert tuple(encoder(x).shape) == tuple(reference(x).shape)
