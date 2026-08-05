# ruff: noqa: E402

"""Unit tests for the unified deep-encoder entry point ``build_encoder``."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.vision import build_encoder


@pytest.mark.unit
class TestBuildEncoderBasic:
    def test_basic_mlp_returns_none(self):
        assert build_encoder("mlp", in_channels=3) is None

    def test_basic_cnn13_forward_produces_features(self):
        encoder = build_encoder("cnn", backbone="cnn13", in_channels=3)
        out = encoder(torch.zeros(2, 3, 8, 8))
        assert out.ndim == 2 and out.shape[0] == 2

    def test_basic_default_normalization_matches_channels(self):
        encoder = build_encoder("cnn", backbone="cnn13", in_channels=1)
        out = encoder(torch.zeros(2, 1, 8, 8))
        assert out.ndim == 2 and out.shape[0] == 2


@pytest.mark.unit
class TestBuildEncoderParams:
    def test_param_invalid_architecture_raises(self):
        with pytest.raises(ValueError, match="architecture"):
            build_encoder("lstm", in_channels=3)

    def test_param_invalid_backbone_raises(self):
        with pytest.raises(ValueError, match="cnn13"):
            build_encoder("cnn", backbone="vgg16", in_channels=3)


@pytest.mark.unit
class TestBuildEncoderEdge:
    def test_edge_custom_normalization_stats(self):
        encoder = build_encoder(
            "cnn",
            backbone="cnn13",
            in_channels=3,
            normalization_mean=(0.1, 0.2, 0.3),
            normalization_std=(0.4, 0.5, 0.6),
        )
        # 前向验证 Normalize 模块已内嵌（模块名包含 Normalize）
        assert any("Normalize" in type(module).__name__ for module in encoder)


@pytest.mark.unit
class TestBuildEncoderDeterministic:
    def test_determ_same_arguments_same_forward(self):
        a = build_encoder("cnn", backbone="cnn13", in_channels=3)
        b = build_encoder("cnn", backbone="cnn13", in_channels=3)
        torch.manual_seed(0)
        xa = a(torch.zeros(2, 3, 8, 8))
        torch.manual_seed(0)
        xb = b(torch.zeros(2, 3, 8, 8))
        np.testing.assert_allclose(xa.detach().numpy(), xb.detach().numpy())
