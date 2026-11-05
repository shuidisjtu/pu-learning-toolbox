# ruff: noqa: E402

"""Unit tests for the unified deep-encoder entry point ``build_encoder``."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.vision import build_encoder, build_wconpu_backbone


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
        # 同 seed 构建 → 相同权重 → 相同前向；随机输入避免 zero 输入的
        # 平台相关数值噪声（BN 训练模式对常数 batch 输出的 subnormal 差异）
        inputs = torch.randn(2, 3, 8, 8)
        torch.manual_seed(0)
        a = build_encoder("cnn", backbone="cnn13", in_channels=3).eval()
        torch.manual_seed(0)
        b = build_encoder("cnn", backbone="cnn13", in_channels=3).eval()
        with torch.no_grad():
            xa = a(inputs)
            xb = b(inputs)
        np.testing.assert_allclose(xa.numpy(), xb.numpy(), rtol=1e-6, atol=1e-6)

    def test_determ_relu_gain_preserves_seed_draws_and_exact_legacy_default(self):
        options = dict(
            in_channels=1, base_channels=2, normalization_mean=(0.0,), normalization_std=(1.0,)
        )
        torch.manual_seed(11)
        legacy = build_wconpu_backbone("cnn13_no_bn", **options)
        state_after_legacy = torch.random.get_rng_state()
        torch.manual_seed(11)
        candidate = build_wconpu_backbone("cnn13_no_bn", initialization="relu_gain", **options)
        assert torch.equal(state_after_legacy, torch.random.get_rng_state())
        for original, modified in zip(legacy.parameters(), candidate.parameters(), strict=True):
            torch.testing.assert_close(modified, original * (6**0.5), rtol=0, atol=0)
        torch.manual_seed(11)
        explicit = build_wconpu_backbone("cnn13_no_bn", initialization="pytorch_default", **options)
        for original, default in zip(legacy.parameters(), explicit.parameters(), strict=True):
            torch.testing.assert_close(default, original, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "name,init",
        [("cnn13", "relu_gain"), ("resnet18", "relu_gain"), ("cnn13_no_bn", "automatic")],
    )
    def test_param_initialization_rejects_unrelated_backbones(self, name, init):
        with pytest.raises(ValueError, match="initialization"):
            build_wconpu_backbone(name, initialization=init)
