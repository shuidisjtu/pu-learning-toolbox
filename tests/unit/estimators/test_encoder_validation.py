# ruff: noqa: N802, N803, N806
"""Unit tests for validate_encoder_features."""

from __future__ import annotations

import pytest

from pu_toolbox.estimators.deep._validation import validate_encoder_features

torch = pytest.importorskip("torch", reason="PyTorch not installed")


@pytest.mark.unit
def test_valid_2d_output_returns_feature_dim():
    features = torch.zeros(4, 7)
    assert validate_encoder_features(features, encoder_param_name="encoder") == 7


@pytest.mark.unit
def test_ndim_not_2_raises():
    with pytest.raises(ValueError, match="2-D"):
        validate_encoder_features(torch.zeros(2, 3, 4), encoder_param_name="encoder")


@pytest.mark.unit
def test_non_finite_raises():
    features = torch.zeros(2, 3)
    features[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_encoder_features(features, encoder_param_name="encoder")


@pytest.mark.unit
def test_non_tensor_raises():
    with pytest.raises(TypeError, match="torch.Tensor"):
        validate_encoder_features([1.0, 2.0], encoder_param_name="encoder")


@pytest.mark.unit
def test_zero_feature_dim_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_encoder_features(torch.zeros(2, 0), encoder_param_name="encoder")
