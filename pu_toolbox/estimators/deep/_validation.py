"""Shared validation for encoder feature outputs (deep estimators)."""

from __future__ import annotations


def validate_encoder_features(features, *, encoder_param_name: str) -> int:
    """Validate an encoder's output and return its feature dimension.

    Encoder contract (dual_architecture_plan.md §4.1): output must be a
    2-D ``(batch, feature_dim)`` tensor with finite values.
    """
    import torch  # lazy: keep the torch-free import chain of deep estimators

    if not torch.is_tensor(features):
        raise TypeError(
            f"encoder {encoder_param_name!r} must return a torch.Tensor; "
            f"got {type(features).__name__}"
        )
    if features.ndim != 2:
        raise ValueError(
            f"encoder {encoder_param_name!r} must output a 2-D "
            f"(batch, feature_dim) tensor; got shape {tuple(features.shape)}"
        )
    if not torch.isfinite(features).all():
        raise ValueError(f"encoder {encoder_param_name!r} output contains NaN or Inf values")
    feature_dim = int(features.shape[1])
    if feature_dim < 1:
        raise ValueError(f"encoder {encoder_param_name!r} output has empty feature dimension")
    return feature_dim
