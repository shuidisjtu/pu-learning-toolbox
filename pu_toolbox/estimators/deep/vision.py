# ruff: noqa: N812

"""Lazy PyTorch vision adapters for deep PU estimators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

try:  # keep this module importable in the base wheel without torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in torch-free installs
    nn = None  # type: ignore[assignment]


if nn is not None:

    class ChannelNormalize(nn.Module):
        def __init__(self, mean, std):
            import torch

            super().__init__()
            self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

        def forward(self, inputs):
            if inputs.ndim != 4 or inputs.shape[1] != self.mean.shape[1]:
                raise ValueError(
                    f"vision backbone expects NCHW inputs with {self.mean.shape[1]} channels"
                )
            return (inputs - self.mean) / self.std

else:  # pragma: no cover - placeholder for torch-free imports

    class ChannelNormalize:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required to build vision modules")


def _normalization_module(mean: Sequence[float], std: Sequence[float]):
    if len(mean) != len(std) or not mean or any(value <= 0 for value in std):
        raise ValueError("normalization mean/std must have equal non-zero length and positive std")
    return ChannelNormalize(mean, std)


if nn is not None:

    class IndependentBatchAugmentation(nn.Module):  # type: ignore[no-redef]
        def __init__(self, transform):
            super().__init__()
            self.transform = transform

        def forward(self, inputs):
            import torch  # torch is not imported at module top (lazy design)

            if inputs.ndim != 4:
                raise ValueError("vision augmentation expects NCHW inputs")
            return torch.stack([self.transform(image) for image in inputs])

else:  # pragma: no cover - placeholder for torch-free imports

    class IndependentBatchAugmentation:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required to build vision modules")


def _conv_block(in_channels: int, out_channels: int):
    from torch import nn

    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


def build_wconpu_backbone(
    name: Literal["cnn13", "resnet18", "resnet50"],
    *,
    in_channels: int = 3,
    base_channels: int = 64,
    small_input_stem: bool = False,
    normalization_mean: Sequence[float] = (0.5, 0.5, 0.5),
    normalization_std: Sequence[float] = (0.5, 0.5, 0.5),
):
    """Build an image feature encoder for the WConPU paper protocol.

    ``cnn13`` is a documented clean-room 13-convolution adapter because the
    paper specifies only the depth. ResNet adapters use torchvision topology
    with random initialization and return flattened feature vectors.
    """
    if name not in {"cnn13", "resnet18", "resnet50"}:
        raise ValueError("name must be 'cnn13', 'resnet18', or 'resnet50'")
    if in_channels < 1 or base_channels < 1:
        raise ValueError("in_channels and base_channels must be positive")
    if len(normalization_mean) != in_channels:
        raise ValueError("normalization statistics must match in_channels")

    from torch import nn

    normalize = _normalization_module(normalization_mean, normalization_std)
    if name == "cnn13":
        channels = [
            base_channels,
            base_channels,
            base_channels,
            base_channels * 2,
            base_channels * 2,
            base_channels * 2,
            base_channels * 4,
            base_channels * 4,
            base_channels * 4,
            base_channels * 8,
            base_channels * 8,
            base_channels * 8,
            base_channels * 8,
        ]
        layers: list[nn.Module] = [normalize]
        previous = in_channels
        for index, width in enumerate(channels):
            layers.extend(_conv_block(previous, width))
            previous = width
            if index in {2, 5, 8}:
                layers.append(nn.MaxPool2d(2))
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(1)])
        return nn.Sequential(*layers)

    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError("ResNet vision adapters require torchvision") from exc
    constructor = models.resnet18 if name == "resnet18" else models.resnet50
    model = constructor(weights=None)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels,
            model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False,
        )
    if small_input_stem:
        model.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
    return nn.Sequential(normalize, *list(model.children())[:-1], nn.Flatten(1))


def build_wconpu_augmentation(
    name: Literal["simaugment", "randaugment"],
    *,
    image_size: int,
    crop_scale: tuple[float, float] = (0.2, 1.0),
    horizontal_flip_probability: float = 0.5,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 10,
):
    """Build an independently sampled per-image tensor augmentation."""
    if name not in {"simaugment", "randaugment"}:
        raise ValueError("name must be 'simaugment' or 'randaugment'")
    if image_size < 1:
        raise ValueError("image_size must be positive")
    if not 0 < crop_scale[0] <= crop_scale[1] <= 1:
        raise ValueError("crop_scale must satisfy 0 < low <= high <= 1")
    if not 0 <= horizontal_flip_probability <= 1:
        raise ValueError("horizontal_flip_probability must be in [0, 1]")
    if randaugment_num_ops < 1 or not 0 <= randaugment_magnitude <= 30:
        raise ValueError("RandAugment num_ops must be positive and magnitude in [0, 30]")

    try:
        from torchvision.transforms import v2
    except ImportError as exc:
        raise ImportError("vision augmentation requires torch and torchvision") from exc

    operations: list[nn.Module] = [
        v2.RandomResizedCrop((image_size, image_size), scale=crop_scale),
        v2.RandomHorizontalFlip(horizontal_flip_probability),
    ]
    if name == "simaugment":
        operations.extend(
            [
                v2.RandomApply(
                    [v2.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )
    else:
        operations.append(
            v2.RandAugment(
                num_ops=randaugment_num_ops,
                magnitude=randaugment_magnitude,
            )
        )
    transform = v2.Compose(operations)
    return IndependentBatchAugmentation(transform)


def build_encoder(
    architecture: Literal["mlp", "cnn"],
    *,
    backbone: str = "cnn13",
    in_channels: int,
    normalization_mean: Sequence[float] | None = None,
    normalization_std: Sequence[float] | None = None,
):
    """Build the encoder for a deep PU classifier.

    ``"mlp"`` returns ``None`` -- the classifier's built-in MLP path is
    used (table data).  ``"cnn"`` returns an image backbone from
    :func:`build_wconpu_backbone` (4-D NCHW inputs); normalization
    statistics default to ``0.5`` per channel when not supplied.
    """
    if architecture not in {"mlp", "cnn"}:
        raise ValueError("architecture must be 'mlp' or 'cnn'")
    if architecture == "mlp":
        return None
    if normalization_mean is None:
        normalization_mean = (0.5,) * in_channels
    if normalization_std is None:
        normalization_std = (0.5,) * in_channels
    return build_wconpu_backbone(
        backbone,
        in_channels=in_channels,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )
