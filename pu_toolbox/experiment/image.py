# ruff: noqa: N803

"""Survey image preprocessing fitted on train data with auditable provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ImageRole = Literal["train", "pu_val", "clean_val", "test"]
TrainAugmentation = Literal["none", "simaugment", "randaugment"]

_IMAGE_CHANNELS = {
    "mnist": 1,
    "fashion_mnist": 1,
    "cifar10": 3,
    "adni": 1,
}
_ALIASES = {"f-mnist": "fashion_mnist", "f_mnist": "fashion_mnist"}


@dataclass(frozen=True)
class SurveyImagePreprocessing:
    """Frozen train-fitted preprocessing settings for one survey image dataset."""

    dataset: str
    in_channels: int
    input_size: tuple[int, int]
    scaling_policy: str
    source_dtype: str
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]
    train_augmentation: TrainAugmentation
    crop_scale: tuple[float, float]
    horizontal_flip_probability: float
    randaugment_num_ops: int
    randaugment_magnitude: int
    small_input_stem: bool
    train_data_sha256: str
    configuration_sha256: str

    def to_manifest(self) -> dict[str, Any]:
        """Return JSON-compatible image preprocessing provenance."""
        if self.small_input_stem:
            first_layer = {
                "in_channels": self.in_channels,
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": [1, 1],
                "maxpool": False,
            }
        else:
            first_layer = {
                "in_channels": self.in_channels,
                "kernel_size": [7, 7],
                "stride": [2, 2],
                "padding": [3, 3],
                "maxpool": True,
            }
        train_augmentation: dict[str, Any] = {"name": self.train_augmentation}
        if self.train_augmentation != "none":
            train_augmentation.update(
                {
                    "image_size": self.input_size[0],
                    "crop_scale": list(self.crop_scale),
                    "horizontal_flip_probability": self.horizontal_flip_probability,
                }
            )
        if self.train_augmentation == "randaugment":
            train_augmentation.update(
                {
                    "num_ops": self.randaugment_num_ops,
                    "magnitude": self.randaugment_magnitude,
                }
            )
        return {
            "schema_version": "1.0",
            "dataset": self.dataset,
            "input_layout": "NCHW",
            "input_size": list(self.input_size),
            "in_channels": self.in_channels,
            "source_dtype": self.source_dtype,
            "scaling_policy": self.scaling_policy,
            "normalization": {
                "source": "train_only_channel_statistics",
                "mean": list(self.normalization_mean),
                "std": list(self.normalization_std),
            },
            "backbone": {
                "name": "resnet18",
                "weights": None,
                "initialization": "random",
                "small_input_stem": self.small_input_stem,
                "first_layer": first_layer,
            },
            "augmentation": {
                "train": train_augmentation,
                "pu_val": {"name": "none"},
                "clean_val": {"name": "none"},
                "test": {"name": "none"},
            },
            "train_data_sha256": self.train_data_sha256,
            "configuration_sha256": self.configuration_sha256,
        }


def fit_survey_image_preprocessing(
    X_train: np.ndarray,
    *,
    dataset: str,
    train_augmentation: TrainAugmentation = "simaugment",
    crop_scale: tuple[float, float] = (0.8, 1.0),
    horizontal_flip_probability: float = 0.5,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 10,
) -> tuple[np.ndarray, SurveyImagePreprocessing]:
    """Fit scaling and channel normalization using only the train partition.

    The returned array is float32 in ``[0, 1]`` but is not normalized;
    normalization lives inside the encoder built by
    :func:`build_survey_image_encoder`. This prevents different callers from
    applying the learned statistics twice.
    """
    canonical_dataset = _resolve_dataset(dataset)
    expected_channels = _IMAGE_CHANNELS[canonical_dataset]
    prepared, scaling_policy, source_dtype = _prepare_array(
        X_train,
        expected_shape=None,
        expected_channels=expected_channels,
    )
    if prepared.shape[2] != prepared.shape[3]:
        raise ValueError("survey image inputs must be square for the locked augmentation path.")
    _validate_augmentation(
        train_augmentation,
        crop_scale,
        horizontal_flip_probability,
        randaugment_num_ops,
        randaugment_magnitude,
    )

    means = tuple(float(value) for value in np.mean(prepared, axis=(0, 2, 3)))
    stds = tuple(float(value) for value in np.std(prepared, axis=(0, 2, 3)))
    if any(value <= 0 for value in stds):
        raise ValueError("every train image channel must have non-zero standard deviation.")
    input_size = (int(prepared.shape[2]), int(prepared.shape[3]))
    train_data_sha256 = _array_sha256(prepared)
    config = {
        "schema_version": "1.0",
        "dataset": canonical_dataset,
        "in_channels": expected_channels,
        "input_size": list(input_size),
        "scaling_policy": scaling_policy,
        "source_dtype": source_dtype,
        "normalization_mean": list(means),
        "normalization_std": list(stds),
        "normalization_source": "train_only_channel_statistics",
        "backbone": "resnet18",
        "backbone_weights": None,
        "train_augmentation": train_augmentation,
        "crop_scale": list(crop_scale),
        "horizontal_flip_probability": horizontal_flip_probability,
        "randaugment_num_ops": randaugment_num_ops,
        "randaugment_magnitude": randaugment_magnitude,
        "small_input_stem": input_size[0] <= 64,
        "train_data_sha256": train_data_sha256,
    }
    spec = SurveyImagePreprocessing(
        dataset=canonical_dataset,
        in_channels=expected_channels,
        input_size=input_size,
        scaling_policy=scaling_policy,
        source_dtype=source_dtype,
        normalization_mean=means,
        normalization_std=stds,
        train_augmentation=train_augmentation,
        crop_scale=crop_scale,
        horizontal_flip_probability=horizontal_flip_probability,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
        small_input_stem=input_size[0] <= 64,
        train_data_sha256=train_data_sha256,
        configuration_sha256=_json_sha256(config),
    )
    return prepared, spec


def transform_survey_images(
    X: np.ndarray,
    preprocessing: SurveyImagePreprocessing,
    *,
    role: ImageRole,
) -> np.ndarray:
    """Apply the frozen input scaling and shape contract to another partition."""
    _validate_role(role)
    prepared, scaling_policy, _ = _prepare_array(
        X,
        expected_shape=(preprocessing.in_channels, *preprocessing.input_size),
        expected_channels=preprocessing.in_channels,
    )
    if scaling_policy != preprocessing.scaling_policy:
        raise ValueError(
            "image partition scaling differs from the train-fitted scaling policy: "
            f"{scaling_policy!r} != {preprocessing.scaling_policy!r}."
        )
    return prepared


def build_survey_image_encoder(preprocessing: SurveyImagePreprocessing):
    """Build the protocol-locked, randomly initialized ResNet-18 encoder."""
    from pu_toolbox.estimators.deep.vision import build_wconpu_backbone

    return build_wconpu_backbone(
        "resnet18",
        in_channels=preprocessing.in_channels,
        small_input_stem=preprocessing.small_input_stem,
        normalization_mean=preprocessing.normalization_mean,
        normalization_std=preprocessing.normalization_std,
    )


def build_survey_image_augmentation(
    preprocessing: SurveyImagePreprocessing,
    *,
    role: ImageRole,
):
    """Build augmentation for train only; all validation/test roles return ``None``."""
    _validate_role(role)
    if role != "train" or preprocessing.train_augmentation == "none":
        return None
    from pu_toolbox.estimators.deep.vision import build_wconpu_augmentation

    return build_wconpu_augmentation(
        preprocessing.train_augmentation,
        image_size=preprocessing.input_size[0],
        crop_scale=preprocessing.crop_scale,
        horizontal_flip_probability=preprocessing.horizontal_flip_probability,
        randaugment_num_ops=preprocessing.randaugment_num_ops,
        randaugment_magnitude=preprocessing.randaugment_magnitude,
    )


def _resolve_dataset(dataset: str) -> str:
    canonical = _ALIASES.get(dataset, dataset)
    if canonical not in _IMAGE_CHANNELS:
        raise ValueError(
            f"unknown survey image dataset {dataset!r}; choose from {sorted(_IMAGE_CHANNELS)}."
        )
    return canonical


def _prepare_array(
    X: np.ndarray,
    *,
    expected_shape: tuple[int, int, int] | None,
    expected_channels: int,
) -> tuple[np.ndarray, str, str]:
    values = np.asarray(X)
    if values.ndim != 4:
        raise ValueError("survey images must use a four-dimensional NCHW array.")
    if values.shape[0] == 0:
        raise ValueError("survey image arrays must not be empty.")
    if values.shape[1] != expected_channels:
        raise ValueError(f"survey image dataset expects {expected_channels} channel(s).")
    if expected_shape is not None and values.shape[1:] != expected_shape:
        raise ValueError(
            f"image partition shape must be (N, {expected_shape}); got {values.shape}."
        )
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise ValueError("survey images must contain real numeric values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("survey images must contain only finite values.")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    source_dtype = str(values.dtype)
    if np.issubdtype(values.dtype, np.integer):
        if minimum < 0 or maximum > 255:
            raise ValueError("integer survey images must be in the [0, 255] range.")
        return values.astype(np.float32) / 255.0, "integer_divide_255", source_dtype
    if minimum < 0 or maximum > 1:
        raise ValueError("floating-point survey images must already be in the [0, 1] range.")
    return values.astype(np.float32), "identity_unit_interval", source_dtype


def _validate_augmentation(
    name: str,
    crop_scale: tuple[float, float],
    horizontal_flip_probability: float,
    randaugment_num_ops: int,
    randaugment_magnitude: int,
) -> None:
    if name not in {"none", "simaugment", "randaugment"}:
        raise ValueError("train_augmentation must be 'none', 'simaugment', or 'randaugment'.")
    if len(crop_scale) != 2 or not 0 < crop_scale[0] <= crop_scale[1] <= 1:
        raise ValueError("crop_scale must satisfy 0 < low <= high <= 1.")
    if not 0 <= horizontal_flip_probability <= 1:
        raise ValueError("horizontal_flip_probability must be in [0, 1].")
    if (
        isinstance(randaugment_num_ops, bool)
        or not isinstance(randaugment_num_ops, int)
        or randaugment_num_ops < 1
    ):
        raise ValueError("randaugment_num_ops must be a positive integer.")
    if (
        isinstance(randaugment_magnitude, bool)
        or not isinstance(randaugment_magnitude, int)
        or not 0 <= randaugment_magnitude <= 30
    ):
        raise ValueError("randaugment_magnitude must be an integer in [0, 30].")


def _validate_role(role: str) -> None:
    if role not in {"train", "pu_val", "clean_val", "test"}:
        raise ValueError("role must be 'train', 'pu_val', 'clean_val', or 'test'.")


def _array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(values.shape).encode())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
