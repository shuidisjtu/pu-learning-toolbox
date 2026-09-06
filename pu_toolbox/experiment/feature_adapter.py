# ruff: noqa: N803, N806

"""CNN feature adapter and cross-method leaderboard fairness gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .bundle import DatasetBundle, DatasetPart, validate_bundle

TrainingPath = Literal["native_cnn", "cnn_feature_adapter"]
EncoderFitScope = Literal["fixed_external", "train_partition_only"]


@dataclass(frozen=True)
class LeaderboardRunSpec:
    """Fields that must agree before methods may share a leaderboard group."""

    method: str
    dataset: str
    training_path: TrainingPath
    adaptation_level: Literal["source-faithful", "benchmark-adapted"]
    split_sha256: str
    representation_sha256: str
    max_epochs: int
    batch_size_candidates: tuple[int, ...]
    tuning_candidate_count: int
    seeds: tuple[int, ...]


def adapt_image_bundle_to_features(
    bundle: DatasetBundle,
    encoder: object,
    *,
    feature_version: str,
    backbone_manifest: dict[str, Any],
    batch_size: int = 64,
    device: str = "cpu",
    encoder_fit_scope: EncoderFitScope = "fixed_external",
    encoder_fit_indices: np.ndarray | None = None,
) -> tuple[DatasetBundle, dict[str, Any]]:
    """Extract fixed 2-D CNN features for all four dataset roles.

    Extraction always runs in evaluation mode without gradients. The encoder
    state is hashed before and after extraction and any mutation fails loudly.
    ``train_partition_only`` is an auditable caller assertion: its supplied
    fit indices must exactly equal the bundle's train indices. The adapter
    itself never trains the encoder.
    """
    validate_bundle(bundle)
    if not isinstance(feature_version, str) or not feature_version.strip():
        raise ValueError("feature_version must be a non-empty immutable version label.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if encoder_fit_scope not in {"fixed_external", "train_partition_only"}:
        raise ValueError("encoder_fit_scope must be 'fixed_external' or 'train_partition_only'.")

    fit_indices_hash = None
    if encoder_fit_scope == "fixed_external":
        if encoder_fit_indices is not None:
            raise ValueError("fixed_external encoders must not declare per-split fit indices.")
    else:
        if encoder_fit_indices is None:
            raise ValueError("train_partition_only requires encoder_fit_indices.")
        fit_indices = np.asarray(encoder_fit_indices)
        train_indices = np.asarray(bundle.train.indices)
        if fit_indices.ndim != 1 or not np.array_equal(
            np.sort(fit_indices), np.sort(train_indices)
        ):
            raise ValueError(
                "encoder_fit_indices must exactly match the train partition and no other role."
            )
        fit_indices_hash = _json_sha256(_json_scalars(fit_indices))

    canonical_backbone_manifest = _canonical_json_object(
        backbone_manifest, name="backbone_manifest"
    )
    source_shapes = {
        np.asarray(getattr(bundle, role).X).shape[1:]
        for role in ("train", "pu_val", "clean_val", "test")
    }
    if len(source_shapes) != 1:
        raise ValueError("source image shape must match across all four dataset roles.")
    torch, module = _resolve_torch_encoder(encoder, device)
    state_before = _encoder_state_sha256(module)
    was_training = bool(module.training)
    module.eval()
    try:
        role_features = {
            role: _extract_features(
                torch,
                module,
                np.asarray(getattr(bundle, role).X),
                batch_size=batch_size,
                device=device,
            )
            for role in ("train", "pu_val", "clean_val", "test")
        }
    finally:
        module.train(was_training)
    state_after = _encoder_state_sha256(module)
    if state_before != state_after:
        raise RuntimeError("CNN feature extraction mutated the encoder state.")

    adapted = DatasetBundle(
        **{
            role: DatasetPart(
                X=role_features[role],
                labels=getattr(bundle, role).labels,
                view=getattr(bundle, role).view,
                indices=getattr(bundle, role).indices,
                for_selection=getattr(bundle, role).for_selection,
            )
            for role in ("train", "pu_val", "clean_val", "test")
        }
    )
    validate_bundle(adapted)

    feature_dimensions = {values.shape[1] for values in role_features.values()}
    if len(feature_dimensions) != 1:
        raise ValueError("CNN encoder output dimension changed across dataset roles.")
    split_indices = {
        role: _json_scalars(np.asarray(getattr(bundle, role).indices))
        for role in ("train", "pu_val", "clean_val", "test")
    }
    representation_payload = {
        "feature_version": feature_version,
        "encoder_state_sha256": state_before,
        "backbone_manifest": canonical_backbone_manifest,
        "encoder_fit_scope": encoder_fit_scope,
        "encoder_fit_indices_sha256": fit_indices_hash,
    }
    manifest = {
        "schema_version": "1.0",
        "training_path": "cnn_feature_adapter",
        "adaptation_level": "benchmark-adapted",
        "feature_version": feature_version,
        "feature_dimension": feature_dimensions.pop(),
        "batch_size": batch_size,
        "device": device,
        "encoder_mode": "eval_no_grad",
        "encoder_fit_scope": encoder_fit_scope,
        "encoder_fit_indices_sha256": fit_indices_hash,
        "encoder_state_sha256": state_before,
        "backbone_manifest": canonical_backbone_manifest,
        "representation_sha256": _json_sha256(representation_payload),
        "split_sha256": _json_sha256(split_indices),
        "feature_sha256": {
            role: _array_sha256(features) for role, features in role_features.items()
        },
    }
    return adapted, manifest


def partition_fair_leaderboard_runs(
    runs: list[LeaderboardRunSpec],
) -> dict[str, dict[str, Any]]:
    """Validate fairness invariants and partition native/adapter rankings.

    Within one dataset, every method must share the split, seeds, epoch cap,
    batch-size candidate set and tuning candidate count. Within each training
    path, the representation hash must also match. The returned keys include
    the path, so native CNN and feature-adapter results cannot be ranked in
    the same group.
    """
    if not runs:
        raise ValueError("leaderboard fairness validation requires at least one run.")
    by_dataset: dict[str, list[LeaderboardRunSpec]] = {}
    for run in runs:
        _validate_run_spec(run)
        by_dataset.setdefault(run.dataset, []).append(run)

    partitions: dict[str, dict[str, Any]] = {}
    for dataset, dataset_runs in sorted(by_dataset.items()):
        reference = dataset_runs[0]
        shared_fields = (
            "split_sha256",
            "max_epochs",
            "batch_size_candidates",
            "tuning_candidate_count",
            "seeds",
        )
        for run in dataset_runs[1:]:
            mismatches = [
                field for field in shared_fields if getattr(run, field) != getattr(reference, field)
            ]
            if mismatches:
                raise ValueError(
                    f"leaderboard fairness mismatch for dataset {dataset!r}, method "
                    f"{run.method!r}: {', '.join(mismatches)}."
                )

        by_path: dict[str, list[LeaderboardRunSpec]] = {}
        for run in dataset_runs:
            by_path.setdefault(run.training_path, []).append(run)
        for training_path, path_runs in sorted(by_path.items()):
            representation = path_runs[0].representation_sha256
            if any(run.representation_sha256 != representation for run in path_runs[1:]):
                raise ValueError(f"representation_sha256 differs within {dataset}/{training_path}.")
            methods = [run.method for run in path_runs]
            if len(set(methods)) != len(methods):
                raise ValueError(f"duplicate method in {dataset}/{training_path} leaderboard.")
            group_payload = {
                "dataset": dataset,
                "training_path": training_path,
                "methods": sorted(methods),
                "adaptation_levels": {
                    run.method: run.adaptation_level
                    for run in sorted(path_runs, key=lambda x: x.method)
                },
                "split_sha256": reference.split_sha256,
                "representation_sha256": representation,
                "max_epochs": reference.max_epochs,
                "batch_size_candidates": list(reference.batch_size_candidates),
                "tuning_candidate_count": reference.tuning_candidate_count,
                "seeds": list(reference.seeds),
            }
            group_payload["fairness_sha256"] = _json_sha256(group_payload)
            partitions[f"{dataset}/{training_path}"] = group_payload
    return partitions


def _resolve_torch_encoder(encoder: object, device: str):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("CNN feature adaptation requires the 'pu-toolbox[torch]' extra.") from exc
    if not isinstance(encoder, nn.Module):
        raise TypeError("encoder must be a torch.nn.Module.")
    try:
        return torch, encoder.to(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"cannot move CNN encoder to device {device!r}.") from exc


def _extract_features(torch, encoder, X, *, batch_size: int, device: str) -> np.ndarray:
    if X.ndim != 4 or len(X) == 0:
        raise ValueError("CNN feature adapter expects non-empty NCHW arrays for every role.")
    if not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X)):
        raise ValueError("CNN feature adapter inputs must be finite numeric arrays.")
    batches = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            inputs = torch.as_tensor(
                X[start : start + batch_size], dtype=torch.float32, device=device
            )
            output = encoder(inputs)
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                shape = getattr(output, "shape", None)
                raise ValueError(f"CNN encoder must return a 2-D tensor; got shape {shape!r}.")
            batches.append(output.detach().cpu().numpy())
    features = np.asarray(np.concatenate(batches, axis=0), dtype=np.float32)
    if features.shape[0] != len(X) or features.shape[1] == 0:
        raise ValueError("CNN encoder returned an empty or misaligned feature matrix.")
    if not np.all(np.isfinite(features)):
        raise ValueError("CNN encoder returned non-finite features.")
    return features


def _encoder_state_sha256(encoder) -> str:
    digest = hashlib.sha256()
    state = encoder.state_dict()
    if not state:
        raise ValueError("CNN encoder must expose a non-empty state_dict for provenance.")
    for name, tensor in sorted(state.items()):
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(values.dtype).encode())
        digest.update(json.dumps(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _validate_run_spec(run: LeaderboardRunSpec) -> None:
    if (
        not isinstance(run.method, str)
        or not run.method.strip()
        or not isinstance(run.dataset, str)
        or not run.dataset.strip()
    ):
        raise ValueError("leaderboard method and dataset must be non-empty.")
    if run.training_path not in {"native_cnn", "cnn_feature_adapter"}:
        raise ValueError("training_path must be 'native_cnn' or 'cnn_feature_adapter'.")
    if run.adaptation_level not in {"source-faithful", "benchmark-adapted"}:
        raise ValueError("adaptation_level must be 'source-faithful' or 'benchmark-adapted'.")
    if run.training_path == "cnn_feature_adapter" and run.adaptation_level != "benchmark-adapted":
        raise ValueError("cnn_feature_adapter runs must be marked 'benchmark-adapted'.")
    if not _is_sha256(run.split_sha256) or not _is_sha256(run.representation_sha256):
        raise ValueError("split_sha256 and representation_sha256 must be 64-character hex hashes.")
    if (
        isinstance(run.max_epochs, bool)
        or not isinstance(run.max_epochs, int)
        or run.max_epochs < 1
        or isinstance(run.tuning_candidate_count, bool)
        or not isinstance(run.tuning_candidate_count, int)
        or run.tuning_candidate_count < 1
    ):
        raise ValueError("max_epochs and tuning_candidate_count must be positive integers.")
    if not run.batch_size_candidates or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in run.batch_size_candidates
    ):
        raise ValueError("batch_size_candidates must contain positive integers.")
    if (
        not run.seeds
        or any(isinstance(value, bool) or not isinstance(value, int) for value in run.seeds)
        or len(set(run.seeds)) != len(run.seeds)
    ):
        raise ValueError("seeds must be non-empty unique integers.")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _canonical_json_object(value: dict[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty JSON object.")
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite JSON-compatible values.") from exc


def _json_scalars(values: np.ndarray) -> list[int | float | str | bool | None]:
    result = []
    for value in values.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        if value is not None and not isinstance(value, bool | int | float | str):
            raise ValueError("bundle indices must contain JSON scalar values.")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("floating-point bundle indices must be finite.")
        result.append(value)
    return result


def _array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(values.shape).encode())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()
