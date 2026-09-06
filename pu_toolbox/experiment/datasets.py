# ruff: noqa: N803, N806

"""Survey dataset catalog and deterministic four-way split preparation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split

from .bundle import DatasetBundle, DatasetPart, validate_bundle

Modality = Literal["image", "text", "tabular"]


@dataclass(frozen=True)
class SurveyDatasetSpec:
    """Locked binary mapping and test-source policy for one survey dataset."""

    name: str
    modality: Modality
    positive_classes: tuple[int | str, ...]
    negative_classes: tuple[int | str, ...]
    has_official_test: bool


_CATALOG = {
    "mnist": SurveyDatasetSpec("mnist", "image", (0, 2, 4, 6, 8), (1, 3, 5, 7, 9), True),
    "fashion_mnist": SurveyDatasetSpec(
        "fashion_mnist", "image", (0, 2, 3, 4, 6), (1, 5, 7, 8, 9), True
    ),
    "cifar10": SurveyDatasetSpec("cifar10", "image", (0, 1, 8, 9), (2, 3, 4, 5, 6, 7), True),
    "adni": SurveyDatasetSpec("adni", "image", (0,), (1, 2, 3), False),
    "imdb": SurveyDatasetSpec("imdb", "text", (1,), (0,), True),
    "twenty_newsgroups": SurveyDatasetSpec(
        "twenty_newsgroups", "text", (0, 1, 2, 3), (4, 5, 6), True
    ),
    "spambase": SurveyDatasetSpec("spambase", "tabular", (1,), (0,), False),
    "connect_4": SurveyDatasetSpec("connect_4", "tabular", ("win",), ("loss", "draw"), False),
}

_ALIASES = {
    "20news": "twenty_newsgroups",
    "connect-4": "connect_4",
    "f-mnist": "fashion_mnist",
    "f_mnist": "fashion_mnist",
}


def survey_dataset_catalog() -> dict[str, SurveyDatasetSpec]:
    """Return a copy of the eight-dataset protocol catalog."""
    return dict(_CATALOG)


def binaryize_survey_labels(
    labels: np.ndarray,
    dataset: str | SurveyDatasetSpec,
) -> np.ndarray:
    """Apply the protocol-locked positive/negative class mapping."""
    spec = _resolve_spec(dataset)
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("survey labels must be one-dimensional.")
    included = np.isin(labels, spec.positive_classes + spec.negative_classes)
    if not np.all(included):
        unknown = np.unique(labels[~included]).tolist()
        raise ValueError(
            f"{spec.name} contains labels outside its locked binary mapping: {unknown!r}."
        )
    binary = np.isin(labels, spec.positive_classes).astype(int)
    if np.unique(binary).size != 2:
        raise ValueError(f"{spec.name} data must contain both mapped classes.")
    return binary


def prepare_survey_dataset(
    X_source: np.ndarray,
    y_source: np.ndarray,
    *,
    dataset: str | SurveyDatasetSpec,
    seed: int,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    source_indices: np.ndarray | None = None,
    test_indices: np.ndarray | None = None,
) -> tuple[DatasetBundle, dict[str, Any]]:
    """Prepare the survey's train/PU-val/clean-val/test clean-label bundle.

    Datasets with an official test split require ``X_test``/``y_test``.
    Spambase, Connect-4 and ADNI instead derive a stratified 20% test split
    from ``X_source``. The remaining training source is split 90%/5%/5%.
    """
    spec = _resolve_spec(dataset)
    X_source = np.asarray(X_source)
    y_source = np.asarray(y_source)
    _validate_arrays(X_source, y_source, role="source")
    y_source_binary = binaryize_survey_labels(y_source, spec)
    source_ids = _indices_or_default(source_indices, len(X_source), role="source")

    if spec.has_official_test:
        if X_test is None or y_test is None:
            raise ValueError(f"{spec.name} requires its official X_test and y_test split.")
        X_test_array = np.asarray(X_test)
        y_test_array = np.asarray(y_test)
        _validate_arrays(X_test_array, y_test_array, role="test")
        if X_test_array.shape[1:] != X_source.shape[1:]:
            raise ValueError("source and official test sample shapes must match.")
        y_test_binary = binaryize_survey_labels(y_test_array, spec)
        default_test_ids = np.arange(len(X_source), len(X_source) + len(X_test_array))
        final_test_ids = _indices_or_default(
            test_indices,
            len(X_test_array),
            role="test",
            default=default_test_ids,
        )
        development_positions = np.arange(len(X_source))
        final_test_positions = np.arange(len(X_test_array))
        test_source = "official"
    else:
        if X_test is not None or y_test is not None or test_indices is not None:
            raise ValueError(f"{spec.name} derives test from source; do not pass test arrays.")
        development_positions, final_test_positions = _stratified_split(
            np.arange(len(X_source)),
            y_source_binary,
            test_size=0.2,
            seed=seed,
            stage="derived test",
        )
        X_test_array = X_source
        y_test_binary = y_source_binary
        final_test_ids = source_ids
        test_source = "stratified_source_20_percent"

    development_labels = y_source_binary[development_positions]
    train_positions, validation_positions = _stratified_split(
        development_positions,
        development_labels,
        test_size=0.1,
        seed=seed,
        stage="validation pool",
    )
    validation_labels = y_source_binary[validation_positions]
    pu_val_positions, clean_val_positions = _stratified_split(
        validation_positions,
        validation_labels,
        test_size=0.5,
        seed=seed,
        stage="PU/clean validation",
    )

    bundle = DatasetBundle(
        train=_part(X_source, y_source_binary, source_ids, train_positions),
        pu_val=_part(X_source, y_source_binary, source_ids, pu_val_positions),
        clean_val=_part(X_source, y_source_binary, source_ids, clean_val_positions),
        test=DatasetPart(
            X=X_test_array[final_test_positions],
            labels=y_test_binary[final_test_positions],
            view="clean",
            indices=final_test_ids[final_test_positions],
            for_selection=False,
        ),
    )
    validate_bundle(bundle)

    role_indices = {
        role: _json_indices(getattr(bundle, role).indices)
        for role in ("train", "pu_val", "clean_val", "test")
    }
    manifest = {
        "schema_version": "1.0",
        "dataset": spec.name,
        "modality": spec.modality,
        "seed": int(seed),
        "positive_classes": list(spec.positive_classes),
        "negative_classes": list(spec.negative_classes),
        "test_source": test_source,
        "split_policy": "stratified_90_train_5_pu_val_5_clean_val",
        "role_sizes": {role: len(indices) for role, indices in role_indices.items()},
        "role_positive_rates": {
            role: float(np.mean(getattr(bundle, role).labels))
            for role in ("train", "pu_val", "clean_val", "test")
        },
        "indices": role_indices,
        "indices_sha256": hashlib.sha256(
            json.dumps(role_indices, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return bundle, manifest


def _resolve_spec(dataset: str | SurveyDatasetSpec) -> SurveyDatasetSpec:
    if isinstance(dataset, SurveyDatasetSpec):
        return dataset
    canonical = _ALIASES.get(dataset, dataset)
    try:
        return _CATALOG[canonical]
    except KeyError as exc:
        choices = sorted(_CATALOG)
        raise ValueError(f"unknown survey dataset {dataset!r}; choose from {choices}.") from exc


def _validate_arrays(X: np.ndarray, y: np.ndarray, *, role: str) -> None:
    if X.ndim < 2:
        raise ValueError(f"{role} features must be batch-first with ndim >= 2.")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"{role} features and labels must align.")


def _indices_or_default(
    indices: np.ndarray | None,
    size: int,
    *,
    role: str,
    default: np.ndarray | None = None,
) -> np.ndarray:
    result = np.asarray(default if indices is None and default is not None else indices)
    if indices is None and default is None:
        result = np.arange(size)
    if result.ndim != 1 or len(result) != size:
        raise ValueError(f"{role}_indices must be one-dimensional and align with its data.")
    if len(set(result.tolist())) != size:
        raise ValueError(f"{role}_indices must be unique.")
    return result


def _stratified_split(
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    test_size: float,
    seed: int,
    stage: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        left, right = train_test_split(
            positions,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    except ValueError as exc:
        raise ValueError(
            f"cannot create the stratified {stage} split; provide more samples per class."
        ) from exc
    return np.sort(left), np.sort(right)


def _part(
    X: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    positions: np.ndarray,
) -> DatasetPart:
    return DatasetPart(
        X=X[positions],
        labels=labels[positions],
        view="clean",
        indices=indices[positions],
    )


def _json_indices(indices: np.ndarray) -> list[int | str]:
    return [item.item() if isinstance(item, np.generic) else item for item in indices]
