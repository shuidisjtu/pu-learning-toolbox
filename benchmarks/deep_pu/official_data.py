# ruff: noqa: N803, N806

"""Official-data smoke runner and reproducible PU dataset construction."""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from .._common import canonical_hash
from .runner import SUPPORTED_METHODS, _build_estimator, _git_value

PUBLIC_DATASETS = {
    "mnist",
    "fashion_mnist",
    "cifar10",
    "svhn",
    "stl10",
    "twenty_newsgroups",
}
RESTRICTED_DATASETS = {"alzheimer_mri", "celeba"}


@dataclass(frozen=True)
class PUDataset:
    """Materialized, auditable train/test data for one PU trial."""

    X_train: np.ndarray
    y_pu: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    manifest: dict[str, Any]
    X_validation: np.ndarray | None = None
    y_validation_pu: np.ndarray | None = None
    y_validation_true: np.ndarray | None = None


def _require_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"dataset.{name} must be a positive integer")
    return value


def load_official_data_config(path: str | Path) -> dict[str, Any]:
    """Load a deliberately non-paper-claiming official-data run config."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("official-data config schema_version must be 1")
    if config.get("protocol") != "official_data":
        raise ValueError("official-data runner requires protocol='official_data'")
    if config.get("paper_claim") is not False:
        raise ValueError("official-data smoke runs must set paper_claim=false")
    if config.get("fidelity_level") not in {"smoke", "paper_protocol"}:
        raise ValueError("fidelity_level must be 'smoke' or 'paper_protocol'")

    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("seeds must be a non-empty list of non-negative integers")

    dataset = config.get("dataset", {})
    name = dataset.get("name")
    if name not in PUBLIC_DATASETS:
        raise ValueError(f"dataset.name must be one of {sorted(PUBLIC_DATASETS)}")
    positive_classes = dataset.get("positive_classes")
    if not isinstance(positive_classes, list) or not positive_classes:
        raise ValueError("dataset.positive_classes must be a non-empty list")
    for field in ("n_labeled_positive", "n_unlabeled", "n_test"):
        _require_int(dataset, field)
    validation_fields = {"validation_positive", "validation_unlabeled"}
    configured_validation = validation_fields & set(dataset)
    if configured_validation and configured_validation != validation_fields:
        raise ValueError("validation_positive and validation_unlabeled must be set together")
    for field in configured_validation:
        _require_int(dataset, field)
    clean_validation_fraction = dataset.get("clean_validation_fraction", 0.0)
    if isinstance(clean_validation_fraction, bool) or not isinstance(
        clean_validation_fraction, int | float
    ):
        raise ValueError("dataset.clean_validation_fraction must be a number in [0, 1)")
    if not 0.0 <= float(clean_validation_fraction) < 1.0:
        raise ValueError("dataset.clean_validation_fraction must be in [0, 1)")
    if configured_validation and clean_validation_fraction:
        raise ValueError(
            "PU validation counts and clean_validation_fraction are mutually exclusive"
        )
    unlabeled_class_prior = dataset.get("unlabeled_class_prior")
    if unlabeled_class_prior is not None and (
        isinstance(unlabeled_class_prior, bool)
        or not isinstance(unlabeled_class_prior, int | float)
        or not 0.0 < float(unlabeled_class_prior) < 1.0
    ):
        raise ValueError("dataset.unlabeled_class_prior must be a number in (0, 1)")

    methods = config.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("methods must be a non-empty object")
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"unsupported deep PU methods: {unknown}")
    for method, method_config in methods.items():
        if not isinstance(method_config, dict):
            raise ValueError(f"methods.{method} must be an object")
        class_prior_mode = method_config.get("class_prior_mode", "known")
        if class_prior_mode not in {"known", "estimate"}:
            raise ValueError(f"methods.{method}.class_prior_mode must be 'known' or 'estimate'")
        if class_prior_mode == "estimate" and method != "infomax_pu":
            raise ValueError("estimated class priors are currently supported for infomax_pu")
        parameters = method_config.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"methods.{method}.parameters must be an object")
        prior_config = parameters.get("prior_estimator")
        if prior_config is not None:
            if method != "infomax_pu" or not isinstance(prior_config, dict):
                raise ValueError("structured prior_estimator is only supported for infomax_pu")
            if prior_config.get("name") != "kernel_mean":
                raise ValueError("prior_estimator.name must be 'kernel_mean'")
            if not isinstance(prior_config.get("parameters", {}), dict):
                raise ValueError("prior_estimator.parameters must be an object")
        vision_config = parameters.get("vision")
        if vision_config is not None:
            if method != "weighted_contrastive_pu" or not isinstance(vision_config, dict):
                raise ValueError(
                    "structured vision config is only supported for weighted_contrastive_pu"
                )
            required_vision = {"backbone", "weak_augmentation", "strong_augmentation"}
            missing_vision = required_vision - set(vision_config)
            if missing_vision:
                raise ValueError(f"vision config is missing: {sorted(missing_vision)}")
            for section in required_vision:
                value = vision_config[section]
                if not isinstance(value, dict) or not isinstance(value.get("name"), str):
                    raise ValueError(f"vision.{section}.name must be configured")
        fit_config = method_config.get("fit", {})
        if not isinstance(fit_config, dict):
            raise ValueError(f"methods.{method}.fit must be an object")
        use_validation = fit_config.get("use_validation_for_early_stopping", False)
        if not isinstance(use_validation, bool):
            raise ValueError("use_validation_for_early_stopping must be boolean")
        if use_validation and (method != "infomax_pu" or not configured_validation):
            raise ValueError(
                "validation forwarding requires infomax_pu and a configured validation split"
            )
        selection_config = method_config.get("model_selection")
        if selection_config is not None:
            if method != "weighted_contrastive_pu" or not isinstance(selection_config, dict):
                raise ValueError(
                    "clean validation model selection is only supported for weighted_contrastive_pu"
                )
            if selection_config.get("strategy") != "clean_validation_grid":
                raise ValueError("model_selection.strategy must be 'clean_validation_grid'")
            if not clean_validation_fraction:
                raise ValueError(
                    "clean validation model selection requires clean_validation_fraction"
                )
            if selection_config.get("metric", "accuracy") not in {
                "accuracy",
                "balanced_accuracy",
                "roc_auc",
            }:
                raise ValueError("model_selection.metric is unsupported")
            if selection_config.get("refit", True) is not True:
                raise ValueError("model_selection.refit must be true")
            parameter_grid = selection_config.get("parameter_grid")
            if not isinstance(parameter_grid, dict) or not parameter_grid:
                raise ValueError("model_selection.parameter_grid must be a non-empty object")
            forbidden = {"class_prior", "random_state", "vision", "device"} & set(parameter_grid)
            if forbidden:
                raise ValueError(f"model-selection parameters are runner-controlled: {forbidden}")
            if any(
                not isinstance(values, list) or not values for values in parameter_grid.values()
            ):
                raise ValueError("every model-selection parameter must have a non-empty value list")
    if dataset.get("representation") == "image_tensor" and not any(
        "vision" in method.get("parameters", {}) for method in methods.values()
    ):
        raise ValueError("image_tensor representation requires a configured vision pipeline")
    return config


def make_pu_split(
    X_train: np.ndarray,
    labels_train: np.ndarray,
    X_test: np.ndarray,
    labels_test: np.ndarray,
    *,
    positive_classes: list[int | str],
    n_labeled_positive: int,
    n_unlabeled: int,
    n_test: int,
    seed: int,
    validation_positive: int = 0,
    validation_unlabeled: int = 0,
    clean_validation_fraction: float = 0.0,
    unlabeled_class_prior: float | None = None,
) -> PUDataset:
    """Create a deterministic case-control split without train/test leakage."""
    X_train = np.asarray(X_train)
    labels_train = np.asarray(labels_train)
    X_test = np.asarray(X_test)
    labels_test = np.asarray(labels_test)
    if len(X_train) != len(labels_train) or len(X_test) != len(labels_test):
        raise ValueError("feature and label lengths do not align")
    if X_train.ndim < 2 or X_test.ndim < 2 or X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("train/test features must be aligned batch-first arrays")

    train_binary = np.isin(labels_train, positive_classes).astype(int)
    test_binary = np.isin(labels_test, positive_classes).astype(int)
    if not 0.0 <= clean_validation_fraction < 1.0:
        raise ValueError("clean_validation_fraction must be in [0, 1)")
    if clean_validation_fraction and (validation_positive or validation_unlabeled):
        raise ValueError("clean and PU validation protocols are mutually exclusive")
    if validation_positive < 0 or validation_unlabeled < 0:
        raise ValueError("validation sample counts must be non-negative")
    if (validation_positive == 0) != (validation_unlabeled == 0):
        raise ValueError("validation positive and unlabeled counts must both be zero or positive")
    if unlabeled_class_prior is not None and not 0.0 < unlabeled_class_prior < 1.0:
        raise ValueError("unlabeled_class_prior must be in (0, 1)")
    rng = np.random.default_rng(seed)
    source_indices = np.arange(len(X_train))
    clean_validation_size = int(round(clean_validation_fraction * len(X_train)))
    clean_validation_indices = (
        rng.choice(source_indices, size=clean_validation_size, replace=False)
        if clean_validation_size
        else np.empty(0, dtype=int)
    )
    training_source = np.setdiff1d(
        source_indices,
        clean_validation_indices,
        assume_unique=False,
    )
    positive_pool = training_source[train_binary[training_source] == 1]
    total_labeled_positive = n_labeled_positive + validation_positive
    if total_labeled_positive > len(positive_pool):
        raise ValueError("not enough training positives for n_labeled_positive")
    total_selected = total_labeled_positive + n_unlabeled + validation_unlabeled
    if total_selected > len(training_source):
        raise ValueError("requested labeled and unlabeled samples exceed training data")
    if n_test > len(X_test):
        raise ValueError("requested n_test exceeds test data")

    all_labeled = rng.choice(positive_pool, size=total_labeled_positive, replace=False)
    labeled = all_labeled[:n_labeled_positive]
    validation_labeled = all_labeled[n_labeled_positive:]
    remaining = np.setdiff1d(training_source, all_labeled, assume_unique=False)
    if unlabeled_class_prior is None:
        all_unlabeled = rng.choice(
            remaining, size=n_unlabeled + validation_unlabeled, replace=False
        )
        unlabeled = all_unlabeled[:n_unlabeled]
        validation_unlabeled_indices = all_unlabeled[n_unlabeled:]
    else:
        remaining_positive = remaining[train_binary[remaining] == 1]
        remaining_negative = remaining[train_binary[remaining] == 0]

        def sample_unlabeled(size: int) -> np.ndarray:
            nonlocal remaining_positive, remaining_negative
            n_positive = int(round(size * unlabeled_class_prior))
            n_negative = size - n_positive
            if n_positive > len(remaining_positive) or n_negative > len(remaining_negative):
                raise ValueError(
                    "not enough class-specific samples for requested unlabeled_class_prior"
                )
            selected_positive = rng.choice(remaining_positive, size=n_positive, replace=False)
            selected_negative = rng.choice(remaining_negative, size=n_negative, replace=False)
            remaining_positive = np.setdiff1d(
                remaining_positive, selected_positive, assume_unique=False
            )
            remaining_negative = np.setdiff1d(
                remaining_negative, selected_negative, assume_unique=False
            )
            selected = np.concatenate([selected_positive, selected_negative])
            rng.shuffle(selected)
            return selected

        unlabeled = sample_unlabeled(n_unlabeled)
        validation_unlabeled_indices = sample_unlabeled(validation_unlabeled)
    selected_test = rng.choice(len(X_test), size=n_test, replace=False)
    if np.unique(test_binary[selected_test]).size < 2:
        raise ValueError("sampled test split contains only one true class")

    train_indices = np.concatenate([labeled, unlabeled])
    y_pu = np.concatenate(
        [np.ones(n_labeled_positive, dtype=int), np.zeros(n_unlabeled, dtype=int)]
    )
    pu_validation_indices = np.concatenate([validation_labeled, validation_unlabeled_indices])
    validation_indices = (
        clean_validation_indices if clean_validation_size else pu_validation_indices
    )
    all_indices = np.concatenate([train_indices, validation_indices, selected_test])
    digest = hashlib.sha256(all_indices.astype("<i8").tobytes()).hexdigest()
    manifest = {
        "seed": seed,
        "positive_classes": list(positive_classes),
        "n_labeled_positive": n_labeled_positive,
        "n_unlabeled": n_unlabeled,
        "n_test": n_test,
        "validation_positive": validation_positive,
        "validation_unlabeled": validation_unlabeled,
        "validation_kind": (
            "clean" if clean_validation_size else "pu" if len(pu_validation_indices) else "none"
        ),
        "clean_validation_fraction": float(clean_validation_fraction),
        "clean_validation_size": clean_validation_size,
        "hidden_unlabeled_positive_rate": float(train_binary[unlabeled].mean()),
        "target_unlabeled_class_prior": unlabeled_class_prior,
        "test_positive_rate": float(test_binary[selected_test].mean()),
        "split_indices_sha256": digest,
        "labeled_unlabeled_overlap": int(np.intersect1d(labeled, unlabeled).size),
        "train_validation_overlap": int(np.intersect1d(train_indices, validation_indices).size),
    }
    X_validation = None
    y_validation_pu = None
    y_validation_true = None
    if len(validation_indices):
        X_validation = np.asarray(X_train[validation_indices], dtype=np.float32)
        y_validation_true = train_binary[validation_indices]
        manifest["validation_true_positive_rate"] = float(y_validation_true.mean())
        if not clean_validation_size:
            y_validation_pu = np.concatenate(
                [
                    np.ones(validation_positive, dtype=int),
                    np.zeros(validation_unlabeled, dtype=int),
                ]
            )
    return PUDataset(
        X_train=np.asarray(X_train[train_indices], dtype=np.float32),
        y_pu=y_pu,
        X_test=np.asarray(X_test[selected_test], dtype=np.float32),
        y_test=test_binary[selected_test],
        manifest=manifest,
        X_validation=X_validation,
        y_validation_pu=y_validation_pu,
        y_validation_true=y_validation_true,
    )


def _vision_arrays(
    name: str,
    root: Path,
    *,
    download: bool,
    representation: str,
) -> tuple[np.ndarray, ...]:
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError("vision datasets require the 'research' optional dependencies") from exc

    if name == "svhn":
        train = datasets.SVHN(root=str(root), split="train", download=download)
        test = datasets.SVHN(root=str(root), split="test", download=download)
    elif name == "stl10":
        train = datasets.STL10(root=str(root), split="train", download=download)
        test = datasets.STL10(root=str(root), split="test", download=download)
    else:
        classes = {
            "mnist": datasets.MNIST,
            "fashion_mnist": datasets.FashionMNIST,
            "cifar10": datasets.CIFAR10,
        }
        dataset_class = classes[name]
        train = dataset_class(root=str(root), train=True, download=download)
        test = dataset_class(root=str(root), train=False, download=download)
    train_data = np.asarray(train.data)
    test_data = np.asarray(test.data)
    train_labels = np.asarray(getattr(train, "targets", getattr(train, "labels", None)))
    test_labels = np.asarray(getattr(test, "targets", getattr(test, "labels", None)))
    scale = 255.0
    if representation == "image_tensor":
        if train_data.ndim == 3:
            train_data = train_data[:, None, :, :]
            test_data = test_data[:, None, :, :]
        elif train_data.shape[-1] in {1, 3}:
            train_data = np.moveaxis(train_data, -1, 1)
            test_data = np.moveaxis(test_data, -1, 1)
        return (
            train_data.astype(np.float32) / scale,
            train_labels,
            test_data.astype(np.float32) / scale,
            test_labels,
        )
    if representation != "flattened_pixels":
        raise ValueError("vision representation must be 'flattened_pixels' or 'image_tensor'")
    return (
        train_data.reshape(len(train_data), -1).astype(np.float32) / scale,
        train_labels,
        test_data.reshape(len(test_data), -1).astype(np.float32) / scale,
        test_labels,
    )


def _newsgroups_arrays(
    root: Path,
    *,
    download: bool,
    max_features: int,
) -> tuple[np.ndarray, ...]:
    from sklearn.datasets import fetch_20newsgroups

    kwargs = {
        "data_home": root,
        "download_if_missing": download,
        "remove": ("headers", "footers", "quotes"),
    }
    train = fetch_20newsgroups(subset="train", **kwargs)
    test = fetch_20newsgroups(subset="test", **kwargs)
    vectorizer = TfidfVectorizer(max_features=max_features, dtype=np.float32)
    X_train = vectorizer.fit_transform(train.data).toarray()
    X_test = vectorizer.transform(test.data).toarray()
    return X_train, np.asarray(train.target), X_test, np.asarray(test.target)


def load_public_dataset(
    config: dict[str, Any],
    root: str | Path,
    *,
    download: bool,
) -> tuple[np.ndarray, ...]:
    """Load and flatten one supported public dataset using its canonical split."""
    name = config["name"]
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if name == "twenty_newsgroups":
        return _newsgroups_arrays(
            root,
            download=download,
            max_features=int(config.get("max_features", 2000)),
        )
    return _vision_arrays(
        name,
        root,
        download=download,
        representation=config.get("representation", "flattened_pixels"),
    )


def build_dataset(
    dataset_config: dict[str, Any],
    root: str | Path,
    *,
    seed: int,
    download: bool,
    loader: Callable[..., tuple[np.ndarray, ...]] = load_public_dataset,
) -> PUDataset:
    """Load a public dataset and apply the configured deterministic PU split."""
    arrays = loader(dataset_config, root, download=download)
    return make_pu_split(
        *arrays,
        positive_classes=dataset_config["positive_classes"],
        n_labeled_positive=int(dataset_config["n_labeled_positive"]),
        n_unlabeled=int(dataset_config["n_unlabeled"]),
        n_test=int(dataset_config["n_test"]),
        seed=seed,
        validation_positive=int(dataset_config.get("validation_positive", 0)),
        validation_unlabeled=int(dataset_config.get("validation_unlabeled", 0)),
        clean_validation_fraction=float(dataset_config.get("clean_validation_fraction", 0.0)),
        unlabeled_class_prior=(
            None
            if dataset_config.get("unlabeled_class_prior") is None
            else float(dataset_config["unlabeled_class_prior"])
        ),
    )


def environment_preflight(
    config: dict[str, Any],
    *,
    data_root: str | Path,
    download: bool,
) -> dict[str, Any]:
    """Report execution readiness without interpreting blockers as results."""
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_devices = int(torch.cuda.device_count())
    except ImportError:
        cuda_available = False
        cuda_devices = 0

    requested_device = config.get("runtime", {}).get("device", "cpu")
    dataset_name = config["dataset"]["name"]
    blockers: list[str] = []
    warnings: list[str] = []
    if requested_device.startswith("cuda") and not cuda_available:
        blockers.append("CUDA was requested but no usable CUDA device is available")
    if "dgpu" in config["methods"] and not config["methods"]["dgpu"].get("generator_backend"):
        blockers.append("DGPU requires a configured conditional EDM generator backend")
    if dataset_name in RESTRICTED_DATASETS:
        blockers.append(f"{dataset_name} requires separately authorized dataset access")
    if config["fidelity_level"] == "paper_protocol":
        warnings.append("paper_protocol validates orchestration but is not a paper result")
    if (
        config["dataset"].get("representation") == "flattened_pixels"
        and config["dataset"].get("representation_fidelity") != "paper"
    ):
        warnings.append("flattened pixels do not reproduce the paper visual backbone")

    return {
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "requested_device": requested_device,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_devices,
        "dataset": dataset_name,
        "data_root": str(Path(data_root).expanduser().resolve()),
        "download_enabled": download,
    }


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_worktree_dirty(project_root: Path, output: Path) -> bool | None:
    """Check pre-run source state without counting this runner's output."""
    command = ["git", "status", "--porcelain", "--untracked-files=all"]
    try:
        relative_output = output.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        pass
    else:
        command.extend(
            [
                "--",
                ".",
                f":(exclude){relative_output}",
                f":(exclude){relative_output}/**",
            ]
        )
    status = _git_value(project_root, command)
    return None if status is None else bool(status)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_artifacts(data_root: str | Path, dataset_name: str) -> list[dict[str, Any]]:
    """Hash canonical downloaded archives without copying data into the repository."""
    folder_names = {
        "mnist": "MNIST",
        "fashion_mnist": "FashionMNIST",
        "cifar10": "cifar-10-batches-py",
    }
    root = Path(data_root).expanduser().resolve()
    folder = folder_names.get(dataset_name)
    candidates = sorted((root / folder / "raw").glob("*.gz")) if folder else []
    if dataset_name == "cifar10":
        candidates = sorted(root.glob("cifar-10-python.tar.gz"))
    elif dataset_name == "svhn":
        candidates = sorted(root.glob("*_32x32.mat"))
    elif dataset_name == "stl10":
        candidates = sorted(root.glob("stl10_binary.tar.gz"))
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in candidates
    ]


def _grid_candidates(parameter_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a deterministic Cartesian parameter grid."""
    names = list(parameter_grid)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(parameter_grid[name] for name in names))
    ]


def _clean_validation_score(
    estimator: Any,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    metric: str,
) -> float:
    scores = np.asarray(estimator.decision_function(X_validation), dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("model-selection validation scores contain NaN or Inf")
    predictions = (scores >= 0.0).astype(int)
    if metric == "accuracy":
        return float(accuracy_score(y_validation, predictions))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_validation, predictions))
    if np.unique(y_validation).size < 2:
        raise ValueError("roc_auc model selection requires both validation classes")
    return float(roc_auc_score(y_validation, scores))


def _select_wconpu_parameters(
    method: str,
    method_config: dict[str, Any],
    dataset: PUDataset,
    *,
    class_prior: float,
    seed: int,
    selection_rows: list[dict[str, Any]],
    selection_path: Path,
) -> tuple[dict[str, Any], float, int]:
    """Evaluate a clean-validation grid and persist each completed candidate."""
    if dataset.X_validation is None or dataset.y_validation_true is None:
        raise ValueError("clean validation model selection requires true validation labels")
    if dataset.manifest.get("validation_kind") != "clean":
        raise ValueError("model selection cannot consume a PU-labeled validation split")

    selection_config = method_config["model_selection"]
    metric = selection_config.get("metric", "accuracy")
    candidates = _grid_candidates(selection_config["parameter_grid"])
    existing = {
        (str(row["method"]), int(row["seed"]), str(row["parameter_json"])): row
        for row in selection_rows
    }
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for candidate_index, candidate in enumerate(candidates):
        parameter_json = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        key = (method, seed, parameter_json)
        if key in existing:
            score = float(existing[key]["validation_score"])
        else:
            candidate_config = dict(method_config)
            candidate_config["parameters"] = {
                **method_config.get("parameters", {}),
                **candidate,
            }
            started = time.perf_counter()
            estimator = _build_estimator(
                method,
                candidate_config,
                class_prior=class_prior,
                seed=seed,
            ).fit(dataset.X_train, dataset.y_pu)
            score = _clean_validation_score(
                estimator,
                dataset.X_validation,
                dataset.y_validation_true,
                metric,
            )
            row = {
                "method": method,
                "seed": seed,
                "candidate_index": candidate_index,
                "parameter_json": parameter_json,
                "validation_metric": metric,
                "validation_score": score,
                "elapsed_seconds": time.perf_counter() - started,
                **{f"parameter_{name}": value for name, value in candidate.items()},
            }
            selection_rows.append(row)
            existing[key] = row
            pd.DataFrame(selection_rows).sort_values(["method", "seed", "candidate_index"]).to_csv(
                selection_path, index=False
            )
        scored.append((score, candidate_index, candidate))

    best_score, _, best_parameters = max(scored, key=lambda item: (item[0], -item[1]))
    return best_parameters, float(best_score), len(candidates)


def run_official_data_benchmark(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    data_root: str | Path,
    download: bool = False,
    resume: bool = False,
    loader: Callable[..., tuple[np.ndarray, ...]] = load_public_dataset,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run public-data trials and persist every completed seed incrementally."""
    output = Path(output_dir)
    project_root = Path(__file__).resolve().parents[2]
    worktree_dirty_before_run = _git_worktree_dirty(project_root, output)
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_config.json"
    if resume and resolved_path.exists():
        previous_config = json.loads(resolved_path.read_text(encoding="utf-8"))
        if canonical_hash(previous_config) != canonical_hash(config):
            raise ValueError("cannot resume: current config differs from resolved_config.json")
    preflight = environment_preflight(config, data_root=data_root, download=download)
    _write_json(output / "preflight.json", preflight)
    if not preflight["ready"]:
        raise RuntimeError("preflight blocked execution: " + "; ".join(preflight["blockers"]))

    trials_path = output / "trials.csv"
    previous = pd.read_csv(trials_path) if resume and trials_path.exists() else pd.DataFrame()
    completed = (
        set(zip(previous["method"], previous["seed"], strict=False))
        if not previous.empty
        else set()
    )
    rows: list[dict[str, Any]] = previous.to_dict("records")
    selection_path = output / "model_selection.csv"
    previous_selection = (
        pd.read_csv(selection_path) if resume and selection_path.exists() else pd.DataFrame()
    )
    selection_rows: list[dict[str, Any]] = previous_selection.to_dict("records")
    split_manifests: dict[str, Any] = {}
    class_prior = float(config["dataset"]["class_prior"])
    for seed in config["seeds"]:
        dataset = build_dataset(
            config["dataset"], data_root, seed=seed, download=download, loader=loader
        )
        split_manifests[str(seed)] = dataset.manifest
        for method, method_config in config["methods"].items():
            if (method, seed) in completed:
                continue
            started = time.perf_counter()
            class_prior_mode = method_config.get("class_prior_mode", "known")
            if class_prior_mode not in {"known", "estimate"}:
                raise ValueError("class_prior_mode must be 'known' or 'estimate'")
            if class_prior_mode == "estimate" and method != "infomax_pu":
                raise ValueError("estimated class priors are currently supported for infomax_pu")
            selected_parameters: dict[str, Any] = {}
            selection_score: float | None = None
            selection_candidates = 0
            final_method_config = method_config
            if method_config.get("model_selection") is not None:
                selected_parameters, selection_score, selection_candidates = (
                    _select_wconpu_parameters(
                        method,
                        method_config,
                        dataset,
                        class_prior=class_prior,
                        seed=seed,
                        selection_rows=selection_rows,
                        selection_path=selection_path,
                    )
                )
                final_method_config = dict(method_config)
                final_method_config["parameters"] = {
                    **method_config.get("parameters", {}),
                    **selected_parameters,
                }
            estimator = _build_estimator(
                method,
                final_method_config,
                class_prior=class_prior if class_prior_mode == "known" else None,
                seed=seed,
            )
            fit_parameters: dict[str, Any] = {}
            use_validation = method_config.get("fit", {}).get(
                "use_validation_for_early_stopping", False
            )
            if use_validation:
                if method != "infomax_pu":
                    raise ValueError("validation forwarding is currently supported for infomax_pu")
                if dataset.X_validation is None or dataset.y_validation_pu is None:
                    raise ValueError("validation forwarding requires a configured validation split")
                fit_parameters["validation_data"] = (
                    dataset.X_validation,
                    dataset.y_validation_pu,
                )
            estimator.fit(dataset.X_train, dataset.y_pu, **fit_parameters)
            class_prior_used = float(estimator.class_prior_)
            scores = np.asarray(estimator.decision_function(dataset.X_test), dtype=float)
            predictions = (scores >= 0.0).astype(int)
            rows.append(
                {
                    "protocol": "official_data",
                    "fidelity_level": config["fidelity_level"],
                    "paper_claim": False,
                    "dataset": config["dataset"]["name"],
                    "method": method,
                    "implementation_variant": method_config["variant"],
                    "seed": seed,
                    "n_train": len(dataset.X_train),
                    "n_test": len(dataset.X_test),
                    "true_class_prior": class_prior,
                    "class_prior_mode": class_prior_mode,
                    "class_prior_used": class_prior_used,
                    "class_prior_absolute_error": abs(class_prior_used - class_prior),
                    "selected_parameters": json.dumps(
                        selected_parameters,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "selection_metric": (
                        method_config.get("model_selection", {}).get("metric")
                        if selection_candidates
                        else None
                    ),
                    "selection_score": selection_score,
                    "selection_candidates": selection_candidates,
                    "hidden_unlabeled_positive_rate": dataset.manifest[
                        "hidden_unlabeled_positive_rate"
                    ],
                    "accuracy": float(accuracy_score(dataset.y_test, predictions)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(dataset.y_test, predictions)
                    ),
                    "roc_auc": float(roc_auc_score(dataset.y_test, scores)),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            pd.DataFrame(rows).sort_values(["method", "seed"]).to_csv(trials_path, index=False)

    trials = pd.DataFrame(rows).sort_values(["method", "seed"]).reset_index(drop=True)
    metrics = ["accuracy", "balanced_accuracy", "roc_auc", "elapsed_seconds"]
    summary = (
        trials.groupby(["dataset", "method"])[metrics].agg(["mean", "std", "count"]).reset_index()
    )
    summary.columns = ["_".join(filter(None, item)) for item in summary.columns.to_flat_index()]
    summary.to_csv(output / "summary.csv", index=False)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "official_data",
        "fidelity_level": config["fidelity_level"],
        "paper_claim": False,
        "n_trials": len(trials),
        "n_model_selection_candidates": len(selection_rows),
        "resume_enabled": resume,
        "config_sha256": canonical_hash(config),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": _git_value(project_root, ["git", "rev-parse", "HEAD"]),
        "git_worktree_dirty": worktree_dirty_before_run,
        "dataset_artifacts": _dataset_artifacts(data_root, config["dataset"]["name"]),
        "dataset_splits": split_manifests,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scikit_learn": _version("scikit-learn"),
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
        },
        "claim_policy": config["claim_policy"],
        "known_gaps": config.get("known_gaps", []),
    }
    _write_json(output / "run_manifest.json", manifest)
    _write_json(resolved_path, config)
    return trials, summary
