"""Load and audit the six datasets reported in the PUSB paper's Table 2."""

# Dataset matrices follow sklearn's conventional X/y names.
# ruff: noqa: N803, N806

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_svmlight_file

DEFAULT_MANIFEST = Path(__file__).parent / "configs" / "pusb_table2_datasets.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load a complete, locked Table 2 dataset manifest."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("dataset manifest schema_version must be 1")
    if manifest.get("protocol") != "pusb_table2_dataset_lock":
        raise ValueError("unexpected dataset manifest protocol")
    if manifest.get("status") != "locked":
        raise ValueError("Table 2 dataset manifest is not locked")
    if len(manifest.get("datasets", {})) != 6:
        raise ValueError("Table 2 dataset manifest must contain exactly six datasets")
    return manifest


def _load_raw_dataset(spec: dict[str, Any], data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    paths = [data_root / item["target"] for item in spec["downloads"]]
    for path, item in zip(paths, spec["downloads"], strict=True):
        if not path.is_file():
            raise FileNotFoundError(f"locked dataset target is missing: {path}")
        actual_hash = _sha256(path)
        if actual_hash != item["sha256"]:
            raise ValueError(
                f"dataset SHA-256 mismatch for {path.name}: "
                f"expected {item['sha256']}, found {actual_hash}"
            )

    data_format = spec["format"]
    if data_format in {"svmlight", "svmlight_concat", "svmlight_concat_bzip2"}:
        chunks = [load_svmlight_file(path) for path in paths]
        X = np.vstack([chunk[0].toarray() for chunk in chunks])
        labels = np.concatenate([np.asarray(chunk[1]) for chunk in chunks])
    elif data_format == "dense_whitespace_last_label":
        data = np.loadtxt(paths[0])
        X, labels = data[:, :-1], data[:, -1]
    elif data_format == "dense_csv_last_label":
        data = np.loadtxt(paths[0], delimiter=",")
        X, labels = data[:, :-1], data[:, -1]
    else:
        raise ValueError(f"unsupported Table 2 dataset format: {data_format}")
    return np.asarray(X, dtype=float), np.asarray(labels)


def _map_labels(dataset: str, labels: np.ndarray) -> np.ndarray:
    if dataset == "mushrooms":
        return (labels == 2).astype(int)
    if dataset == "usps":
        return (labels.astype(int) % 2 == 1).astype(int)
    if dataset in {"shuttle", "pageblocks", "connect-4", "spambase"}:
        return (labels == 1).astype(int)
    raise ValueError(f"unsupported Table 2 dataset: {dataset}")


def load_table2_dataset(
    dataset: str,
    data_root: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one locked dataset and apply the released loader's label and scale rules."""
    manifest = load_manifest(manifest_path)
    try:
        spec = manifest["datasets"][dataset]
    except KeyError as exc:
        raise ValueError(f"unknown PUSB Table 2 dataset: {dataset}") from exc
    if spec.get("status") != "locked":
        raise ValueError(f"dataset is not locked: {dataset}")

    X, original_labels = _load_raw_dataset(spec, Path(data_root))
    y = _map_labels(dataset, original_labels)
    expected_shape = (spec["expected_samples"], spec["expected_features"])
    if X.shape != expected_shape:
        raise ValueError(f"{dataset} shape mismatch: expected {expected_shape}, found {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{dataset} contains non-finite features")
    counts = {"negative": int(np.sum(y == 0)), "positive": int(np.sum(y == 1))}
    if counts != spec["class_counts"]:
        raise ValueError(
            f"{dataset} class-count mismatch: expected {spec['class_counts']}, found {counts}"
        )

    maxima = np.max(X, axis=0)
    if np.any(maxima == 0):
        raise ValueError(f"{dataset} has a zero-maximum feature; official scaling is undefined")
    X = X / maxima
    provenance = {
        "dataset": dataset,
        "shape": list(X.shape),
        "class_counts": counts,
        "positive_fraction": counts["positive"] / len(y),
        "target_sha256": {item["target"]: item["sha256"] for item in spec["downloads"]},
    }
    return X, y, provenance


def audit_sampling_schedule(
    y: np.ndarray,
    *,
    initial_seed: int = 2018,
    repetitions: int = 100,
    class_priors: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
    unlabeled_sizes: tuple[int, ...] = (800, 1600, 3200),
    positive_size: int = 400,
    test_size: int = 1000,
    holdout_size: int = 3000,
) -> list[dict[str, Any]]:
    """Audit exact official-loop seeds without fitting models or changing sample semantics."""
    if len(y) <= holdout_size:
        raise ValueError("dataset must contain more rows than the official holdout")
    rows = []
    seed = initial_seed
    for unlabeled_size in unlabeled_sizes:
        for class_prior in class_priors:
            feasible = 0
            min_released_test = test_size
            min_released_unlabeled = unlabeled_size
            failure_reasons: set[str] = set()
            for _ in range(repetitions):
                permutation = np.random.RandomState(seed).permutation(len(y))
                train_y = y[permutation[:-holdout_size]]
                holdout_y = y[permutation[-holdout_size:]]
                train_positive = int(np.sum(train_y == 1))
                train_negative = len(train_y) - train_positive
                holdout_positive = int(np.sum(holdout_y == 1))
                holdout_negative = holdout_size - holdout_positive
                required_u_positive = int(unlabeled_size * class_prior)
                required_u_negative = unlabeled_size - required_u_positive
                required_test_positive = int(test_size * class_prior)
                required_test_negative = test_size - required_test_positive

                released_unlabeled = min(train_positive, required_u_positive) + min(
                    train_negative, required_u_negative
                )
                released_test = min(holdout_positive, required_test_positive) + min(
                    holdout_negative, required_test_negative
                )
                min_released_unlabeled = min(min_released_unlabeled, released_unlabeled)
                min_released_test = min(min_released_test, released_test)
                trial_reasons = []
                if train_positive < positive_size:
                    trial_reasons.append("selected_positive_pool")
                if released_unlabeled < unlabeled_size:
                    trial_reasons.append("unlabeled_pool")
                if released_test < test_size:
                    trial_reasons.append("holdout_pool")
                if trial_reasons:
                    failure_reasons.update(trial_reasons)
                else:
                    feasible += 1
                seed += 1
            rows.append(
                {
                    "unlabeled_size": unlabeled_size,
                    "class_prior": class_prior,
                    "repetitions": repetitions,
                    "strictly_feasible_repetitions": feasible,
                    "minimum_released_unlabeled_size": min_released_unlabeled,
                    "minimum_released_test_size": min_released_test,
                    "failure_reasons": sorted(failure_reasons),
                }
            )
    return rows


def build_audit_report(
    data_root: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Validate all six locks and summarize official sampling feasibility."""
    manifest = load_manifest(manifest_path)
    datasets = []
    for name in manifest["datasets"]:
        _, y, provenance = load_table2_dataset(name, data_root, manifest_path=manifest_path)
        schedule = audit_sampling_schedule(y)
        datasets.append(
            {
                **provenance,
                "strictly_feasible_cells": sum(
                    row["strictly_feasible_repetitions"] == row["repetitions"] for row in schedule
                ),
                "total_cells": len(schedule),
                "sampling_schedule": schedule,
            }
        )
    total_cells = sum(dataset["total_cells"] for dataset in datasets)
    fully_feasible_cells = sum(dataset["strictly_feasible_cells"] for dataset in datasets)
    return {
        "schema_version": 1,
        "protocol": "pusb_table2_data_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(Path(manifest_path)),
        "data_root": str(Path(data_root)),
        "official_schedule": {
            "initial_seed": 2018,
            "repetitions_per_cell": 100,
            "class_priors": [0.2, 0.4, 0.6, 0.8],
            "unlabeled_sizes": [800, 1600, 3200],
            "positive_size": 400,
            "test_size": 1000,
            "holdout_size": 3000,
            "seed_policy": "increment once after every trial across U, prior, repetition",
        },
        "summary": {
            "total_cells": total_cells,
            "fully_feasible_cells": fully_feasible_cells,
            "cells_with_at_least_one_undersized_split": total_cells - fully_feasible_cells,
            "audited_trials": total_cells * 100,
        },
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_audit_report(args.data_root, manifest_path=args.manifest)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
