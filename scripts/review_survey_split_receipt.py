# ruff: noqa: N803, N806

"""Read-only audit of received P1.4 splits; prints a JSON evidence report."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pu_toolbox.core.training_views import ROLES
from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart, validate_bundle
from pu_toolbox.experiment.datasets import survey_dataset_catalog
from pu_toolbox.experiment.split_archive import load_index, verify

DEFAULT_DATASETS = ("spambase", "imdb", "cifar10")
DEFAULT_SEEDS = tuple(range(5))
IMAGE_SHAPES = {
    "mnist": (1, 28, 28),
    "fashion_mnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
    "adni": (1, 128, 128),
}
TABULAR_FEATURES = {"spambase": 57, "connect_4": 126}


def _check_assertions() -> None:
    if not __debug__:
        raise ValueError("Assertions are required; do not run this audit with python -O.")


def _selection(datasets: Sequence[str], seeds: Sequence[int]) -> None:
    catalog = survey_dataset_catalog()
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("datasets must be a nonempty list without duplicates")
    if any(dataset not in catalog for dataset in datasets):
        raise ValueError("unknown dataset; choose from " + ",".join(catalog))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list without duplicates")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")


def _tabular_statistics(X: np.ndarray, entry: dict[str, Any]) -> None:
    assert X.ndim == 2
    mean = X.mean(axis=0, dtype=np.float64)
    std = X.std(axis=0, dtype=np.float64)
    # A train-fitted constant feature becomes exactly zero, not unit variance.
    constant = np.all(X[0] == X, axis=0)
    target = np.where(constant, 0.0, 1.0)
    entry["train_mean_max_abs"] = float(np.max(np.abs(mean)))
    entry["train_std_max_error"] = float(np.max(np.abs(std - target)))
    assert entry["train_mean_max_abs"] < 1e-6
    assert entry["train_std_max_error"] < 1e-6


def _image_statistics(X: np.ndarray, manifest: dict[str, Any], entry: dict[str, Any]) -> None:
    assert X.ndim == 4 and all(size > 0 for size in X.shape)
    channels, height, width = X.shape[1:]
    sums = np.zeros(channels)
    squares = np.zeros(channels)
    n = 0
    for offset in range(0, len(X), 256):
        batch = X[offset : offset + 256].astype(np.float32) / 255.0
        sums += np.sum(batch, axis=(0, 2, 3), dtype=np.float64)
        squares += np.sum(batch.astype(np.float64) ** 2, axis=(0, 2, 3))
        n += len(batch) * height * width
    mean = sums / n
    std = np.sqrt(np.maximum(squares / n - mean**2, 0))
    ref = manifest["preprocessing"]["normalization"]
    assert np.asarray(ref["mean"]).shape == np.asarray(ref["std"]).shape == (channels,)
    entry["channel_mean"] = mean.tolist()
    entry["channel_std"] = std.tolist()
    entry["channel_mean_max_error"] = float(np.max(np.abs(mean - ref["mean"])))
    entry["channel_std_max_error"] = float(np.max(np.abs(std - ref["std"])))
    # Float32 scaling is bounded at its actual precision; integrity is byte-exact.
    entry["channel_statistics_atol"] = float(np.finfo(np.float32).eps)
    assert np.allclose(mean, ref["mean"], atol=np.finfo(np.float32).eps, rtol=0)
    assert np.allclose(std, ref["std"], atol=np.finfo(np.float32).eps, rtol=0)


def audit_split(base: Path, *, dataset: str, seed: int) -> dict[str, Any]:
    """Audit one split's semantic contracts (not its receipt authenticity)."""
    _check_assertions()
    _selection((dataset,), (seed,))
    spec = survey_dataset_catalog()[dataset]
    base = Path(base)
    manifest_path = base / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == dataset and manifest["seed"] == seed
    assert manifest["modality"] == spec.modality
    expected_test_source = "official" if spec.has_official_test else "stratified_source_20_percent"
    assert manifest["test_source"] == expected_test_source
    parts = {}
    entry = {"dataset": dataset, "seed": seed, "roles": {}, "issues": []}
    for role in ROLES:
        with np.load(base / f"{role}.npz", allow_pickle=False) as z:
            X, y, ids = z["X"], z["y"], z["indices"]
        assert len(X) == len(y) == len(ids) == manifest["role_sizes"][role]
        assert ids.tolist() == manifest["indices"][role]
        assert np.isfinite(X).all()
        if spec.modality == "image":
            assert X.dtype == np.uint8
            assert X.ndim == 4 and X.shape[1:] == IMAGE_SHAPES[dataset]
        elif spec.modality == "tabular":
            assert X.ndim == 2 and X.shape[1] == TABULAR_FEATURES[dataset]
        assert np.isclose(y.mean(), manifest["role_positive_rates"][role], rtol=0, atol=1e-14)
        parts[role] = DatasetPart(
            X=X, labels=y, indices=ids, view="clean", for_selection=role != "test"
        )
        entry["roles"][role] = {
            "shape": list(X.shape),
            "dtype": str(X.dtype),
            "positive_rate": float(y.mean()),
            "finite": True,
        }
        if spec.modality == "text":
            assert X.ndim == 2 and X.shape[1] == 384
            norm = np.linalg.norm(X, axis=1)
            assert np.allclose(norm, 1, atol=3e-7, rtol=0)
            entry["roles"][role]["norm_max_error"] = float(np.max(np.abs(norm - 1)))
    validate_bundle(DatasetBundle(**parts))
    population_roles = ROLES[:-1] if spec.has_official_test else ROLES
    population_y = np.concatenate([parts[r].labels for r in population_roles])
    assert np.isclose(
        population_y.mean(), manifest["class_prior"]["population"], atol=1e-14, rtol=0
    )
    assert np.isclose(
        parts["train"].labels.mean(), manifest["class_prior"]["train"], atol=1e-14, rtol=0
    )
    assert manifest["provenance"]["status"] == "recorded"
    entry["bundle_contract"] = "passed"
    entry["population_prior"] = float(population_y.mean())
    entry["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if spec.modality == "tabular":
        _tabular_statistics(parts["train"].X, entry)
    elif spec.modality == "image":
        _image_statistics(parts["train"].X, manifest, entry)
    return entry


def audit_receipt(
    splits: Path,
    index_path: Path,
    archive_dir: Path,
    *,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Verify the complete receipt, then audit only the requested dataset/seed pairs."""
    _check_assertions()
    _selection(datasets, seeds)
    splits = Path(splits)
    index = load_index(index_path)
    # Never filter the trusted index: selection limits semantics, not authenticity.
    report = {"integrity": verify(splits, index, archive_dir=archive_dir), "splits": []}
    if not report["integrity"]["ok"]:
        raise SystemExit("Receipt integrity check failed: " + json.dumps(report["integrity"]))
    report["test_file_hash_count"] = {}
    for dataset in datasets:
        test_hashes = []
        for seed in seeds:
            split_name = f"split_{seed}"
            if dataset not in index["datasets"] or split_name not in index["datasets"][dataset]:
                raise ValueError(f"{dataset}/{split_name} not in receipt index")
            report["splits"].append(
                audit_split(splits / dataset / split_name, dataset=dataset, seed=seed)
            )
            test_hashes.append(
                index["datasets"][dataset][split_name]["files"]["test.npz"]["sha256"]
            )
        report["test_file_hash_count"][dataset] = len(set(test_hashes))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Print the evidence report; reject optimized Python before any receipt reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("docs/research/pu_survey/data/split_artifacts_index.json"),
    )
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="comma-separated catalog dataset names",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, DEFAULT_SEEDS)),
        help="comma-separated nonnegative seeds",
    )
    args = parser.parse_args(argv)
    try:
        _check_assertions()
        datasets = tuple(name.strip() for name in args.datasets.split(","))
        seeds = tuple(int(seed.strip()) for seed in args.seeds.split(","))
        _selection(datasets, seeds)
    except ValueError as exc:
        parser.error(str(exc))
    report = audit_receipt(
        args.splits, args.index, args.archive_dir, datasets=datasets, seeds=seeds
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
