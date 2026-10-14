"""Small, real split products and archives for offline receipt tests."""

import json

import numpy as np

from pu_toolbox.experiment.datasets import survey_dataset_catalog
from pu_toolbox.experiment.split_archive import dump_index, pack
from pu_toolbox.utils.serialization import canonical_hash

ROLES = ("train", "pu_val", "clean_val", "test")


def write_split(root, dataset="mnist", seed=0, *, constant=False):
    spec = survey_dataset_catalog()[dataset]
    base = root / dataset / f"split_{seed}"
    base.mkdir(parents=True)
    arrays = {}
    ids = {}
    labels = {}
    offset = 0
    for role, size in zip(ROLES, (8, 2, 2, 4), strict=True):
        labels[role] = np.tile([0, 1], size // 2)
        if role == "test" and spec.has_official_test:
            labels[role] = np.ones(size, dtype=int)
        ids[role] = list(range(offset, offset + size))
        offset += size
        if spec.modality == "image":
            channels, height, width = {
                "mnist": (1, 28, 28),
                "fashion_mnist": (1, 28, 28),
                "cifar10": (3, 32, 32),
                "adni": (1, 128, 128),
            }[dataset]
            arrays[role] = np.arange(size * channels * height * width, dtype=np.uint8).reshape(
                size, channels, height, width
            )
        elif spec.modality == "text":
            arrays[role] = np.zeros((size, 384), dtype=np.float32)
            arrays[role][:, 0] = 1
        else:
            features = {"spambase": 57, "connect_4": 126}[dataset]
            arrays[role] = np.tile(np.tile([-1, 1], size // 2)[:, None], (1, features)).astype(
                np.float32
            )
            if constant:
                arrays[role][:, -1] = 0
        np.savez_compressed(base / f"{role}.npz", X=arrays[role], y=labels[role], indices=ids[role])
    source_roles = ROLES[:-1] if spec.has_official_test else ROLES
    manifest = {
        "dataset": dataset,
        "seed": seed,
        "modality": spec.modality,
        "test_source": "official" if spec.has_official_test else "stratified_source_20_percent",
        "role_sizes": {r: len(labels[r]) for r in ROLES},
        "indices": ids,
        "indices_sha256": canonical_hash(ids),
        "role_positive_rates": {r: float(labels[r].mean()) for r in ROLES},
        "class_prior": {
            "population": float(np.concatenate([labels[r] for r in source_roles]).mean()),
            "train": float(labels["train"].mean()),
        },
        "provenance": {"status": "recorded"},
        "preprocessing": {},
    }
    if spec.modality == "image":
        scaled = arrays["train"].astype(np.float32) / 255
        manifest["preprocessing"]["normalization"] = {
            "mean": scaled.mean(axis=(0, 2, 3), dtype=np.float64).tolist(),
            "std": scaled.astype(np.float64).std(axis=(0, 2, 3)).tolist(),
        }
    (base / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return base, manifest


def make_receipt(tmp_path, datasets=("mnist",), seeds=(0,)):
    root = tmp_path / "splits"
    for dataset in datasets:
        for seed in seeds:
            write_split(root, dataset, seed)
    archives = tmp_path / "archives"
    index_path = tmp_path / "index.json"
    dump_index(pack(root, archives), index_path)
    return root, index_path, archives
