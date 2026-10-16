"""Synthetic four-role fixtures for temporary LBE comparisons."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_lbe_comparison as comparison  # noqa: E402


def write_split(tmp_path, seed=0):
    root = tmp_path / "splits"
    directory = root / f"split_{seed}"
    directory.mkdir(parents=True)
    rng = np.random.RandomState(seed)
    indices, sizes, offset = {}, {}, 0
    for role, size in zip(comparison.ROLES, (120, 40, 40, 40), strict=True):
        labels = np.tile([0, 1], size // 2)
        features = rng.normal(size=(size, 3)).astype("float32")
        features[:, 0] += labels * 1.5
        ids = np.arange(offset, offset + size)
        offset += size
        np.savez(directory / f"{role}.npz", X=features, y=labels, indices=ids)
        indices[role], sizes[role] = ids.tolist(), size
    metadata = {
        "dataset": "spambase",
        "seed": seed,
        "class_prior": {"population": 0.5},
        "indices": indices,
        "role_sizes": sizes,
        "preprocessing": {"kind": "synthetic"},
    }
    (directory / "split_manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def plan_for(root, **kwargs):
    return comparison.build_plan(
        root, dataset="spambase", seeds=[0], c_tokens=["0.5"], smoke=True, **kwargs
    )
