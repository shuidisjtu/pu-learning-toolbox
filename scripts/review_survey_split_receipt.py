"""Read-only audit of received P1.4 splits; prints a JSON evidence report."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart, validate_bundle
from pu_toolbox.experiment.split_archive import load_index, verify

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--splits", type=Path, default=Path("data/splits"))
parser.add_argument(
    "--index", type=Path, default=Path("docs/research/pu_survey/data/split_artifacts_index.json")
)
parser.add_argument("--archive-dir", type=Path, required=True)
args = parser.parse_args()
if not __debug__:
    parser.error("Assertions are required; do not run this audit with python -O.")
ROLES = ("train", "pu_val", "clean_val", "test")
index = load_index(args.index)
report = {"integrity": verify(args.splits, index, archive_dir=args.archive_dir), "splits": []}
if not report["integrity"]["ok"]:
    raise SystemExit("Receipt integrity check failed: " + json.dumps(report["integrity"]))

for dataset in ("spambase", "imdb", "cifar10"):
    test_hashes = []
    for seed in range(5):
        base = args.splits / dataset / f"split_{seed}"
        m = json.loads((base / "split_manifest.json").read_text())
        parts = {}
        entry = {"dataset": dataset, "seed": seed, "roles": {}, "issues": []}
        for role in ROLES:
            with np.load(base / f"{role}.npz", allow_pickle=False) as z:
                X, y, ids = z["X"], z["y"], z["indices"]
            part = DatasetPart(
                X=X, labels=y, indices=ids, view="clean", for_selection=role != "test"
            )
            parts[role] = part
            assert len(X) == len(y) == len(ids) == m["role_sizes"][role]
            assert ids.tolist() == m["indices"][role]
            assert np.isfinite(X).all()
            assert np.isclose(y.mean(), m["role_positive_rates"][role], rtol=0, atol=1e-14)
            entry["roles"][role] = {
                "shape": list(X.shape),
                "dtype": str(X.dtype),
                "positive_rate": float(y.mean()),
                "finite": True,
            }
            if dataset == "imdb":
                assert X.shape[1] == 384
                norm = np.linalg.norm(X, axis=1)
                assert np.allclose(norm, 1, atol=3e-7, rtol=0)
                entry["roles"][role]["norm_max_error"] = float(np.max(np.abs(norm - 1)))
        bundle = DatasetBundle(**parts)
        validate_bundle(bundle)
        all_y = np.concatenate([parts[r].labels for r in ROLES])
        assert np.isclose(all_y.mean(), m["class_prior"]["population"], atol=1e-14, rtol=0)
        assert np.isclose(
            parts["train"].labels.mean(), m["class_prior"]["train"], atol=1e-14, rtol=0
        )
        assert m["provenance"]["status"] == "recorded"
        entry["bundle_contract"] = "passed"
        entry["population_prior"] = float(all_y.mean())
        entry["manifest_sha256"] = hashlib.sha256(
            (base / "split_manifest.json").read_bytes()
        ).hexdigest()
        if dataset == "spambase":
            X = parts["train"].X
            entry["train_mean_max_abs"] = float(np.max(np.abs(X.mean(axis=0, dtype=np.float64))))
            entry["train_std_max_error"] = float(
                np.max(np.abs(X.std(axis=0, dtype=np.float64) - 1))
            )
            assert entry["train_mean_max_abs"] < 1e-6
            assert entry["train_std_max_error"] < 1e-6
        if dataset == "cifar10":
            X = parts["train"].X
            sums = np.zeros(3)
            squares = np.zeros(3)
            n = 0
            for offset in range(0, len(X), 256):
                batch = X[offset : offset + 256].astype(np.float32) / 255.0
                sums += np.sum(batch, axis=(0, 2, 3), dtype=np.float64)
                squares += np.sum(batch.astype(np.float64) ** 2, axis=(0, 2, 3))
                n += len(batch) * 32 * 32
            mean = sums / n
            std = np.sqrt(squares / n - mean**2)
            ref = m["preprocessing"]["normalization"]
            entry["channel_mean"] = mean.tolist()
            entry["channel_std"] = std.tolist()
            entry["channel_mean_max_error"] = float(np.max(np.abs(mean - ref["mean"])))
            entry["channel_std_max_error"] = float(np.max(np.abs(std - ref["std"])))
            # Cross-environment float32 scaling is bounded at its actual precision;
            # file-level checks above are byte-exact and are a separate question.
            entry["channel_statistics_atol"] = float(np.finfo(np.float32).eps)
            assert np.allclose(mean, ref["mean"], atol=np.finfo(np.float32).eps, rtol=0)
            assert np.allclose(std, ref["std"], atol=np.finfo(np.float32).eps, rtol=0)
        test_hashes.append(
            index["datasets"][dataset][f"split_{seed}"]["files"]["test.npz"]["sha256"]
        )
        report["splits"].append(entry)
    report.setdefault("test_file_hash_count", {})[dataset] = len(set(test_hashes))

print(json.dumps(report, indent=2))
