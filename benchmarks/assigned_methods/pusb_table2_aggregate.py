"""Validate and aggregate completed PUSB Table 2 benchmark shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

KEY_COLUMNS = ["dataset", "seed", "class_prior", "unlabeled_size"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key_set(frame: pd.DataFrame) -> set[tuple[str, int, float, int]]:
    if not set(KEY_COLUMNS).issubset(frame.columns):
        raise ValueError(f"trial data is missing key columns: {KEY_COLUMNS}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate trial keys found")
    return {
        (str(row.dataset), int(row.seed), float(row.class_prior), int(row.unlabeled_size))
        for row in frame.itertuples(index=False)
    }


def _summary(trials: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in (
            "quantile_accuracy",
            "quantile_balanced_accuracy",
            "roc_auc",
            "density_ratio_accuracy",
            "density_ratio_roc_auc",
            "elapsed_seconds",
        )
        if column in trials
    ]
    summary = (
        trials.groupby(["dataset", "class_prior", "unlabeled_size"])[metric_columns]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def aggregate_shards(
    shard_root: str | Path,
    *,
    plan_path: str | Path,
    output_dir: str | Path,
    shard_count: int,
) -> pd.DataFrame:
    """Require complete shard coverage, exact trial keys, and consistent provenance."""
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    shard_root = Path(shard_root)
    output = Path(output_dir)
    plan_path = Path(plan_path)
    plan = pd.read_csv(plan_path)
    if "selected_for_execution" not in plan:
        raise ValueError("plan is missing selected_for_execution")
    expected = plan[plan["selected_for_execution"].astype(bool)]
    expected_keys = _key_set(expected)

    frames = []
    shard_records = []
    config_hashes = set()
    fidelities = set()
    for shard_index in range(shard_count):
        shard = shard_root / f"shard-{shard_index:02d}"
        manifest_path = shard / "run_manifest.json"
        trials_path = shard / "trials.csv"
        if not manifest_path.is_file() or not trials_path.is_file():
            raise ValueError(f"shard {shard_index} is missing completed artifacts")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_scope = {"shard_count": shard_count, "shard_index": shard_index}
        if manifest.get("status") != "completed":
            raise ValueError(f"shard {shard_index} is not completed")
        if manifest.get("execution_scope") != expected_scope:
            raise ValueError(f"shard {shard_index} execution scope mismatch")
        if manifest.get("paper_claim") is not False:
            raise ValueError(f"shard {shard_index} has unsafe paper_claim metadata")
        frame = pd.read_csv(trials_path)
        if len(frame) != manifest.get("n_completed_trials"):
            raise ValueError(f"shard {shard_index} manifest trial count mismatch")
        frames.append(frame)
        config_hashes.add(manifest.get("config_sha256"))
        fidelities.add(manifest.get("fidelity_level"))
        shard_records.append(
            {
                "shard_index": shard_index,
                "n_trials": len(frame),
                "trials_sha256": _sha256(trials_path),
                "manifest_sha256": _sha256(manifest_path),
            }
        )
    if len(config_hashes) != 1 or None in config_hashes:
        raise ValueError("shards do not share one valid config hash")
    if len(fidelities) != 1 or None in fidelities:
        raise ValueError("shards do not share one valid fidelity level")

    trials = pd.concat(frames, ignore_index=True, sort=False)
    actual_keys = _key_set(trials)
    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    if missing or unexpected:
        raise ValueError(
            f"aggregated trial keys differ from plan: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    trials = trials.sort_values(KEY_COLUMNS).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output / "trials.csv", index=False)
    _summary(trials).to_csv(output / "summary.csv", index=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "pusb_table2_shard_aggregation",
        "status": "completed",
        "paper_claim": False,
        "fidelity_level": fidelities.pop(),
        "config_sha256": config_hashes.pop(),
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "shard_count": shard_count,
        "n_trials": len(trials),
        "n_expected_trials": len(expected),
        "shards": shard_records,
    }
    (output / "aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    trials = aggregate_shards(
        args.shard_root,
        plan_path=args.plan,
        output_dir=args.output,
        shard_count=args.shard_count,
    )
    print(f"Aggregated {len(trials)} verified PUSB Table 2 trials into {args.output}")


if __name__ == "__main__":
    main()
