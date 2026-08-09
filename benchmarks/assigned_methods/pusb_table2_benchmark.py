"""Plan and run claim-safe PUSB Table 2 benchmarks on locked datasets."""

# Benchmark matrices follow sklearn's conventional X/y names.
# ruff: noqa: N803, N806

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.assigned_methods.pusb_official_data import run_trials
from benchmarks.assigned_methods.pusb_table2_data import (
    DEFAULT_MANIFEST,
    build_sampling_trial_plan,
    load_manifest,
    load_table2_dataset,
)

from .._common import canonical_hash

STRICT_POLICY = "strict_complete_cells"
COMPATIBILITY_POLICY = "released_compatibility"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a Table 2 benchmark configuration."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    if config.get("protocol") != "pusb_table2_benchmark":
        raise ValueError("config protocol must be 'pusb_table2_benchmark'")
    if config.get("paper_claim") is not False:
        raise ValueError("Table 2 runs must explicitly set paper_claim=false")
    policy = config.get("sampling_policy")
    if policy not in {STRICT_POLICY, COMPATIBILITY_POLICY}:
        raise ValueError(f"sampling_policy must be '{STRICT_POLICY}' or '{COMPATIBILITY_POLICY}'")
    expected_fidelity = {
        STRICT_POLICY: "paper_protocol_strict_feasible_subset",
        COMPATIBILITY_POLICY: "official_released_compatibility",
    }[policy]
    if config.get("fidelity_level") != expected_fidelity:
        raise ValueError(f"fidelity_level must be '{expected_fidelity}' for {policy}")
    if not config.get("datasets"):
        raise ValueError("datasets must be non-empty")
    experiment = config.get("experiment", {})
    for field in ("class_priors", "unlabeled_sizes"):
        if not experiment.get(field):
            raise ValueError(f"experiment.{field} must be non-empty")
    if int(experiment.get("repetitions", 0)) <= 0:
        raise ValueError("experiment.repetitions must be positive")
    return config


def _plan_parameters(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config["experiment"]
    return {
        "initial_seed": int(experiment["initial_seed"]),
        "repetitions": int(experiment["repetitions"]),
        "class_priors": tuple(float(value) for value in experiment["class_priors"]),
        "unlabeled_sizes": tuple(int(value) for value in experiment["unlabeled_sizes"]),
        "positive_size": int(experiment["positive_size"]),
        "test_size": int(experiment["test_size"]),
        "holdout_size": int(experiment["holdout_size"]),
    }


def build_dataset_plan(dataset: str, y, config: dict[str, Any]) -> pd.DataFrame:
    """Build one dataset's exact trial plan and apply the configured claim policy."""
    rows = build_sampling_trial_plan(y, **_plan_parameters(config))
    plan = pd.DataFrame(rows)
    plan.insert(0, "dataset", dataset)
    cell_columns = ["unlabeled_size", "class_prior"]
    plan["cell_all_repetitions_feasible"] = plan.groupby(cell_columns)[
        "strictly_feasible"
    ].transform("all")
    if config["sampling_policy"] == STRICT_POLICY:
        plan["selected_for_execution"] = plan["cell_all_repetitions_feasible"]
    else:
        plan["selected_for_execution"] = True
    plan["sampling_policy"] = config["sampling_policy"]
    plan["fidelity_level"] = config["fidelity_level"]
    plan["failure_reasons"] = plan["failure_reasons"].map(lambda values: "|".join(values))
    return plan


def build_benchmark_plan(
    config: dict[str, Any],
    *,
    data_root: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Verify locked data and return the complete multi-dataset execution plan."""
    manifest = load_manifest(manifest_path)
    unknown = sorted(set(config["datasets"]) - set(manifest["datasets"]))
    if unknown:
        raise ValueError(f"datasets are absent from the locked manifest: {unknown}")
    plans = []
    provenance = {}
    for dataset in config["datasets"]:
        _, y, dataset_provenance = load_table2_dataset(
            dataset, data_root, manifest_path=manifest_path
        )
        plans.append(build_dataset_plan(dataset, y, config))
        provenance[dataset] = dataset_provenance
    return pd.concat(plans, ignore_index=True), provenance


def summarize_plan(plan: pd.DataFrame) -> dict[str, Any]:
    """Return cell and trial counts used in plan and run manifests."""
    cells = plan.drop_duplicates(["dataset", "unlabeled_size", "class_prior"])
    return {
        "datasets": int(plan["dataset"].nunique()),
        "total_cells": len(cells),
        "fully_feasible_cells": int(cells["cell_all_repetitions_feasible"].sum()),
        "selected_cells": int(cells["selected_for_execution"].sum()),
        "excluded_cells": int((~cells["selected_for_execution"]).sum()),
        "total_trials": len(plan),
        "selected_trials": int(plan["selected_for_execution"].sum()),
        "undersized_selected_trials": int(
            (plan["selected_for_execution"] & ~plan["strictly_feasible"]).sum()
        ),
    }


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _git_value(project_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _completed_keys(trials: pd.DataFrame) -> set[tuple[str, int, float, int]]:
    required = ["dataset", "seed", "class_prior", "unlabeled_size"]
    if not set(required).issubset(trials.columns):
        raise ValueError("existing trials.csv is missing Table 2 resume key columns")
    if trials.duplicated(required).any():
        raise ValueError("existing trials.csv contains duplicate Table 2 trial keys")
    return {
        (str(row.dataset), int(row.seed), float(row.class_prior), int(row.unlabeled_size))
        for row in trials.itertuples(index=False)
    }


def _manifest_payload(
    config: dict[str, Any],
    plan: pd.DataFrame,
    provenance: dict[str, dict[str, Any]],
    *,
    status: str,
    n_completed_trials: int,
    execution_scope: dict[str, int],
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    git_status = _git_value(project_root, "status", "--porcelain")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config["protocol"],
        "status": status,
        "sampling_policy": config["sampling_policy"],
        "fidelity_level": config["fidelity_level"],
        "paper_claim": False,
        "config_sha256": canonical_hash(config),
        "plan_summary": summarize_plan(plan),
        "execution_scope": execution_scope,
        "shard_selected_trials": int(plan["selected_for_shard"].sum()),
        "n_completed_trials": n_completed_trials,
        "datasets": provenance,
        "git_commit": _git_value(project_root, "rev-parse", "HEAD"),
        "git_worktree_dirty": bool(git_status) if git_status is not None else None,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
            "scipy": _package_version("scipy"),
            "densratio": _package_version("densratio"),
        },
        "limitations": config.get("limitations", []),
    }


def run_benchmark(
    config: dict[str, Any],
    *,
    data_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    resume: bool = False,
    plan_only: bool = False,
    shard_count: int = 1,
    shard_index: int = 0,
) -> pd.DataFrame:
    """Write the exact plan and optionally execute it with per-trial checkpoints."""
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output / "resolved_config.json"
    execution_scope_path = output / "execution_scope.json"
    trials_path = output / "trials.csv"
    execution_scope = {"shard_count": shard_count, "shard_index": shard_index}
    if resume:
        if not resolved_config_path.is_file():
            raise ValueError("cannot resume without resolved_config.json")
        existing_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise ValueError("resume config differs from the existing resolved_config.json")
        if not execution_scope_path.is_file():
            raise ValueError("cannot resume without execution_scope.json")
        existing_scope = json.loads(execution_scope_path.read_text(encoding="utf-8"))
        if existing_scope != execution_scope:
            raise ValueError("resume shard scope differs from the existing execution_scope.json")
        existing_trials = pd.read_csv(trials_path) if trials_path.is_file() else pd.DataFrame()
    else:
        existing_trials = pd.DataFrame()
        if not plan_only:
            for stale_name in ("trials.csv", "summary.csv", "run_manifest.json"):
                stale_path = output / stale_name
                if stale_path.exists():
                    stale_path.unlink()
        resolved_config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        execution_scope_path.write_text(
            json.dumps(execution_scope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    plan, provenance = build_benchmark_plan(
        config, data_root=data_root, manifest_path=manifest_path
    )
    _write_csv_atomic(plan, output / "trial_plan.csv")
    plan["selected_for_shard"] = False
    selected_indices = plan.index[plan["selected_for_execution"]]
    selected_ordinals = pd.Series(range(len(selected_indices)), index=selected_indices)
    shard_indices = selected_ordinals.index[selected_ordinals % shard_count == shard_index]
    plan.loc[shard_indices, "selected_for_shard"] = True
    _write_csv_atomic(plan, output / "trial_plan.csv")
    excluded_cells = plan.loc[
        ~plan["selected_for_execution"],
        [
            "dataset",
            "unlabeled_size",
            "class_prior",
            "cell_all_repetitions_feasible",
        ],
    ].drop_duplicates()
    _write_csv_atomic(excluded_cells, output / "excluded_cells.csv")

    if plan_only:
        manifest = _manifest_payload(
            config,
            plan,
            provenance,
            status="planned_not_executed",
            n_completed_trials=len(existing_trials),
            execution_scope=execution_scope,
        )
        (output / "plan_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return existing_trials

    completed = _completed_keys(existing_trials) if not existing_trials.empty else set()
    selected_plan = plan[plan["selected_for_shard"]]
    for dataset in config["datasets"]:
        X, y, _ = load_table2_dataset(dataset, data_root, manifest_path=manifest_path)
        dataset_plan = selected_plan[selected_plan["dataset"] == dataset]
        for spec in dataset_plan.itertuples(index=False):
            key = (dataset, int(spec.seed), float(spec.class_prior), int(spec.unlabeled_size))
            if key in completed:
                continue
            trial_config = copy.deepcopy(config)
            trial_config["protocol"] = "pusb_table2_benchmark"
            trial_config["experiment"]["seeds"] = [int(spec.seed)]
            trial_config["experiment"]["class_priors"] = [float(spec.class_prior)]
            trial_config["experiment"]["unlabeled_sizes"] = [int(spec.unlabeled_size)]
            trial_config["experiment"]["allow_undersized"] = (
                config["sampling_policy"] == COMPATIBILITY_POLICY
            )
            try:
                row = run_trials(trial_config, X, y).iloc[0].to_dict()
            except Exception as error:
                failure = {
                    "schema_version": 1,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset": dataset,
                    "seed": int(spec.seed),
                    "class_prior": float(spec.class_prior),
                    "unlabeled_size": int(spec.unlabeled_size),
                    "shard_count": shard_count,
                    "shard_index": shard_index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                (output / "last_failure.json").write_text(
                    json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                raise
            row.update(
                {
                    "dataset": dataset,
                    "repetition": int(spec.repetition),
                    "sampling_policy": config["sampling_policy"],
                    "strictly_feasible_split": bool(spec.strictly_feasible),
                    "cell_all_repetitions_feasible": bool(spec.cell_all_repetitions_feasible),
                    "split_failure_reasons": spec.failure_reasons,
                }
            )
            existing_trials = pd.concat(
                [existing_trials, pd.DataFrame([row])], ignore_index=True, sort=False
            )
            _write_csv_atomic(existing_trials, trials_path)
            completed.add(key)
            (output / "last_failure.json").unlink(missing_ok=True)

    if existing_trials.empty:
        raise ValueError("benchmark plan selected no trials")
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
        if column in existing_trials
    ]
    summary = (
        existing_trials.groupby(["dataset", "class_prior", "unlabeled_size"])[metric_columns]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(output / "summary.csv", index=False)
    manifest = _manifest_payload(
        config,
        plan,
        provenance,
        status=(
            "completed"
            if len(existing_trials) == int(selected_plan.shape[0])
            else "partial_checkpoint"
        ),
        n_completed_trials=len(existing_trials),
        execution_scope=execution_scope,
    )
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return existing_trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trials = run_benchmark(
        config,
        data_root=args.data_root,
        output_dir=args.output,
        manifest_path=args.manifest,
        resume=args.resume,
        plan_only=args.plan_only,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    action = "Planned" if args.plan_only else "Wrote"
    print(f"{action} PUSB Table 2 benchmark; completed trials: {len(trials)}")


if __name__ == "__main__":
    main()
