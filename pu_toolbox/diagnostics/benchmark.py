"""Audit persisted benchmark artifacts before results are reported or shared."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pu_toolbox.utils.serialization import canonical_hash, json_safe

__all__ = ["BenchmarkAuditReport", "audit_benchmark_results"]

_REQUIRED_FILES = ("resolved_config.json", "run_manifest.json", "trials.csv", "summary.csv")
# Union of metric columns emitted by the synthetic (assigned_methods/runner.py)
# and deep-PU (deep_pu/runner.py) benchmark runners. Only the intersection with
# the audited trials.csv columns is checked, so entries outside the current
# runner family are inert until a result dir from the other family is audited.
# Keep in sync when a runner emits a new metric column.
_METRIC_COLUMNS = {
    "accuracy",
    "average_precision",
    "balanced_accuracy",
    "bayes_posterior_spearman",
    "brier",
    "class_prior_absolute_error",
    "density_ratio_accuracy",
    "density_ratio_balanced_accuracy",
    "density_ratio_roc_auc",
    "f1",
    "pairwise_ranking_accuracy",
    "posterior_kendall",
    "posterior_spearman",
    "prior_abs_error",
    "quantile_accuracy",
    "quantile_balanced_accuracy",
    "roc_auc",
    "score_brier",
    "zero_threshold_accuracy",
}


@dataclass(frozen=True)
class BenchmarkAuditReport:
    """Machine-readable audit result for one benchmark output directory."""

    result_dir: str
    protocol: str | None
    paper_claim: bool | None
    passed: bool
    n_trials: int
    n_methods: int
    n_seeds: int
    checks: dict[str, bool]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a strict-JSON-compatible representation."""
        return json_safe(
            {
                "schema_version": self.schema_version,
                "result_dir": self.result_dir,
                "protocol": self.protocol,
                "paper_claim": self.paper_claim,
                "passed": self.passed,
                "n_trials": self.n_trials,
                "n_methods": self.n_methods,
                "n_seeds": self.n_seeds,
                "checks": self.checks,
                "errors": list(self.errors),
                "warnings": list(self.warnings),
            }
        )

    def to_json(self) -> str:
        """Render deterministic strict JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    def save(self, path: str | Path) -> None:
        """Write the report as JSON, creating parent directories as needed."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")


def _load_json(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {label}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a JSON object")
        return None
    return value


def _load_trials(path: Path, errors: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"cannot read trials.csv: {exc}")
        return [], []
    if not columns:
        errors.append("trials.csv has no header")
    if not rows:
        errors.append("trials.csv has no trial rows")
    return rows, columns


def _check_summary(path: Path, errors: list[str]) -> bool:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = reader.fieldnames or []
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"cannot read summary.csv: {exc}")
        return False
    if not columns or not rows:
        errors.append("summary.csv must contain a header and at least one data row")
        return False
    return True


def _check_finite_metrics(
    rows: list[dict[str, str]], columns: list[str], errors: list[str]
) -> bool:
    valid = True
    for column in sorted(_METRIC_COLUMNS.intersection(columns)):
        for line, row in enumerate(rows, start=2):
            raw = row.get(column, "").strip()
            if not raw:
                continue
            try:
                finite = np.isfinite(float(raw))
            except ValueError:
                finite = False
            if not finite:
                errors.append(f"trials.csv:{line} metric {column!r} is not finite")
                valid = False
    return valid


def _check_splits(manifest: dict[str, Any], errors: list[str]) -> bool:
    """Validate declared PU splits when the manifest carries ``dataset_splits``.

    Manifests without the key (e.g. synthetic clean-room runners that record
    no per-seed splits) are not subject to this check; a present-but-malformed
    value is an integrity error.
    """
    splits = manifest.get("dataset_splits")
    if splits is None:
        return True
    if not isinstance(splits, dict):
        errors.append("dataset_splits must be an object")
        return False
    valid = True
    for seed, split in splits.items():
        if not isinstance(split, dict):
            errors.append(f"dataset_splits[{seed!r}] must be an object")
            valid = False
            continue
        for field in ("labeled_unlabeled_overlap", "train_validation_overlap"):
            if split.get(field, 0) != 0:
                errors.append(f"dataset_splits[{seed!r}].{field} must be zero")
                valid = False
        target = split.get("target_unlabeled_class_prior")
        actual = split.get("hidden_unlabeled_positive_rate")
        if target is not None and actual is not None and not np.isclose(target, actual, atol=1e-12):
            errors.append(
                f"dataset_splits[{seed!r}] hidden prior {actual} does not match target {target}"
            )
            valid = False
    return valid


def audit_benchmark_results(result_dir: str | Path) -> BenchmarkAuditReport:
    """Validate common provenance, completeness, metric, and PU-split contracts."""
    root = Path(result_dir)
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    checks["result_directory"] = root.is_dir()
    if not checks["result_directory"]:
        errors.append(f"result directory does not exist: {root}")

    missing = [name for name in _REQUIRED_FILES if not (root / name).is_file()]
    checks["required_artifacts"] = not missing
    if missing:
        errors.append(f"missing required artifacts: {', '.join(missing)}")

    config_path = root / "resolved_config.json"
    manifest_path = root / "run_manifest.json"
    trials_path = root / "trials.csv"
    summary_path = root / "summary.csv"
    config = (
        _load_json(config_path, "resolved_config.json", errors) if config_path.is_file() else None
    )
    manifest = (
        _load_json(manifest_path, "run_manifest.json", errors) if manifest_path.is_file() else None
    )
    rows, columns = _load_trials(trials_path, errors) if trials_path.is_file() else ([], [])
    summary_valid = _check_summary(summary_path, errors) if summary_path.is_file() else False
    checks["parseable_artifacts"] = (
        config is not None and manifest is not None and bool(columns) and summary_valid
    )

    if config is not None and manifest is not None:
        checks["config_hash"] = manifest.get("config_sha256") == canonical_hash(config)
        if not checks["config_hash"]:
            errors.append("resolved_config.json does not match run_manifest.json config_sha256")
    else:
        checks["config_hash"] = False

    declared_trials = manifest.get("n_trials") if manifest else None
    checks["trial_count"] = isinstance(declared_trials, int) and declared_trials == len(rows)
    if rows and not checks["trial_count"]:
        errors.append(
            f"run_manifest.json declares {declared_trials!r} trials but trials.csv has {len(rows)}"
        )

    serialized_rows = [tuple(row.get(column, "") for column in columns) for row in rows]
    checks["no_duplicate_trials"] = len(serialized_rows) == len(set(serialized_rows))
    if rows and not checks["no_duplicate_trials"]:
        errors.append("trials.csv contains duplicate trial rows")

    checks["finite_metrics"] = _check_finite_metrics(rows, columns, errors)
    checks["pu_split_integrity"] = _check_splits(manifest or {}, errors)

    seeds = {row["seed"] for row in rows if row.get("seed", "").strip()}
    methods = {
        row.get("method", "").strip() or row.get("implementation_variant", "").strip()
        for row in rows
    }
    methods.discard("")
    if config is not None and isinstance(config.get("seeds"), list):
        expected_seeds = {str(seed) for seed in config["seeds"]}
        checks["seed_coverage"] = seeds == expected_seeds
        if not checks["seed_coverage"]:
            errors.append(
                "trial seeds "
                f"{sorted(seeds)} do not match configured seeds {sorted(expected_seeds)}"
            )
    else:
        checks["seed_coverage"] = True

    if manifest and manifest.get("git_worktree_dirty") is True:
        warnings.append("benchmark was executed from a dirty Git worktree")
    paper_claim = manifest.get("paper_claim") if manifest else None
    if paper_claim is False:
        warnings.append(
            "paper_claim=false: results must not be presented as reproduced paper results"
        )

    return BenchmarkAuditReport(
        result_dir=str(root),
        protocol=manifest.get("protocol") if manifest else None,
        paper_claim=paper_claim if isinstance(paper_claim, bool) else None,
        passed=not errors,
        n_trials=len(rows),
        n_methods=len(methods),
        n_seeds=len(seeds),
        checks=checks,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
