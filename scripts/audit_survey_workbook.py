"""Read-only P2.1/P2.2 Excel reconciliation; not a manifest or checkpoint audit.

Requires the optional review dependency openpyxl. The original workbook is
never modified. JSON is printed to stdout, including every reconstructed row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from pu_toolbox.experiment.pilot_plan import planned_runs
from pu_toolbox.experiment.survey_protocol import digest, load_protocol, resolve_protocol_path

KEY_FIELDS = (
    "dataset",
    "training_path",
    "comparability_group",
    "run_view",
    "mechanism",
    "method",
    "selection_protocol",
    "c",
)
RUN_FIELDS = (
    "dataset",
    "method",
    "training_path",
    "mechanism",
    "c",
    "seed",
)


def token(value):
    return "c_independent" if value is None else str(value)


def codes(value):
    return sorted(str(value).split(",")) if value else []


def batch_for(dataset, method, path):
    if dataset == "spambase" and path == "native_2d":
        return "B1"
    if dataset == "imdb" and path == "native_2d":
        return "B2"
    if dataset == "cifar10" and path == "native_cnn" and method == "nnpu":
        return "B4"
    if dataset == "cifar10" and path == "cnn_feature_adapter":
        return "B3b" if method in {"dist_pu", "self_pu"} else "B3a"
    return "unknown"


def audit_workbook(path, index_path, protocol_path="survey-v1.2"):
    import openpyxl

    path = Path(path)
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    protocol = load_protocol(resolve_protocol_path(protocol_path))
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    problems = []

    def check(condition, message):
        if not condition:
            problems.append(message)

    def records(sheet):
        iterator = iter(workbook[sheet].values)
        header = next(iterator)
        check(len(header) == len(set(header)), f"{sheet}: duplicate headers")
        return [
            dict(zip(header, row, strict=False))
            for row in iterator
            if any(v is not None for v in row)
        ]

    check(
        digest(protocol) == index["identity"]["protocol_sha256"], "loaded protocol digest differs"
    )

    received_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    check(
        received_hash == index["release_table"]["sha256"],
        "workbook sha256 differs from release index",
    )
    check(
        path.stat().st_size == index["release_table"]["bytes"],
        "workbook size differs from release index",
    )
    runs, summary, coverage = records("runs"), records("P2.2 汇总"), records("coverage")
    readme = "\n".join(str(row[0]) for row in workbook["readme"].values if row[0])
    for name in ("protocol_sha256", "comparison_sha256"):
        check(index["identity"][name] in readme, f"readme missing {name}")
    check(index["produced_by_commit"] in readme, "readme producer commit differs")
    expected = {
        (
            r.dataset,
            r.method,
            r.training_path,
            "pn_oracle" if r.is_oracle else r.mechanism,
            token(r.c_token),
            r.seed,
        )
        for r in planned_runs(protocol)
    }
    observed = Counter(tuple(token(r[k]) if k == "c" else r[k] for k in RUN_FIELDS) for r in runs)
    check(all(n == 1 for n in observed.values()), "duplicate run keys")
    check(set(observed) == expected, "run keys differ from frozen protocol")
    check(len(runs) == index["inputs"]["total"]["manifests"], "run total differs from index")
    units = {
        (r["dataset"], r["method"], r["training_path"]): r for r in protocol["execution_units"]
    }
    projected = defaultdict(list)
    costs = defaultdict(list)
    for row_number, run in enumerate(runs, 2):
        prefix = f"runs row {row_number}"
        check(
            run["batch"] == batch_for(run["dataset"], run["method"], run["training_path"]),
            prefix + ": batch assignment",
        )
        unit = units.get((run["dataset"], run["method"], run["training_path"]))
        check(unit is not None and unit.get("runnable") is True, prefix + ": non-runnable unit")
        if unit:
            check(
                run["comparability_group"] == unit["comparability_group"],
                prefix + ": group differs from protocol",
            )
        oracle = run["method"] == "pn_oracle"
        expected_view = (
            "os-compatible" if run["method"] in {"lbe", "pn_oracle"} else "ts-compatible"
        )
        check(run["run_view"] == expected_view, prefix + ": training view")
        check(
            run["calibration_applied"] is (expected_view == "ts-compatible"),
            prefix + ": calibration flag",
        )
        check(run["execution_mode"] == "versioned_pilot", prefix + ": execution mode")
        variant = (
            "without_clean_validation_meta_reweighting"
            if run["method"] == "self_pu"
            else "shared_spec_engineering"
        )
        check(run["method_variant"] == variant, prefix + ": method variant")
        check(run["formal_eligible"] is (not oracle), prefix + ": eligibility flag")
        check(
            codes(run["protocol_deviation"]) == (["c_grid"] if oracle else []),
            prefix + ": deviations",
        )
        check(
            codes(run["formal_blockers"]) == (["protocol_deviation"] if oracle else []),
            prefix + ": blockers",
        )
        check(
            run["single_config_candidates"] == len(protocol["candidate_pool"]),
            prefix + ": candidate count",
        )
        for field in ("single_config_seconds", "tuning_seconds", "peak_gpu_memory_mb"):
            value = run[field]
            check(
                (field == "peak_gpu_memory_mb" and value is None)
                or (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                ),
                prefix + f": invalid resource {field}",
            )
        check(
            run["tuning_seconds"] >= run["single_config_seconds"] - 0.00005,
            prefix + ": tuning shorter than training",
        )
        base = {k: run[k] for k in KEY_FIELDS if k not in {"selection_protocol", "c"}}
        base["c"] = token(run["c"])
        cost_key = tuple(base[k] for k in KEY_FIELDS if k != "selection_protocol")
        costs[cost_key].append(run)
        for selection in ("PA", "OA"):
            required = selection == "OA" or (not oracle and run["mechanism"] == "scar")
            metric_prefix = selection.lower()
            acc, auc = run[metric_prefix + "_accuracy"], run[metric_prefix + "_auc"]
            check((acc is not None) == required, prefix + f": {selection} coverage")
            for metric, value in (("accuracy", acc), ("auc", auc)):
                check(
                    value is None
                    or (
                        isinstance(value, int | float) and math.isfinite(value) and 0 <= value <= 1
                    ),
                    prefix + f": invalid {selection} {metric}",
                )
            check(
                not required
                or auc is not None
                or bool(run[metric_prefix + "_auc_unavailable_reason"]),
                prefix + f": absent {selection} AUC lacks reason",
            )
            check(
                auc is None or not run[metric_prefix + "_auc_unavailable_reason"],
                prefix + f": available {selection} AUC has reason",
            )
            check(
                required or (auc is None and not run[metric_prefix + "_auc_unavailable_reason"]),
                prefix + f": unexpected {selection} AUC",
            )
            if required:
                key = tuple(selection if k == "selection_protocol" else base[k] for k in KEY_FIELDS)
                projected[key].append(run)
    summary_by_key = {}
    for row in summary:
        key = tuple(row[k] for k in KEY_FIELDS)
        check(key not in summary_by_key, "duplicate summary key: " + str(key))
        summary_by_key[key] = row
    check(
        set(summary_by_key) == set(projected), "summary coverage differs from raw PA/OA projection"
    )
    rebuilt = []
    max_errors = defaultdict(float)
    for key, observations in sorted(projected.items()):
        observed_row = summary_by_key.get(key)
        if not observed_row:
            continue
        row = dict(zip(KEY_FIELDS, key, strict=False))
        selection = row["selection_protocol"].lower()
        seeds = sorted(r["seed"] for r in observations)
        check(seeds == protocol["seeds"], "summary seed coverage: " + str(key))
        values = [r[selection + "_accuracy"] for r in observations]
        aucs = [r[selection + "_auc"] for r in observations if r[selection + "_auc"] is not None]
        cost_rows = costs[tuple(row[k] for k in KEY_FIELDS if k != "selection_protocol")]
        peaks = [r["peak_gpu_memory_mb"] for r in cost_rows if r["peak_gpu_memory_mb"] is not None]
        row.update(
            n_observed=len(values),
            mean_percent=statistics.mean(values) * 100,
            sample_std_percent=statistics.stdev(values) * 100 if len(values) >= 2 else None,
            auc_mean_percent=statistics.mean(aucs) * 100 if aucs else None,
            auc_n_observed=len(aucs),
            run_single_config_seconds_sum=math.fsum(r["single_config_seconds"] for r in cost_rows),
            run_tuning_seconds_sum=math.fsum(r["tuning_seconds"] for r in cost_rows),
            run_peak_gpu_memory_mb=max(peaks) if peaks else None,
        )
        expected_status = "partial" if row["method"] == "pn_oracle" else "formal"
        expected_reasons = (
            ["formal_blockers_present", "protocol_deviation"]
            if expected_status == "partial"
            else []
        )
        row.update(status=expected_status, reasons=expected_reasons, missing_seeds=[])
        check(observed_row["status"] == expected_status, "summary status: " + str(key))
        check(codes(observed_row["reasons"]) == expected_reasons, "summary reasons: " + str(key))
        check(not observed_row["missing_seeds"], "summary missing seeds: " + str(key))
        for field in (
            "n_observed",
            "mean_percent",
            "sample_std_percent",
            "auc_mean_percent",
            "auc_n_observed",
            "run_single_config_seconds_sum",
            "run_tuning_seconds_sum",
            "run_peak_gpu_memory_mb",
        ):
            actual, wanted = observed_row[field], row[field]
            if actual is None or wanted is None:
                check(actual is wanted, f"{key}: {field} null mismatch")
                continue
            delta = abs(actual - wanted)
            max_errors[field] = max(max_errors[field], delta)
            # Raw training seconds are rounded to 4 decimals; GPU MiB to 2.
            tolerance = (
                len(cost_rows) * 0.00005 + 1e-9
                if field == "run_single_config_seconds_sum"
                else 1e-9
            )
            if field == "run_peak_gpu_memory_mb":
                tolerance = 0.010000001
            check(delta <= tolerance, f"{key}: {field} delta={delta} exceeds {tolerance}")
        rebuilt.append(row)
    batch_reports = {}
    for batch in index["inputs"]["batches"]:
        batch_rows = [r for r in runs if r["batch"] == batch]
        groups = {(r["comparability_group"], r["run_view"], r["mechanism"]) for r in batch_rows}
        stats = dict(
            runs=len(batch_rows),
            methods=dict(Counter(r["method"] for r in batch_rows)),
            views=dict(Counter(r["run_view"] for r in batch_rows)),
            calibration=dict(Counter(str(r["calibration_applied"]) for r in batch_rows)),
            mechanisms=dict(Counter(r["mechanism"] for r in batch_rows)),
            group_count=len(groups),
            formal_eligible=dict(Counter(str(r["formal_eligible"]) for r in batch_rows)),
            protocol_deviation_count=sum(bool(r["protocol_deviation"]) for r in batch_rows),
            peak_gpu_memory_unmeasured=sum(r["peak_gpu_memory_mb"] is None for r in batch_rows),
            no_recorded_blockers=all(not r["formal_blockers"] for r in batch_rows),
        )
        check(
            stats["runs"] == index["inputs"]["batches"][batch]["manifests"],
            f"{batch}: indexed coverage",
        )
        coverage_rows = [r for r in coverage if r["batch"] == batch]
        check(len(coverage_rows) == 1, f"{batch}: coverage sheet unique row")
        if coverage_rows:
            c = coverage_rows[0]
            for field in ("manifests", "peak_gpu_memory_unmeasured"):
                check(
                    c[field] == stats["runs" if field == "manifests" else field],
                    f"{batch}: coverage {field}",
                )
            for field, stat in (
                ("methods", "methods"),
                ("view_distribution", "views"),
                ("calibration_distribution", "calibration"),
                ("mechanism_distribution", "mechanisms"),
            ):
                parsed = dict(item.rsplit(" = ", 1) for item in c[field].split(", "))
                check(
                    {k: int(v) for k, v in parsed.items()} == stats[stat],
                    f"{batch}: coverage {field}",
                )
            expected_deviation = (
                f"c_grid = {stats['protocol_deviation_count']}"
                if stats["protocol_deviation_count"]
                else "none"
            )
            check(c["protocol_deviation"] == expected_deviation, f"{batch}: coverage deviations")
        batch_reports[batch] = stats
    workbook.close()
    return {
        "scope": (
            "workbook identity, frozen-plan coverage and independent raw-to-summary arithmetic; "
            "not source manifest audit"
        ),
        "input": {"filename": path.name, "bytes": path.stat().st_size, "sha256": received_hash},
        "ok": not problems,
        "problems": problems,
        "runs": len(runs),
        "summary_rows": len(summary),
        "run_cost_groups": len(costs),
        "selection_rows": dict(Counter(r["selection_protocol"] for r in summary)),
        "summary_states": dict(Counter(r["status"] for r in summary)),
        "batches": batch_reports,
        "max_absolute_differences": dict(max_errors),
        "single_config_seconds_total_no_double_count": math.fsum(
            r["single_config_seconds"] for r in runs
        ),
        "tuning_seconds_total_no_double_count": math.fsum(r["tuning_seconds"] for r in runs),
        "gpu_unmeasured": sum(r["peak_gpu_memory_mb"] is None for r in runs),
        "gpu_unmeasured_by_method": dict(
            Counter(r["method"] for r in runs if r["peak_gpu_memory_mb"] is None)
        ),
        "metric_observations": {
            k: sum(r[k] is not None for r in runs)
            for k in ("pa_accuracy", "oa_accuracy", "pa_auc", "oa_auc")
        },
        "variants": dict(Counter(r["method_variant"] for r in runs)),
        "environment_combinations": [
            {"identity": list(key), "runs": count}
            for key, count in Counter(
                (r["python_version"], r["torch_version"], r["cuda_runtime_version"], r["gpu_name"])
                for r in runs
            ).items()
        ],
        "performance_diagnostics": {
            "interpretation": "descriptive flags, not preregistered rejection or method ranking",
            "summary_rows_auc_below_50": [
                {k: row[k] for k in (*KEY_FIELDS, "mean_percent", "auc_mean_percent")}
                for row in rebuilt
                if row["auc_mean_percent"] is not None and row["auc_mean_percent"] < 50
            ],
        },
        "readme_correction_requests": [
            entry
            for entry in [
                {"field": "c", "correction": "label frequency P(S=1|Y=1), not class prior pi"},
                {
                    "field": "run_view",
                    "correction": "OS/TS describe sampling/risk inputs, not prior-free vs transfer",
                },
                {
                    "field": "PA",
                    "correction": "proxy accuracy on PU validation, not merely prior-aware",
                },
            ]
            if (entry["field"] == "c" and "class-prior token" in readme)
            or (entry["field"] == "run_view" and "priors/oracle-free" in readme)
            or (entry["field"] == "PA" and "prior-aware" in readme)
        ],
        "source_manifests_independently_verified": False,
        "reconstructed_summary": rebuilt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument(
        "--index", type=Path, default=Path("docs/research/pu_survey/data/p2_2_artifacts_index.json")
    )
    parser.add_argument("--protocol", default="survey-v1.2")
    args = parser.parse_args()
    try:
        report = audit_workbook(args.workbook, args.index, args.protocol)
    except (ImportError, KeyError, ValueError, OSError) as exc:
        parser.exit(1, f"error: workbook audit failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
