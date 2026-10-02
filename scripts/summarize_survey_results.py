#!/usr/bin/env python
r"""Summarize the P2.1 batch manifests into P2.2 report rows.

The row is the unit of delivery: one mean and one sample standard deviation per
``(comparability group, view, mechanism, method, selection protocol, c)``, over
the five protocol seeds.  Four things a naive version gets wrong, all pinned by
tests:

* **The fairness gate is called once per group, not once per tree.**  It raises on
  the first bad group and names the rule, not the file, so a single call would
  report one problem and hide the rest -- and would have no members to name.
  Bucketing uses the gate's own ``group_key``, so the two cannot group differently.
* **Cost belongs to the run, not to the PA/OA row.**  One manifest pays for one
  training pass and scores both protocols from it; a cost block copied onto both
  rows and summed would bill the experiment twice.  Rows therefore reference a
  run-scoped cost entry instead of carrying a copy.
* **A missing seed is reported, never filled,** and a seed the protocol did not
  ask for is refused outright rather than counted beside the designed five.
* **A refusal still produces numbers.**  A group the gate will not compare is
  ``partial``: the metric exists, and what is missing is the right to rank it.

Runs the batch whitelist from :mod:`audit_survey_batches` and writes only to
``--out-dir``.

Run:  uv run python scripts/summarize_survey_results.py --config <config.json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ``scripts/`` is not a package, so siblings are reached through this file's own
# directory.  Resolving from ``__file__`` keeps them in the same checkout
# whatever the working directory is.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import aggregate_survey_runs as gate  # noqa: E402
import audit_survey_batches as audit  # noqa: E402

from pu_toolbox.experiment.survey_audit import AuditError  # noqa: E402
from pu_toolbox.experiment.survey_protocol import digest, load_protocol  # noqa: E402
from pu_toolbox.experiment.survey_summary import (  # noqa: E402
    METRIC_UNIT,
    SCHEMA_VERSION,
    SummaryError,
    aggregate_metric,
    group_key,  # noqa: E402
    normalize_c_token,
    resolve_status,
    result_labeling_mechanism,
    row_key,
    status_for_completeness,
    status_for_gate_block,
    status_for_gate_error,
    summarize_costs,
)

#: Metrics a manifest actually carries.  F1 / Precision / Recall are logged by
#: the protocol but are not in the manifest, so this tool does not claim them.
CARRIED_METRICS = ("accuracy", "auc")

#: The metric the pre-registered comparison is decided on (protocol §1.2).
PRIMARY_METRIC = "accuracy"


def cost_id(manifest: dict[str, Any]) -> str:
    """A stable, JSON-safe identifier for one run's cost entry.

    Run-scoped rather than row-scoped: it names the group, view, mechanism,
    method and ``c`` but no selection protocol, because the training cost does
    not depend on which protocol later read the weights.
    """
    group, view, mechanism = group_key(manifest)
    return "|".join(
        (group, view, mechanism, manifest["execution_unit"]["method"], normalize_c_token(manifest))
    )


def result_identity(manifest: dict[str, Any], *, selection_protocol: str, metric: str) -> dict:
    """The seven selector fields the pre-registered matrix keys on."""
    return {
        "method": manifest["execution_unit"]["method"],
        "dataset": manifest["execution_unit"]["dataset"],
        "labeling_mechanism": result_labeling_mechanism(manifest),
        "c_token": normalize_c_token(manifest),
        "selection_protocol": selection_protocol,
        "metric": metric,
        "training_path": manifest["training_path"],
    }


def collect_observations(
    entries: list[dict[str, Any]], *, metric: str
) -> tuple[dict[tuple, list], dict[str, list], list[dict[str, Any]]]:
    """Gather ``(seed, value)`` per row and ``(seed, resources)`` per run.

    Returns ``(rows, runs, refusals)``.  A manifest the gate refused contributes
    no observation: it produced no result to average, and treating its absence as
    a zero would move every mean it touches.
    """
    rows: dict[tuple, list] = {}
    runs: dict[str, list] = {}
    refusals: list[dict[str, Any]] = []
    for entry in entries:
        payload, path = entry["payload"], entry["path"]
        if entry.get("refused_reason"):
            refusals.append({"path": str(path), "reason": entry["refused_reason"]})
            continue
        if entry.get("status") == "not_reproducible":
            continue
        runs.setdefault(cost_id(payload), []).append(
            (payload["seed"], payload.get("resources") or {})
        )
        for protocol in ("PA", "OA"):
            block = (payload.get("test_results") or {}).get(protocol)
            if not isinstance(block, dict) or block.get(metric) is None:
                continue
            key = row_key(payload, selection_protocol=protocol)
            rows.setdefault(key, []).append((entry, float(block[metric])))
    return rows, runs, refusals


def build_rows(
    rows: dict[tuple, list],
    *,
    expected_seeds: list[int],
    gate_reasons: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """One report row per key: metric block, status, reasons and members."""
    built: list[dict[str, Any]] = []
    for key in sorted(rows, key=str):
        members = rows[key]
        group, view, mechanism, method, protocol, c_token = key
        manifest = members[0][0]["payload"]
        metric_block = aggregate_metric(
            ((entry["payload"]["seed"], value) for entry, value in members),
            metric_name=PRIMARY_METRIC,
            expected_seeds=expected_seeds,
        )
        candidates: set[str] = set()
        reasons: set[str] = set()

        blockers = sorted(
            {
                blocker
                for entry, _ in members
                for blocker in (entry["payload"].get("formal_blockers") or [])
            }
        )
        if blockers:
            status, codes = status_for_gate_block(blockers)
            candidates.add(status)
            reasons.update(codes)

        status, codes = status_for_completeness(
            missing_seeds=metric_block["missing_seeds"],
            total_seeds=len(expected_seeds),
        )
        candidates.add(status)
        reasons.update(codes)

        # A group the gate refused to compare still has numbers; what it lacks is
        # the right to rank them.
        refused_reasons = {
            code for entry, _ in members for code in gate_reasons.get(str(entry["path"]), [])
        }
        if refused_reasons:
            candidates.add("partial")
            reasons.update(refused_reasons)

        status, codes = resolve_status(candidates, sorted(reasons))
        built.append(
            {
                "result_identity": result_identity(
                    manifest, selection_protocol=protocol, metric=PRIMARY_METRIC
                ),
                "row_key": {
                    "comparability_group": group,
                    "run_view": view,
                    "mechanism": mechanism,
                    "method": method,
                    "selection_protocol": protocol,
                    "c_token": c_token,
                },
                "cost_id": cost_id(manifest),
                "metric": metric_block,
                "status": status,
                "reasons": codes,
                "members": sorted(str(entry["path"]) for entry, _ in members),
            }
        )
    return built


def build_run_costs(runs: dict[str, list], *, expected_seeds: list[int]) -> list[dict[str, Any]]:
    """One cost entry per run, keyed by :func:`cost_id`."""
    return [
        {"cost_id": key, "costs": summarize_costs(runs[key], expected_seeds=expected_seeds)}
        for key in sorted(runs)
    ]


def empty_summary(*, protocol: dict[str, Any]) -> dict[str, Any]:
    """The report skeleton, so a failed run still writes a readable artifact.

    A summary that raised instead of writing would leave the reviewer with
    nothing to read, and one that returned a partial dict would render with a
    missing key.  Every exit path fills this skeleton.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": None,
        "protocol_version": protocol.get("protocol_version"),
        "protocol_sha256": digest(protocol),
        "metric_unit": METRIC_UNIT,
        "batches": [],
        # Zero-filled rather than empty: a failed run still renders, and the
        # renderer reads these counts.
        "coverage": {
            "manifests": 0,
            "not_reproducible": 0,
            "refused": 0,
            "rows": 0,
            "run_costs": 0,
        },
        "rows": [],
        "run_costs": [],
        "refusals": [],
        "checks": [],
        "not_recorded": {
            "metrics": [
                name for name in ("f1", "precision", "recall") if name not in CARRIED_METRICS
            ],
            "reason": (
                "protocol §1.2 logs these as additional test-stage metrics; the manifest "
                "carries accuracy and auc only, so P2.2 does not report them"
            ),
        },
        "overall": "pass",
    }


def build_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Audit, gate per group, and summarize.  Never reads a clock."""
    protocol = load_protocol()
    expected_seeds = list(protocol.get("seeds", []))
    if not expected_seeds:
        # Every row would read as incomplete against an empty seed list, so this
        # is refused rather than reported as a batch-wide defect.
        raise SummaryError("the loaded protocol carries no seed list")

    try:
        entries = audit.discover_batches(config)
    except audit.ConfigError as exc:
        report = empty_summary(protocol=protocol)
        report["checks"] = audit.build_audit(config)["checks"]
        report["overall"] = "fail"
        report["error"] = str(exc)
        return report

    audit_report = audit.build_audit(config, entries=entries)
    loaded = [entry for entry in entries if "payload" in entry]
    for entry in loaded:
        status, reasons, _ = audit.preflight_status(entry["payload"])
        entry["status"] = status
        entry["reasons"] = reasons

    # Phase B: one gate call per group, so a refusal is attributable.
    buckets: dict[tuple, list[dict[str, Any]]] = {}
    for entry in loaded:
        if entry["status"] == "not_reproducible":
            continue
        buckets.setdefault(group_key(entry["payload"]), []).append(entry)

    gate_reasons: dict[str, list[str]] = {}
    failures: list[tuple[list, str]] = []
    for members in sorted(buckets.values(), key=lambda group: str(group[0]["path"])):
        paths = [entry["path"] for entry in members]
        try:
            outcome = gate.aggregate(paths, protocol=protocol, require_formal=False)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            # The gate names the rule, not the file, so the whole group inherits
            # the reason and the finding names every member.
            _status, codes = status_for_gate_error(str(exc))
            failures.append((paths, str(exc)))
            for entry in members:
                gate_reasons[str(entry["path"])] = codes
            continue
        by_path = {str(item["path"]): item["reason"] for item in outcome["refused"]}
        for entry in members:
            reason = by_path.get(str(entry["path"]))
            if reason:
                entry["refused_reason"] = reason

    rows, runs, refusals = collect_observations(loaded, metric=PRIMARY_METRIC)
    primary_rows = build_rows(rows, expected_seeds=expected_seeds, gate_reasons=gate_reasons)

    # The audit already ran the same gate over the same groups, so its A08 check
    # is the one this report carries; the per-path reasons above are this
    # module's own use of the same mapping.
    checks = audit_report["checks"]
    report = empty_summary(protocol=protocol)
    report["batches"] = sorted({entry["batch"] for entry in loaded})
    report["coverage"] = {
        "manifests": len(loaded),
        "not_reproducible": sum(1 for e in loaded if e["status"] == "not_reproducible"),
        "refused": len(refusals),
        "rows": len(primary_rows),
        "run_costs": len(runs),
    }
    report["rows"] = primary_rows
    report["run_costs"] = build_run_costs(runs, expected_seeds=expected_seeds)
    report["refusals"] = refusals
    report["checks"] = checks
    report["overall"] = audit.overall_status(checks)
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P2.2 数值汇总",
        "",
        f"- 协议：`{report['protocol_version']}` / `{report['protocol_sha256']}`",
        f"- 主指标：`{PRIMARY_METRIC}`（均值与**样本**标准差，单位 `{report['metric_unit']}`）",
        f"- 行数：{report['coverage']['rows']}；run 成本条目：{report['coverage']['run_costs']}",
        f"- 结论：**{report['overall']}**",
        "",
        "> 行按 `(组, 视图, 机制, 方法, 选择协议, c)` 取键，**不含 seed**——seed 是聚合维度。",
        "> 成本按 run 记录，不复制到 PA/OA 两行。",
        "",
        "| 状态 | 方法 | 协议 | c | n | 均值 | 样本标准差 | 缺失 seed | reasons |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        identity, metric = row["result_identity"], row["metric"]
        mean = "—" if metric["mean"] is None else f"{metric['mean_percent']:.2f}%"
        std = "—" if metric["std"] is None else f"{metric['std_percent']:.2f}"
        lines.append(
            f"| {row['status']} | {identity['method']} | {identity['selection_protocol']} | "
            f"{identity['c_token']} | {metric['n_observed']}/{metric['n_expected']} | {mean} | "
            f"{std} | {metric['missing_seeds']} | {', '.join(row['reasons']) or '—'} |"
        )
    if report.get("refusals"):
        lines.extend(["", "## 门槛拒收的 run", ""])
        lines.extend(f"- `{item['reason']}` — `{item['path']}`" for item in report["refusals"])
    lines.extend(
        [
            "",
            "## 未记录的指标",
            "",
            f"- {', '.join(report['not_recorded']['metrics'])}：{report['not_recorded']['reason']}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write summary.json / summary.md / summary.csv / coverage.csv / refusals.json."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    json_path = target / "summary.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    written.append(json_path)

    md_path = target / "summary.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    written.append(md_path)

    costs = {item["cost_id"]: item["costs"] for item in report["run_costs"]}
    rows_path = target / "summary.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "status",
                "method",
                "dataset",
                "training_path",
                "mechanism",
                "selection_protocol",
                "c_token",
                "n_observed",
                "n_expected",
                "mean_fraction",
                "std_fraction",
                "missing_seeds",
                "tuning_cost_seconds",
                "peak_gpu_memory_bytes",
                "reasons",
            ]
        )
        for row in report["rows"]:
            identity, metric = row["result_identity"], row["metric"]
            cost = costs.get(row["cost_id"], {})
            tuning = cost.get("tuning_cost_seconds") or {}
            writer.writerow(
                [
                    row["status"],
                    identity["method"],
                    identity["dataset"],
                    identity["training_path"],
                    identity["labeling_mechanism"],
                    identity["selection_protocol"],
                    identity["c_token"],
                    metric["n_observed"],
                    metric["n_expected"],
                    metric["mean"],
                    metric["std"],
                    json.dumps(metric["missing_seeds"]),
                    tuning.get("sum"),
                    cost.get("peak_gpu_memory_bytes"),
                    json.dumps(row["reasons"], ensure_ascii=False),
                ]
            )
    written.append(rows_path)

    coverage_path = target / "coverage.csv"
    with coverage_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "count"])
        for name, value in sorted(report["coverage"].items()):
            writer.writerow([name, value])
    written.append(coverage_path)

    refusals_path = target / "refusals.json"
    refusals_path.write_text(
        json.dumps(report["refusals"], indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    written.append(refusals_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="batch-root whitelist JSON")
    parser.add_argument("--out-dir", required=True, help="where the summary artifacts go")
    args = parser.parse_args(argv)
    try:
        report = build_summary(audit.load_config(args.config))
    except (AuditError, SummaryError, audit.ConfigError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    for path in write_report(report, args.out_dir):
        print(f"wrote {path}")
    print(f"overall: {report['overall']}; rows: {report['coverage']['rows']}")
    return 0 if report["overall"] != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
