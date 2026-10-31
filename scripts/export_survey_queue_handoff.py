"""Dated observation workbook with explicit correction replacement, not admission.

Read only a queue snapshot, its previous accepted manifests and completed jobs.
Do not scan abandoned retry roots, mix collaborator uploads, rewrite historical
artifacts or count PA/OA rows as independent fits. Files are generated outside Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from pu_toolbox.experiment.pilot_plan import manifest_identity
from pu_toolbox.experiment.queue_scheduling import validate_fit_ownership

GROUP = (
    "dataset",
    "profile",
    "training_path",
    "comparability_group",
    "view",
    "mechanism",
    "c",
    "selection",
    "protocol_sha256",
    "recipe_registry_sha256",
    "source_family",
)
METRICS = ("accuracy_pct", "auc", "f1", "precision", "recall")


def load_record(path, protocol_hash):
    path = Path(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"manifest changed during snapshot: {path}")
    manifest = json.loads(raw)
    key = manifest_identity(manifest)
    if key is None or manifest.get("failures") or manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError(f"invalid completed manifest: {path}")
    if not manifest.get("selection") or set(manifest["selection"]) != set(
        manifest.get("test_results", {})
    ):
        raise ValueError(f"selection/test tracks disagree: {path}")
    for metrics in manifest["test_results"].values():
        if metrics.get("accuracy") is None:
            raise ValueError(f"missing accuracy: {path}")
        for metric in ("accuracy", "auc", "f1", "precision", "recall"):
            value = metrics.get(metric)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"invalid metric {metric}: {path}")
    return dict(key=key, manifest=manifest, path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def collect(queue, verifier):
    state = json.loads((queue / "queue_status.json").read_text())
    evidence = json.loads((queue / "batch_evidence.json").read_text())
    validate_fit_ownership(state["jobs"])
    protocol_hash = evidence["protocol_sha256"]
    baseline, corrected, replacements = {}, {}, []
    for filename in evidence["previous_manifest_paths"]:
        record = load_record(filename, protocol_hash)
        if record["key"] in baseline:
            raise ValueError("duplicate previous fit")
        baseline[record["key"]] = record
    for job in state["jobs"]:
        if job["status"] != "completed":
            continue
        if not verifier(job):
            raise ValueError(f"completed job failed artifact verification: {job['label']}")
        target = corrected if job.get("campaign_id") else baseline
        for path in Path(job["root"]).rglob("manifest.json"):
            record = load_record(path, protocol_hash)
            if record["key"] in target:
                raise ValueError(f"duplicate fit within source family: {record['key']}")
            target[record["key"]] = record
    effective = dict(baseline)
    for key, record in corrected.items():
        if key not in baseline:
            raise ValueError("correction has no baseline fit to replace")
        original = baseline[key]
        replacements.append(
            dict(
                dataset=key[0],
                profile=key[1],
                training_path=key[2],
                mechanism=key[3],
                c=key[4],
                seed=key[5],
                old_manifest=original["path"],
                old_sha256=original["sha256"],
                corrected_manifest=record["path"],
                corrected_sha256=record["sha256"],
            )
        )
        effective[key] = record
    rows = []
    for key, record in sorted(effective.items(), key=lambda pair: repr(pair[0])):
        m = record["manifest"]
        for selection, metrics in m["test_results"].items():
            rows.append(
                dict(
                    dataset=key[0],
                    profile=key[1],
                    training_path=key[2],
                    mechanism=key[3] or "oracle",
                    c=key[4] or "independent",
                    seed=key[5],
                    selection=selection,
                    view=m.get("run_view"),
                    comparability_group=m.get("comparability_group"),
                    source_family="corrected" if key in corrected else "baseline",
                    accuracy_pct=100 * metrics["accuracy"],
                    **{metric: metrics.get(metric) for metric in METRICS[1:]},
                    tuning_seconds=m.get("resources", {}).get("tuning", {}).get("elapsed_seconds"),
                    peak_gpu_bytes=m.get("resources", {}).get("peak_gpu_memory_bytes"),
                    selected_epoch=m["selection"][selection].get("epoch"),
                    formal_eligible=False,
                    reported_manifest_formal_eligible=m.get("formal_eligible", False),
                    formal_blockers=";".join(m.get("formal_blockers", [])),
                    protocol_sha256=m.get("protocol_sha256"),
                    recipe_registry_sha256=m.get("recipe_registry_sha256"),
                    batch_identity_present=bool(m.get("batch")),
                    manifest=record["path"],
                    manifest_sha256=record["sha256"],
                )
            )
    return (
        rows,
        replacements,
        dict(
            snapshot_at=state["updated_at"],
            exported_at=datetime.now(timezone.utc).isoformat(),
            baseline_completed_fits=len(baseline),
            corrected_fits=len(corrected),
            effective_completed_fits=len(effective),
            result_rows=len(rows),
            formal_admission=False,
            previous_recipe_split_acceptance_inherited=True,
            baseline_remaining={
                status: sum(
                    j["expected_runs"]
                    for j in state["jobs"]
                    if not j.get("campaign_id") and j["status"] == status
                )
                for status in ("pending", "running", "failed", "blocked", "held")
            },
        ),
    )


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in GROUP)].append(row)
    complete, incomplete = [], []
    for key, members in sorted(groups.items(), key=lambda pair: repr(pair[0])):
        seeds = [r["seed"] for r in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError("duplicate seed within result group")
        record = dict(
            zip(GROUP, key, strict=True),
            seeds=",".join(map(str, sorted(seeds))),
            n_seeds=len(seeds),
            formal_eligible=False,
        )
        if set(seeds) != set(range(5)):
            record["missing_seeds"] = ",".join(map(str, sorted(set(range(5)) - set(seeds))))
            record["status"] = "incomplete_no_mean_or_conclusion"
            incomplete.append(record)
            continue
        record["status"] = "five_seed_observation_not_formal_admission"
        for metric in METRICS:
            values = [row.get(metric) for row in members]
            available = all(value is not None for value in values)
            record[f"{metric}_mean"] = statistics.mean(values) if available else None
            record[f"{metric}_std_ddof1"] = statistics.stdev(values) if available else None
        complete.append(record)
    return complete, incomplete


def export(queue, output):
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter
    from run_survey_parallel_queue import successful_artifacts

    if output.exists():
        raise ValueError("output directory exists; choose a new dated snapshot")
    rows, replacements, summary = collect(queue, successful_artifacts)
    complete, incomplete = summarize(rows)
    summary.update(five_seed_groups=len(complete), incomplete_groups=len(incomplete))
    notes = [
        dict(
            item="Scope",
            detail="Local observations; excludes collaborator workbooks; not formal admission.",
        ),
        dict(
            item="Corrections",
            detail=(
                f"{len(replacements)} repairs replace matching fits in this export only; "
                "originals unchanged. See replacements."
            ),
        ),
        dict(
            item="Units",
            detail=(
                "accuracy_pct is percent; std is percentage points. "
                "Other metrics are 0..1; std uses ddof=1."
            ),
        ),
        dict(
            item="Counts",
            detail="One fit can have PA/OA rows; missing seeds have no mean or conclusion.",
        ),
        dict(
            item="Validation",
            detail=(
                "Queue completion/recipe checks and metric ranges; "
                "no fresh raw-data, environment-lock or weight-replay audit."
            ),
        ),
        dict(
            item="Prior batches",
            detail=(
                "Previous split/recipe acceptance is inherited. "
                "Early fits without batch packets remain explicitly flagged."
            ),
        ),
        dict(
            item="Clinical scope",
            detail="ADNI slot uses the documented substitute, not a clinical ADNI claim.",
        ),
    ] + [dict(item=k, detail=json.dumps(v, ensure_ascii=False)) for k, v in summary.items()]
    sheets = [
        ("readme", notes),
        ("SCAR_PA", [r for r in complete if r["mechanism"] == "scar" and r["selection"] == "PA"]),
        ("SCAR_OA", [r for r in complete if r["mechanism"] == "scar" and r["selection"] == "OA"]),
        (
            "SAR_OA",
            [r for r in complete if r["mechanism"].startswith("sar") and r["selection"] == "OA"],
        ),
        ("PN_oracle", [r for r in complete if r["mechanism"] == "oracle"]),
        ("PA_clean_support", [r for r in complete if r["selection"] == "PA+clean-support"]),
        ("five_seed_all", complete),
        ("incomplete", incomplete),
        ("per_run", rows),
        ("replacements", replacements),
    ]
    book = Workbook()
    book.remove(book.active)
    for title, records in sheets:
        sheet = book.create_sheet(title)
        fields = list(dict.fromkeys(k for row in records for k in row)) or ["empty"]
        sheet.append(fields)
        for row in records:
            sheet.append([row.get(k) for k in fields])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for index, cell in enumerate(sheet[1], start=1):
            cell.font = Font(bold=True)
            sheet.column_dimensions[get_column_letter(index)].width = min(
                45, max(14, len(str(cell.value)) + 2)
            )
    output.mkdir(parents=True, exist_ok=False)
    filename = output / "PU_Survey_results_20261101.xlsx"
    book.save(filename)
    with ZipFile(filename) as archive:
        if archive.testzip() is not None:
            raise ValueError("workbook ZIP integrity failure")
    check = load_workbook(filename, read_only=True, data_only=True)
    if (
        check["per_run"].max_row != len(rows) + 1
        or check["replacements"].max_row != len(replacements) + 1
    ):
        raise ValueError("workbook row-count readback mismatch")
    check.close()
    for name, records in (
        ("per_run.csv", rows),
        ("five_seed.csv", complete),
        ("replacements.csv", replacements),
    ):
        fields = list(dict.fromkeys(k for row in records for k in row)) or ["empty"]
        with (output / name).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    contents = sorted(p for p in output.iterdir() if p.is_file())
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in contents)
    )
    with ZipFile(
        output / "PU_Survey_results_20261101.zip", "x", compression=ZIP_DEFLATED
    ) as archive:
        for path in [*contents, output / "SHA256SUMS.txt"]:
            archive.write(path, path.name)
    print(json.dumps(dict(output=str(output), **summary), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.queue.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
