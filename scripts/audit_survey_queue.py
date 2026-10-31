"""Reconcile runnable fit coverage without mutating a live queue or training.

Counts are fits, not PA/OA rows. Correction campaigns have separate ownership
and never inflate the baseline denominator. Artifact checks establish queue
completion only, not formal admission or numerical paper reproduction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pu_toolbox.experiment.pilot_plan import manifest_identity, planned_runs
from pu_toolbox.experiment.queue_scheduling import validate_fit_ownership
from pu_toolbox.experiment.survey_protocol import digest, load_protocol


def reconcile(expected, previous, jobs, verify_job):
    """Pure coverage accounting plus an explicitly supplied artifact verifier."""
    expected = set(expected)
    previous = list(previous)
    if None in previous or len(set(previous)) != len(previous):
        raise ValueError("previous fit identities are missing or duplicated")
    owners = validate_fit_ownership(jobs)
    baseline = {key for key in owners if len(key) == 6}
    prior = set(previous)
    if prior & baseline:
        raise ValueError("previous fits also owned by the continuation queue")
    statuses = {"pending", "running", "completed", "failed", "blocked", "held"}
    if any(j.get("status") not in statuses for j in jobs):
        raise ValueError("unknown queue status")
    counts, corrections, invalid = Counter(), Counter(), []
    verified = set(prior)
    for job in jobs:
        target = corrections if job.get("campaign_id") else counts
        target[job["status"]] += job["expected_runs"]
        if job["status"] == "completed":
            if verify_job(job):
                target["artifact_verified_completed"] += job["expected_runs"]
                if not job.get("campaign_id"):
                    verified.update(tuple(k) for k in job["expected_keys"])
            else:
                invalid.append(job["label"])
    missing = expected - (prior | baseline)
    extra = (prior | baseline) - expected
    return dict(
        planned_fits=len(expected),
        previous_fits=len(prior),
        continuation_fits=len(baseline),
        baseline_status_fits=dict(counts),
        correction_status_fits=dict(corrections),
        completed_fit_lower_bound=len(verified & expected),
        remaining_fit_upper_bound=len(expected - verified),
        missing_fit_keys=sorted(missing, key=repr),
        unexpected_fit_keys=sorted(extra, key=repr),
        invalid_completed_jobs=invalid,
        coverage_complete=not missing and not extra,
        completed_artifacts_valid=not invalid,
        formal_admission_verified=False,
    )


def audit(queue, protocol_path):
    """Read an atomic queue snapshot and its pinned previous completion set."""
    from run_survey_parallel_queue import successful_artifacts

    state = json.loads((queue / "queue_status.json").read_text())
    evidence = json.loads((queue / "batch_evidence.json").read_text())
    protocol = load_protocol(protocol_path)
    protocol_hash = digest(protocol)
    if evidence["protocol_sha256"] != protocol_hash:
        raise ValueError("protocol differs from this queue's pinned protocol")
    if any(j["protocol_sha256"] != protocol_hash for j in state["jobs"]):
        raise ValueError("job protocol differs from the pinned queue protocol")
    previous = []
    for filename in evidence["previous_manifest_paths"]:
        manifest = json.loads(Path(filename).read_text())
        if (
            manifest.get("protocol_sha256") != protocol_hash
            or manifest.get("execution_mode") != "versioned_pilot"
            or manifest.get("failures")
            or not manifest.get("selection")
            or not manifest.get("test_results")
        ):
            raise ValueError(f"previous completion manifest is invalid: {filename}")
        previous.append(manifest_identity(manifest))
    report = reconcile(
        (r.key for r in planned_runs(protocol)), previous, state["jobs"], successful_artifacts
    )
    report.update(
        snapshot_at=state["updated_at"],
        audited_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=protocol_hash,
        scheduler=state.get("scheduler"),
        scope=(
            "read-only queue coverage and completion receipt checks; running partial outputs "
            "not counted; previous recipe/split acceptance inherited from batch evidence; "
            "not a fresh data, environment, metric or formal-admission audit"
        ),
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new report file; never overwrite an artifact")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.exists():
            raise ValueError("output already exists; choose a new report path")
        report = audit(args.queue, args.protocol)
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
        print(serialized)
        return 0 if report["coverage_complete"] and report["completed_artifacts_valid"] else 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
