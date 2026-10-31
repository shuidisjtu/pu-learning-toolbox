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
from pu_toolbox.experiment.survey_protocol import digest, load_protocol, profile_mechanisms


def protocol_backlog(protocol):
    """Inventory declared blockers AND absent template rows; never enable them.

    The 2026-11-01 progress report counted only the declared blocked units,
    omitting README section 2's later dataset expansion. These are conditional
    workload estimates, not approved run identities or a launch plan. The
    support exclusion policy comes from protocol section 2.4 / D50.
    """
    templates = {
        "connect_4": "spambase",
        "twenty_newsgroups": "imdb",
        "mnist": "cifar10",
        "fashion_mnist": "cifar10",
        "adni": "cifar10",
    }
    datasets = sorted({"spambase", "imdb", "cifar10", *templates})
    support_datasets = {"spambase", "connect_4", "mnist", "fashion_mnist", "cifar10"}
    units = protocol["execution_units"]
    profiles = protocol["method_profiles"]

    def fit_count(method):
        if method == "pn_oracle":
            return len(protocol["seeds"])
        allowed = profile_mechanisms(protocol, method)
        return len(protocol["seeds"]) * sum(
            len(tokens)
            for mechanism, tokens in protocol["c_tokens"].items()
            if allowed is None or mechanism in allowed
        )

    blocked = [
        dict(
            dataset=u["dataset"],
            profile=u["method"],
            training_path=u["training_path"],
            conditional_fits=fit_count(u["method"]),
            reason=u.get("non_runnable_reason"),
        )
        for u in units
        if u.get("runnable") is not True
    ]
    absent = []
    for method, profile in sorted(profiles.items()):
        existing = [u for u in units if u["method"] == method]
        support = profile.get("supervision") == "independent_clean_support"
        for dataset in datasets:
            if any(u["dataset"] == dataset for u in existing):
                continue
            source = templates.get(dataset)
            template_units = [u for u in existing if u["dataset"] == source]
            if support and dataset not in support_datasets:
                kind, count = "policy_excluded_support_dataset", 0
            elif template_units:
                kind, count = "missing_template_extension", len(template_units) * fit_count(method)
            else:
                kind, count = "missing_path_or_scope_decision", None
            absent.append(
                dict(
                    dataset=dataset,
                    profile=method,
                    classification=kind,
                    template_dataset=source,
                    template_paths=[u["training_path"] for u in template_units],
                    conditional_fits=count,
                )
            )
    profile_cells = {(u["dataset"], u["method"]) for u in units}
    algorithm_cells = {
        (u["dataset"], profiles[u["method"]].get("algorithm", u["method"])) for u in units
    }
    runnable_algorithm_cells = {
        (u["dataset"], profiles[u["method"]].get("algorithm", u["method"]))
        for u in units
        if u.get("runnable") is True
    }
    counts = Counter(row["classification"] for row in absent)
    return dict(
        scope="full_protocol_backlog_not_runnable_queue_coverage",
        conditional_estimates_only=True,
        runnable_fits=len(planned_runs(protocol)),
        declared_blocked_units=len(blocked),
        declared_blocked_conditional_fits=sum(row["conditional_fits"] for row in blocked),
        absent_profile_dataset_counts=dict(counts),
        missing_template_conditional_fits=sum(
            row["conditional_fits"]
            for row in absent
            if row["classification"] == "missing_template_extension"
        ),
        profile_dataset_cells_present=len(profile_cells),
        algorithm_dataset_cells_present=len(algorithm_cells),
        runnable_algorithm_dataset_cells=len(runnable_algorithm_cells),
        blocked_units=blocked,
        absent_profile_dataset_cells=absent,
        completion_warning=(
            "Queue coverage_complete covers runnable fits only; it is not project or "
            "formal-admission completion. Candidate campaigns cannot be subtracted as formal fits."
        ),
    )


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
        protocol_backlog=protocol_backlog(protocol),
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
