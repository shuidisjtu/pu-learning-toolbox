#!/usr/bin/env python
"""Aggregate versioned survey pilot manifests into path-separated groups.

This is the production caller the leaderboard-separation gates never had.
``validate_comparable_manifests`` checked one dataset/seed/c group and
``partition_fair_leaderboard_runs`` partitioned by training path, but nothing
in a real run path went through either -- path separation was a library
function plus a manifest field.  This entry point forces both.

Four things a naive version would get wrong, all pinned by tests:

* **Group by ``comparability_group``, not by dataset.**  One dataset spans
  budget families (a closed-form row caps no epochs, a mini-batch row caps
  hundreds), so a dataset-wide gate would have to compare rows the protocol
  keeps in separate leaderboards.  The group key separates *datasets*, not
  families -- its classical group holds four of them -- so what a group's
  members must agree on is the fairness fields the gates consume
  (``budget_fairness_fields``), not the raw budget dicts.
* **Gate each ``(seed, c)`` unit, not the whole group.**  ``partition_fair_
  leaderboard_runs`` compares ``split_sha256``, and the pilot gives every seed
  its own split, so a unit is the widest set of runs that can share one.  The
  comparability gate compares ``seed`` for exact equality for the same reason.
* **Build one spec per method, not per manifest.**  A pilot runs every method
  at every (seed, c), and the leaderboard gate keys a leaderboard on method: a
  spec per run reads a method as a duplicate of itself.  Every group in a real
  tree has that shape.
* **Take ``seeds`` from the protocol.**  A successful manifest never records
  the requested seed list, so deriving it from the files found would let a
  2-of-5-seed pilot satisfy "the seeds agree" trivially.

``--diagnostic`` relaxes eligibility only.  It is named for what the delivery
record already calls a technical diagnostic rather than for what it turns off,
and the fairness gates run in both modes -- an opt-out that also skipped those
would be a hole, not a mode.

Scope is the wiring.  The expected-unit completeness grid, anomaly units and
reproduction-field auditing belong to P2.2 and are deliberately absent; do not
read their absence here as an oversight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pu_toolbox.experiment.feature_adapter import (
    LeaderboardRunSpec,
    partition_fair_leaderboard_runs,
)
from pu_toolbox.experiment.manifest import load_manifest
from pu_toolbox.experiment.survey_protocol import (
    budget_fairness_fields,
    digest,
    load_protocol,
    validate_comparable_manifests,
)

SCHEMA_VERSION = "1.0"
_MANIFEST_NAME = "manifest.json"
#: Fields a versioned manifest must carry for the entry point to read it.  The
#: comparability gate indexes ``representation`` directly and the rest are read
#: while grouping and comparing, so a missing one would escape as a KeyError --
#: a crash rather than the malformed artifact it is.
_REQUIRED_FIELDS = (
    "protocol_version",
    "execution_unit",
    "representation",
    "budget",
    "candidate_runs",
    "generation",
    "adaptation_level",
    "seed",
    "training_path",
)

#: Fairness fields that must agree across the units of one comparability group.
#: The split and representation hashes are deliberately absent: every seed gets
#: its own split, so those differ between units by construction.
_GROUP_SHARED_FIELDS = (
    "dataset",
    "training_path",
    "adaptation_level",
    "max_epochs",
    "batch_size_candidates",
    "tuning_candidate_count",
)


def discover_manifests(out_root: str | Path) -> list[Path]:
    """Every manifest under ``out_root``, in a stable order."""
    root = Path(out_root)
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    return sorted(root.rglob(_MANIFEST_NAME))


def unit_key(manifest: dict) -> tuple[int, Any]:
    """The ``(seed, c)`` one manifest belongs to.

    A c-independent row (the PN oracle) is broadcast across c values rather
    than run at each, so it is its own unit rather than a member of every one.
    """
    if manifest.get("c_independent"):
        return manifest["seed"], "c_independent"
    return manifest["seed"], manifest["generation"]["train"]["c_requested"]


def run_spec_from_manifest(manifest: dict, *, seeds: list[int]) -> LeaderboardRunSpec:
    """Build the fairness-gate spec for one manifest.

    Every field is read through a named check: the gate treats a missing key
    as a ``KeyError``, which reads as a crash rather than as the malformed
    artifact it is.
    """
    unit = manifest["execution_unit"]
    representation = manifest["representation"]
    if "split_sha256" not in representation:
        raise ValueError("manifest representation is missing split_sha256")
    fairness = budget_fairness_fields(manifest["budget"])
    return LeaderboardRunSpec(
        method=unit["method"],
        dataset=unit["dataset"],
        training_path=manifest["training_path"],
        adaptation_level=manifest["adaptation_level"],
        split_sha256=representation["split_sha256"],
        representation_sha256=digest(representation),
        max_epochs=fairness["max_epochs"],
        batch_size_candidates=fairness["batch_size_candidates"],
        tuning_candidate_count=len(manifest["candidate_runs"]),
        seeds=tuple(seeds),
    )


def _refusal_reason(payload: dict) -> str | None:
    """Why this manifest cannot be aggregated, or ``None`` if it can."""
    if payload.get("execution_mode") == "rejected_versioned_pilot":
        return "rejected_versioned_pilot"
    unit = payload.get("execution_unit")
    if not isinstance(unit, dict):
        return "no_execution_unit"
    if unit.get("runnable") is not True:
        # The matrix marks the rows it never runs, and one of those reaching a
        # leaderboard is what the path separation exists to prevent.  A row
        # carrying no flag at all is not one this entry point may assume ran.
        return "execution_unit_not_runnable"
    if not payload.get("selection"):
        # Every candidate was excluded: the runner writes this manifest and then
        # re-raises, so the artifact explains a failure rather than recording a
        # result.  Ranking it would put a run that produced nothing beside runs
        # that produced something, and no field inside it says so.
        return "no_selected_candidate"
    for field in _REQUIRED_FIELDS:
        if field not in payload:
            # A manifest that announces itself as a survey unit but lacks a
            # field the entry point reads is malformed, not merely
            # non-aggregatable.
            raise ValueError(f"versioned manifest is missing {field}")
    for field in ("budget", "generation", "representation"):
        if not isinstance(payload[field], dict):
            raise ValueError(f"versioned manifest {field} is not an object")
    for field in ("method", "dataset", "comparability_group"):
        if not isinstance(unit.get(field), str):
            raise ValueError(f"versioned manifest execution_unit is missing {field}")
    if not isinstance(payload["candidate_runs"], list):
        raise ValueError("versioned manifest candidate_runs is not a list")
    if not isinstance(payload.get("formal_blockers", []), list):
        raise ValueError("versioned manifest formal_blockers is not a list")
    if not isinstance(payload["representation"].get("feature_sha256"), dict):
        raise ValueError("versioned manifest representation is missing feature_sha256")
    if not payload.get("c_independent"):
        train_view = payload["generation"].get("train")
        if not isinstance(train_view, dict) or "c_requested" not in train_view:
            raise ValueError("versioned manifest generation is missing the train label view")
    return None


def _unit_report(
    key: tuple, members: list[tuple[Path, dict]], *, seeds: list[int], require_formal: bool
) -> dict:
    """One (seed, c) unit: fairness always checked, eligibility optionally reported."""
    seed, c = key
    payloads = [payload for _, payload in members]
    methods = sorted(payload["execution_unit"]["method"] for payload in payloads)
    blockers = sorted(
        {
            blocker
            for payload in payloads
            if not payload.get("formal_eligible", False)
            for blocker in payload.get("formal_blockers", [])
        }
    )
    # The fairness checks are unconditional; only eligibility is optional.  A
    # blocked unit is still compared, so a mismatch inside it is not hidden by
    # the fact that its results are not yet formal.
    validate_comparable_manifests(payloads, require_formal=require_formal and not blockers)
    fairness = _unit_fairness(members, seeds=seeds)
    return {
        "seed": seed,
        "c": c,
        "state": "blocked" if blockers and require_formal else "comparable",
        "methods": methods,
        "blockers": blockers,
        "formal_eligible": not blockers,
        "fairness_sha256": fairness["fairness_sha256"],
    }


def _method_specs(
    members: list[tuple[Path, dict]], *, seeds: list[int]
) -> list[LeaderboardRunSpec]:
    """One spec per method: the leaderboard gate keys a leaderboard on method.

    The gate refuses a method listed twice, so a unit holding several runs of
    one method has to collapse to a single spec.  The collapsed runs also have
    to agree: two manifests for one method in one unit are a re-run that changed
    something, which is worth naming rather than silently taking the first of.
    """
    by_method: dict[str, list[tuple[Path, dict]]] = {}
    for path, payload in members:
        by_method.setdefault(payload["execution_unit"]["method"], []).append((path, payload))

    specs: list[LeaderboardRunSpec] = []
    for method in sorted(by_method):
        runs = by_method[method]
        spec = run_spec_from_manifest(runs[0][1], seeds=seeds)
        for _, payload in runs[1:]:
            if run_spec_from_manifest(payload, seeds=seeds) != spec:
                raise ValueError(f"method {method!r} appears twice with different specs")
        specs.append(spec)
    return specs


def _unit_fairness(members: list[tuple[Path, dict]], *, seeds: list[int]) -> dict[str, Any]:
    """The leaderboard partition for one ``(seed, c)`` unit.

    The gate compares ``split_sha256``, and the pilot gives every seed its own
    split, so a unit is the widest set of runs that can share one: handing it a
    whole comparability group would compare five seeds' splits and refuse the
    group.  The group key already pins dataset and path, so a unit spanning two
    paths is a protocol defect rather than a partition to report.
    """
    partitions = partition_fair_leaderboard_runs(_method_specs(members, seeds=seeds))
    if len(partitions) != 1:
        raise ValueError(
            f"unit spans training paths {sorted(partitions)}; "
            "a comparability group must pin one dataset and path"
        )
    return next(iter(partitions.values()))


def _group_consistency(members: list[tuple[Path, dict]], *, seeds: list[int]) -> None:
    """Invariants that span units, which no single unit can check.

    Every unit is gated on its own, so whatever varies *between* units has no
    other checker.  Re-running the missing seeds after a budget or protocol
    change is routine, and it leaves a group whose units are each internally
    consistent while disagreeing with each other -- ranking two training budgets
    side by side, or mixing two protocol revisions, and still reporting ready.
    """
    versions = sorted({payload.get("protocol_version") for _, payload in members})
    if len(versions) > 1:
        raise ValueError(f"comparability group mixes protocol versions {versions}")

    seen: dict[str, tuple[tuple, str]] = {}
    for _, payload in members:
        spec = run_spec_from_manifest(payload, seeds=seeds)
        signature = tuple(getattr(spec, field) for field in _GROUP_SHARED_FIELDS)
        budget = json.dumps(payload["budget"], sort_keys=True)
        reference = seen.setdefault(spec.method, (signature, budget))
        if reference[0] != signature:
            mismatched = [
                field
                for field, value, expected in zip(
                    _GROUP_SHARED_FIELDS, signature, reference[0], strict=True
                )
                if value != expected
            ]
            raise ValueError(
                f"method {spec.method!r} disagrees between units on {', '.join(mismatched)}"
            )
        if reference[1] != budget:
            # The fairness fields cover the epoch cap and the batch size, but a
            # re-run can move anything else the budget defines -- a learning
            # rate, an inner iteration count.  Comparing the whole dictionary is
            # safe here and only here: the comparison is per method, and a
            # method maps to one budget family, so the classical group's four
            # families still differ from each other.
            raise ValueError(f"method {spec.method!r} ran under different budgets across units")


def _group_report(
    key: str, members: list[tuple[Path, dict]], *, protocol: dict, require_formal: bool
) -> dict:
    seeds = list(protocol.get("seeds", []))
    _group_consistency(members, seeds=seeds)
    by_unit: dict[tuple, list[tuple[Path, dict]]] = {}
    for path, payload in members:
        by_unit.setdefault(unit_key(payload), []).append((path, payload))
    units = [
        _unit_report(unit, by_unit[unit], seeds=seeds, require_formal=require_formal)
        for unit in sorted(by_unit, key=str)
    ]
    return {
        "comparability_group": key,
        "dataset": members[0][1]["execution_unit"]["dataset"],
        "training_path": members[0][1]["training_path"],
        "methods": sorted({payload["execution_unit"]["method"] for _, payload in members}),
        "units": units,
    }


def aggregate(paths: list[Path], *, protocol: dict, require_formal: bool = True) -> dict[str, Any]:
    """Group, gate and report; never write anything back."""
    refused: list[dict] = []
    usable: list[tuple[Path, dict]] = []
    for path in paths:
        payload = load_manifest(path)
        reason = _refusal_reason(payload)
        if reason is None:
            usable.append((path, payload))
        else:
            refused.append({"path": str(path), "reason": reason})

    grouped: dict[str, list[tuple[Path, dict]]] = {}
    for path, payload in usable:
        grouped.setdefault(payload["execution_unit"]["comparability_group"], []).append(
            (path, payload)
        )

    groups = [
        _group_report(key, grouped[key], protocol=protocol, require_formal=require_formal)
        for key in sorted(grouped)
    ]

    ready = (
        require_formal
        and not refused
        and all(unit["state"] == "comparable" for group in groups for unit in group["units"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "formal": require_formal,
        "protocol_version": protocol.get("protocol_version"),
        "groups": groups,
        "refused": refused,
        "formal_ready": ready,
    }


def _print_report(report: dict) -> None:
    if not report["formal"]:
        print("NON-FORMAL (technical diagnostic) -- not a pilot leaderboard")
    print(f"protocol {report['protocol_version']}: {len(report['groups'])} group(s)")
    for group in report["groups"]:
        print(f"\n[{group['comparability_group']}] {group['dataset']}/{group['training_path']}")
        print(f"  methods: {', '.join(group['methods'])}")
        for unit in group["units"]:
            marker = "ok" if unit["state"] == "comparable" else "BLOCKED"
            print(
                f"  seed={unit['seed']} c={unit['c']}: {marker} "
                f"({', '.join(unit['methods'])}) fairness={unit['fairness_sha256'][:12]}…"
            )
            # In diagnostic mode the blockers are still worth printing, but
            # "blocked by" under an ok unit would read as a contradiction.
            label = "blocked by" if unit["state"] == "blocked" else "not formal"
            for blocker in unit["blockers"]:
                print(f"    {label}: {blocker}")
    for item in report["refused"]:
        print(f"\nnot aggregated: {item['reason']} -- {item['path']}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate versioned survey pilot manifests by comparability group."
    )
    parser.add_argument("out_root", help="results root containing run manifests")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "technical diagnostic: report non-formal results instead of refusing them. "
            "The fairness gates still run."
        ),
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        paths = discover_manifests(args.out_root)
        if not paths:
            print(f"error: no manifests under {args.out_root}", file=sys.stderr)
            return 1
        report = aggregate(paths, protocol=load_protocol(), require_formal=not args.diagnostic)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        _print_report(report)
    return 0 if (not report["formal"] or report["formal_ready"]) else 1


if __name__ == "__main__":
    sys.exit(main())
