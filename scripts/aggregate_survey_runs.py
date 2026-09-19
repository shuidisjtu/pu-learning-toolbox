#!/usr/bin/env python
"""Aggregate versioned survey pilot manifests into path-separated groups.

This is the production caller the leaderboard-separation gates never had.
``validate_comparable_manifests`` checked one dataset/seed/c group and
``partition_fair_leaderboard_runs`` partitioned by training path, but nothing
in a real run path went through either -- path separation was a library
function plus a manifest field.  This entry point forces both.

Three things a naive version would get wrong, all pinned by tests:

* **Group by ``comparability_group``, not by dataset.**  One dataset spans
  budget families (a closed-form row caps no epochs, a mini-batch row caps
  hundreds), and mixing them trips the fairness check on rows the protocol
  intends to list side by side.
* **Sub-group by ``(seed, c)`` inside each partition.**  ``partition_fair_
  leaderboard_runs`` keys its output by ``dataset/training_path`` across every
  seed, while the comparability gate compares ``seed`` for exact equality.
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
    digest,
    load_protocol,
    validate_comparable_manifests,
)

SCHEMA_VERSION = "1.0"
_MANIFEST_NAME = "manifest.json"
#: Fields a versioned manifest must carry for the gates to read it.  The
#: comparability gate indexes ``representation`` directly, so a missing one
#: would escape as a KeyError instead of naming the field.
_REQUIRED_FIELDS = ("protocol_version", "execution_unit", "representation", "budget")


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
    budget = manifest["budget"]
    if "split_sha256" not in representation:
        raise ValueError("manifest representation is missing split_sha256")
    return LeaderboardRunSpec(
        method=unit["method"],
        dataset=unit["dataset"],
        training_path=manifest["training_path"],
        adaptation_level=manifest["adaptation_level"],
        split_sha256=representation["split_sha256"],
        representation_sha256=digest(representation),
        max_epochs=_shared_budget_value(budget, "epochs"),
        batch_size_candidates=(_shared_budget_value(budget, "batch_size"),),
        tuning_candidate_count=len(manifest["candidate_runs"]),
        seeds=tuple(seeds),
    )


def _shared_budget_value(budget: dict, field: str) -> int:
    """A value for a fairness field this budget family does not define.

    Four of the seven budget families cap no epochs and define no batch size:
    a closed-form solve has neither.  The gate requires positive ints and only
    ever compares a field within a group, where every member shares one budget
    family -- the group key is what keeps families apart, so a placeholder
    here never decides anything.
    """
    return budget.get(field) or 1


def _refusal_reason(payload: dict) -> str | None:
    """Why this manifest cannot be aggregated, or ``None`` if it can."""
    if payload.get("execution_mode") == "rejected_versioned_pilot":
        return "rejected_versioned_pilot"
    if not isinstance(payload.get("execution_unit"), dict):
        return "no_execution_unit"
    for field in _REQUIRED_FIELDS:
        if field not in payload:
            # A manifest that announces itself as a survey unit but lacks a
            # field the gates read is malformed, not merely non-aggregatable.
            raise ValueError(f"versioned manifest is missing {field}")
    return None


def _unit_report(key: tuple, members: list[tuple[Path, dict]], *, require_formal: bool) -> dict:
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
    return {
        "seed": seed,
        "c": c,
        "state": "blocked" if blockers and require_formal else "comparable",
        "methods": methods,
        "blockers": blockers,
        "formal_eligible": not blockers,
    }


def _group_report(
    key: str, members: list[tuple[Path, dict]], *, protocol: dict, require_formal: bool
) -> dict:
    seeds = list(protocol.get("seeds", []))
    specs = [run_spec_from_manifest(payload, seeds=seeds) for _, payload in members]
    partitions = partition_fair_leaderboard_runs(specs)
    if len(partitions) != 1:
        raise ValueError(
            f"comparability group {key!r} spans training paths {sorted(partitions)}; "
            "the group key must pin one dataset and path"
        )
    fairness = next(iter(partitions.values()))

    by_unit: dict[tuple, list[tuple[Path, dict]]] = {}
    for path, payload in members:
        by_unit.setdefault(unit_key(payload), []).append((path, payload))
    units = [
        _unit_report(unit, by_unit[unit], require_formal=require_formal)
        for unit in sorted(by_unit, key=str)
    ]
    return {
        "comparability_group": key,
        "dataset": fairness["dataset"],
        "training_path": fairness["training_path"],
        "methods": fairness["methods"],
        "fairness_sha256": fairness["fairness_sha256"],
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
        print(
            f"\n[{group['comparability_group']}] {group['dataset']}/{group['training_path']} "
            f"fairness={group['fairness_sha256'][:12]}…"
        )
        print(f"  methods: {', '.join(group['methods'])}")
        for unit in group["units"]:
            marker = "ok" if unit["state"] == "comparable" else "BLOCKED"
            print(f"  seed={unit['seed']} c={unit['c']}: {marker} ({', '.join(unit['methods'])})")
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
