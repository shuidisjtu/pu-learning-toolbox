#!/usr/bin/env python
r"""Audit the P2.1 batch result trees and report every finding, not the first.

Three things a naive version gets wrong, all pinned by tests:

* **It must not recurse into a merged working copy.**  ``discover_manifests``
  walks a root recursively, so pointing it at a parent that holds both a batch
  and a flattened copy of that batch counts every manifest twice.  Roots come
  from a whitelist file instead, and known working-copy directory names are
  refused even if someone whitelists a parent containing one.
* **A group-level refusal may not be pinned on a manifest.**  The fairness
  gates raise on the first bad group and name the rule, not the file.  Each group
  is therefore gated on its own -- bucketed by the gate's own ``group_key`` --
  so the finding names every member instead of inventing one culprit.
* **A check whose inputs are absent is ``not_run``, not ``pass``.**  Archive
  existence, snapshot wording and probe separation need inputs this entry point
  does not read; recording them as passes would make a gap look like coverage.

The report is written to ``--out-dir``; nothing under a result root is written,
moved or deleted.

Run:  uv run python scripts/audit_survey_batches.py --config <config.json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ``scripts/`` is not a package, so the sibling entry point is reached through
# this file's own directory.  Resolving it from ``__file__`` keeps both entry
# points in the same checkout whatever the working directory is.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import aggregate_survey_runs as gate  # noqa: E402

from pu_toolbox.experiment.manifest import load_manifest  # noqa: E402
from pu_toolbox.experiment.survey_audit import (  # noqa: E402
    PROBE_ROLE,
    AuditError,
    make_check,
    overall_status,
    preflight_status,
    rollup_gate_check,
)
from pu_toolbox.experiment.survey_audit import (  # noqa: E402
    SCHEMA_VERSION as AUDIT_SCHEMA_VERSION,
)
from pu_toolbox.experiment.survey_comparison import (  # noqa: E402
    expected_result_units,
    load_comparison_protocol,
)
from pu_toolbox.experiment.survey_protocol import digest, load_protocol  # noqa: E402
from pu_toolbox.experiment.survey_summary import (  # noqa: E402
    STATUSES,
    group_key,
    matrix_selection_protocol,
    mechanism_of,
    normalize_c_token,
    result_labeling_mechanism,
)

CONFIG_SCHEMA_VERSION = "survey-batch-roots-1"

#: Directory names that hold a derived copy rather than a batch of its own.
#: A manifest under one of these would be counted twice, once as itself and once
#: as its flattened copy, so discovery refuses the whole run instead.
WORKING_COPY_DIRS = ("B3ab_merged",)

#: Checks this entry point cannot decide, and why.  Recorded so that a reader
#: sees them as unwired rather than as reviewed-and-clean.
_UNWIRED_CHECKS = {
    "A04": "plan identity needs the batch plan JSONs, which are not wired yet",
    "A05": "exit codes and completion lines live in the run logs, not the manifests",
    "A11": "archive existence and digests need the backup directories",
    "A14": "snapshot wording is a documentation check, not a manifest one",
    "A16": "ablation disclosure is a snapshot/report check, not a manifest one",
    "A17": "selection-path isolation is asserted by the protocol test suite",
}


class ConfigError(ValueError):
    """A batch-root config that cannot be used as given."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Read and validate the batch-root whitelist."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"config schema_version must be {CONFIG_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    batches = payload.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ConfigError("config needs a non-empty 'batches' list")
    names = [batch.get("name") for batch in batches]
    if len(set(names)) != len(names):
        raise ConfigError(f"batch names repeat: {sorted(names)}")
    for batch in batches:
        if not isinstance(batch.get("root"), str) or not batch["root"]:
            raise ConfigError(f"batch {batch.get('name')!r} needs a 'root'")
        if not isinstance(batch.get("expected_manifests"), int):
            raise ConfigError(f"batch {batch.get('name')!r} needs an expected count")
        if batch.get("role", "formal") not in {"formal", "technical_probe"}:
            raise ConfigError(f"batch {batch.get('name')!r} has an unknown role")
    return payload


def discover_batches(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every whitelisted manifest, with the batch and path it came from.

    Resolution is per batch and by exact root, so two batches never see each
    other's files, and a missing root is a finding rather than a silent zero.
    """
    excluded = tuple(config.get("excluded_subpaths", WORKING_COPY_DIRS))
    entries: list[dict[str, Any]] = []
    for batch in config["batches"]:
        root = Path(batch["root"])
        if not root.is_dir():
            entries.append({"batch": batch["name"], "missing_root": str(root)})
            continue
        for path in sorted(root.rglob("manifest.json")):
            relative = path.relative_to(root)
            if any(part in excluded for part in relative.parts):
                raise ConfigError(
                    f"{path} sits under a working-copy directory {excluded}; "
                    "whitelist the batch roots themselves, not a parent holding copies"
                )
            entries.append(
                {
                    "batch": batch["name"],
                    "role": batch.get("role", "formal"),
                    "path": path,
                    "payload": load_manifest(path),
                }
            )
    return entries


def check_coverage(config: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    """A02: every batch holds the number of formal manifests it should.

    Probe batches are counted separately and never enter the formal total.
    """
    observed = Counter(entry["batch"] for entry in entries if "payload" in entry)
    probe = {entry["batch"] for entry in entries if entry.get("role") == "technical_probe"}
    per_batch = {
        batch["name"]: {
            "expected": batch["expected_manifests"],
            "observed": observed.get(batch["name"], 0),
            "role": batch.get("role", "formal"),
        }
        for batch in config["batches"]
    }
    formal_expected = sum(
        item["expected"] for item in per_batch.values() if item["role"] == "formal"
    )
    formal_observed = sum(
        item["observed"] for item in per_batch.values() if item["role"] == "formal"
    )
    missing_roots = [entry["missing_root"] for entry in entries if "missing_root" in entry]
    mismatch = {
        name: item
        for name, item in per_batch.items()
        if item["role"] == "formal" and item["observed"] != item["expected"]
    }
    result = "pass" if not mismatch and not missing_roots else "fail"
    return make_check(
        check_id="A02",
        result=result,
        severity="error" if result == "fail" else "info",
        scope="batch",
        message=(
            f"{formal_observed}/{formal_expected} formal manifests"
            + (f"; mismatched batches {sorted(mismatch)}" if mismatch else "")
            + (f"; missing roots {missing_roots}" if missing_roots else "")
        ),
        observed={"total": formal_observed, "per_batch": per_batch, "probe_batches": sorted(probe)},
        expected={"total": formal_expected},
        evidence=missing_roots,
        reasons=["missing_manifest_field"] if missing_roots else [],
    )


def check_protocol(entries: list[dict[str, Any]], *, frozen_sha256: str) -> dict[str, Any]:
    """A03: every manifest reports the one frozen protocol digest."""
    digests = Counter(
        entry["payload"].get("protocol_sha256")
        for entry in entries
        if entry.get("role", "formal") == "formal"
    )
    others = {value: count for value, count in digests.items() if value != frozen_sha256}
    if not digests:
        return make_check(
            check_id="A03",
            result="not_run",
            severity="warning",
            scope="batch",
            message="no formal manifests to read a protocol digest from",
        )
    return make_check(
        check_id="A03",
        result="fail" if others else "pass",
        severity="error" if others else "info",
        scope="batch",
        message=(
            f"protocol digests disagree: {sorted(others)}"
            if others
            else f"one frozen digest across {sum(digests.values())} manifests"
        ),
        observed={"digests": {str(k): v for k, v in digests.items()}},
        expected={"protocol_sha256": frozen_sha256},
        reasons=["superseded_protocol"] if others else [],
    )


def check_whitelist(config: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    """A01: results come only from the whitelisted batch roots.

    Reported on the success path too, not only when discovery refuses: a check that
    appears only alongside an error tells a reader nothing about whether the
    whitelist was applied, which is the whole claim.
    """
    absent = [entry["missing_root"] for entry in entries if "missing_root" in entry]
    roots = {batch["name"]: batch["root"] for batch in config["batches"]}
    return make_check(
        check_id="A01",
        result="fail" if absent else "pass",
        severity="error" if absent else "info",
        scope="batch",
        message=(
            f"discovery read {len(roots)} whitelisted root(s), each by its own exact root"
            + (f"; absent: {absent}" if absent else "")
        ),
        observed={
            "batches": sorted(roots),
            "excluded_subpaths": list(config.get("excluded_subpaths", WORKING_COPY_DIRS)),
        },
        expected={"batches": sorted(roots)},
        evidence=absent,
    )


def check_selection(loaded: list[dict[str, Any]]) -> dict[str, Any]:
    """A06: every run that produced a result recorded the candidate it came from.

    ``candidate_index`` is what this reads, not ``checkpoint_index``: the classical
    estimators have no epoch checkpoints, so their ``checkpoint_index`` is
    legitimately ``null`` and demanding it would mark every classical run defective.
    A result whose selection names no candidate is a number nothing can be traced
    back to, which is the finding.
    """
    problems: list[str] = []
    for entry in loaded:
        payload = entry["payload"]
        selection = payload.get("selection")
        if not isinstance(selection, dict):
            problems.append(f"{entry['path']}: no selection block")
            continue
        for protocol in sorted(payload.get("test_results") or {}):
            picked = selection.get(protocol)
            if not isinstance(picked, dict) or picked.get("candidate_index") is None:
                problems.append(
                    f"{entry['path']}: {protocol} has a test result but names no selected candidate"
                )
    return make_check(
        check_id="A06",
        result="fail" if problems else "pass",
        severity="error" if problems else "info",
        scope="global",
        message=(
            f"{len(loaded) - len(problems)}/{len(loaded)} runs name the candidate behind "
            "their results"
            if problems
            else f"all {len(loaded)} runs name the candidate behind every result"
        ),
        observed={"runs": len(loaded), "without_selection": len(problems)},
        evidence=problems[:20],
    )


def check_view_mechanism(loaded: list[dict[str, Any]]) -> dict[str, Any]:
    """A07: the view and calibration distribution the leaderboard boundaries rest on.

    Reported as a distribution rather than only as a verdict, because B3b's TS rows
    and every OS row differ exactly here and a reader has to be able to see the
    split.  ``calibration_applied`` is what this adds: ``run_view`` is already
    required structurally, so its absence is A12's finding.
    """
    distribution: Counter[tuple[str, Any, Any]] = Counter()
    unrecorded: list[str] = []
    for entry in loaded:
        payload = entry["payload"]
        if "calibration_applied" not in payload:
            unrecorded.append(f"{entry['path']}: no calibration_applied")
            continue
        distribution[
            (mechanism_of(payload), payload.get("run_view"), payload["calibration_applied"])
        ] += 1
    observed = {
        f"{mechanism}/{view}/calibrated={calibrated}": count
        for (mechanism, view, calibrated), count in sorted(distribution.items(), key=str)
    }
    return make_check(
        check_id="A07",
        result="fail" if unrecorded else "pass",
        severity="error" if unrecorded else "info",
        scope="global",
        message=(
            f"{len(observed)} view x mechanism x calibration combination(s) recorded"
            + (f"; {len(unrecorded)} run(s) do not record calibration" if unrecorded else "")
        ),
        observed={"distribution": observed},
        evidence=unrecorded[:20],
    )


def check_result_completeness(
    loaded: list[dict[str, Any]], *, protocol: dict[str, Any]
) -> dict[str, Any]:
    """A09: every designed ``(c, protocol, seed)`` unit of a delivered row exists.

    The expectation comes from the protocol's own execution rows and c grid, never
    hand-listed.  Coverage is judged *within* the rows this whitelist delivers: which
    execution rows a batch is supposed to cover is the batch plan's statement (A04's
    input, not wired here), and a check that demanded every row of the protocol would
    report the datasets that have not run yet as defects of the runs that did.

    Mechanisms are named in the matrix's vocabulary here rather than the manifest's,
    because the comparison is against the protocol's execution rows, which are
    written that way: the protocol calls the PN oracle's row ``c_independent`` while
    the manifest records ``pn_oracle``, and reading the manifest's own name would
    report every oracle row as using a mechanism the protocol never defined.
    """
    seeds = sorted(int(seed) for seed in protocol.get("seeds", []))
    expected_by_row: dict[tuple, set[tuple[str, str, str]]] = {}
    for unit in expected_result_units(protocol):
        if unit["labeling_mechanism"] == "non_runnable":
            continue
        key = (unit["dataset"], unit["method"], unit["training_path"])
        expected_by_row.setdefault(key, set()).add(
            (unit["labeling_mechanism"], unit["c_token"], unit["selection_protocol"])
        )

    rows: dict[tuple, set[tuple]] = {}
    row_paths: dict[tuple, set[str]] = {}
    deviations: dict[str, list[str]] = {}
    for entry in loaded:
        payload = entry["payload"]
        unit = payload["execution_unit"]
        row = (
            unit["dataset"],
            unit["method"],
            payload["training_path"],
            result_labeling_mechanism(payload),
        )
        row_paths.setdefault(row, set()).add(str(entry["path"]))
        for name in sorted(payload.get("test_results") or {}):
            rows.setdefault(row, set()).add(
                (normalize_c_token(payload), matrix_selection_protocol(name), int(payload["seed"]))
            )
        if payload.get("protocol_deviation"):
            deviations.setdefault("/".join(row), []).append(
                f"{entry['path']}: {payload['protocol_deviation']}"
            )

    missing: list[str] = []
    undefined: list[str] = []
    problem_rows: set[tuple] = set()
    for (dataset, method, path, mechanism), observed in sorted(rows.items()):
        wanted = {
            unit
            for unit in expected_by_row.get((dataset, method, path), set())
            if unit[0] == mechanism
        }
        if not wanted:
            undefined.append(f"{dataset}/{method}/{path}/{mechanism}")
            problem_rows.add((dataset, method, path, mechanism))
            continue
        for _, c_token, selection_protocol in sorted(wanted):
            for seed in seeds:
                if (c_token, selection_protocol, seed) not in observed:
                    missing.append(
                        f"{dataset}/{method}/{path}/{mechanism}"
                        f"/c={c_token}/{selection_protocol}/seed={seed}"
                    )
                    problem_rows.add((dataset, method, path, mechanism))
    deviation_evidence = [
        f"protocol_deviation {key}: {item}"
        for key, items in sorted(deviations.items())
        for item in items[:2]
    ]
    incomplete = bool(missing or undefined)
    return make_check(
        check_id="A09",
        result="fail" if incomplete else "pass",
        severity="error" if incomplete else "info",
        # Scope follows the finding: a gap belongs to the rows it was found in, and
        # naming those rows' manifests is what makes it attributable.  A clean run
        # is a statement about the batch, and needs no members.
        scope="unit" if incomplete else "batch",
        message=(
            f"{len(rows)} delivered row(s) complete against the protocol's c grid and seeds"
            if not incomplete
            else f"{len(missing)} unit(s) missing; {len(undefined)} row(s) use a mechanism "
            "the protocol does not define for them"
        ),
        observed={
            "rows": len(rows),
            "missing_units": len(missing),
            "rows_with_undefined_mechanism": len(undefined),
            "rows_with_protocol_deviation": len(deviations),
        },
        expected={"seeds": seeds},
        evidence=missing[:20] + deviation_evidence,
        members=sorted(path for row in problem_rows for path in row_paths.get(row, set())),
        reasons=["protocol_deviation"] if deviations else [],
    )


def _reclaim_refs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``epoch_checkpoints`` references of a manifest that carry ``reclaimed``.

    The field is nested under ``candidate_runs[*].epoch_checkpoints[*]`` and is not a
    top-level key, so a reader looking for it at the top would find nothing and
    conclude that no batch reclaims anything.
    """
    return [
        checkpoint
        for candidate in payload.get("candidate_runs") or []
        for checkpoint in candidate.get("epoch_checkpoints") or []
        if isinstance(checkpoint, dict) and "reclaimed" in checkpoint
    ]


def check_reclaim(loaded: list[dict[str, Any]]) -> dict[str, Any]:
    """A10: ``reclaimed_true + files_on_disk == refs``, run by run.

    Only batches that actually reclaimed can be judged, so a tree whose manifests
    carry no ``reclaimed`` field at all is ``not_applicable`` rather than ``fail``
    (B1-B3b predate the field) or ``pass`` (nothing was checked).  The count is per
    manifest: one run's references cannot be reconciled against a whole result tree's
    file count.  Where the recorded checkpoint paths do not resolve on this host the
    count is refused rather than guessed, because a local success against a foreign
    path would be an accident.
    """
    applicable = {str(entry["path"]): _reclaim_refs(entry["payload"]) for entry in loaded}
    applicable = {path: refs for path, refs in applicable.items() if refs}
    if not applicable:
        return make_check(
            check_id="A10",
            result="not_applicable",
            severity="info",
            scope="global",
            message=(
                f"no manifest of {len(loaded)} carries the reclaimed field; reclaim "
                "accounting applies to batches run with reclamation enabled"
            ),
            observed={"manifests": len(loaded), "with_reclaim_references": 0},
        )
    unreachable: list[str] = []
    mismatched: list[str] = []
    for path, refs in sorted(applicable.items()):
        reclaimed = sum(1 for ref in refs if ref.get("reclaimed"))
        directories = {Path(str(ref["path"])).parent for ref in refs if ref.get("path")}
        if not directories or any(not directory.is_dir() for directory in directories):
            unreachable.append(f"{path}: recorded checkpoint directories do not resolve locally")
            continue
        on_disk = sum(1 for directory in directories for _ in directory.glob("epoch_*.pt"))
        if reclaimed + on_disk != len(refs):
            mismatched.append(
                f"{path}: reclaimed {reclaimed} + on disk {on_disk} != {len(refs)} references"
            )
    if unreachable:
        return make_check(
            check_id="A10",
            result="not_run",
            severity="warning",
            scope="group",
            message=f"{len(unreachable)} manifest(s) record checkpoint paths outside this tree",
            observed={"manifests_with_reclaim": len(applicable)},
            evidence=unreachable[:20],
            members=sorted(applicable),
        )
    return make_check(
        check_id="A10",
        result="fail" if mismatched else "pass",
        severity="error" if mismatched else "info",
        scope="group",
        message=(
            f"reclaimed + on disk balances against references in all {len(applicable)} manifest(s)"
            if not mismatched
            else f"{len(mismatched)} manifest(s) do not balance"
        ),
        observed={"manifests_with_reclaim": len(applicable), "unbalanced": len(mismatched)},
        evidence=mismatched[:20],
        members=sorted(applicable),
    )


def check_status_labels(loaded: list[dict[str, Any]]) -> dict[str, Any]:
    """A13: every label the report assigns is in the closed set, with its reasons.

    The invariant ``status == "formal"`` iff ``reasons == []`` is checked here rather
    than left to a reader, because it is what makes a label mean exactly one thing.
    The gate's own ``comparable`` / ``blocked`` vocabulary is deliberately not
    admitted: those describe a unit's state, and a report that copied one into
    ``status`` would put a value there that no consumer is allowed to read.
    """
    problems: list[tuple[str, str]] = []
    distribution: Counter[str] = Counter()
    for entry in loaded:
        status, reasons = entry.get("status"), entry.get("reasons") or []
        if status is None:
            continue
        distribution[status] += 1
        path = str(entry["path"])
        if status not in STATUSES:
            problems.append((path, f"{path}: status {status!r} is not in the closed set"))
        elif (status == "formal") != (not reasons):
            problems.append((path, f"{path}: status {status!r} carries reasons {list(reasons)}"))
    return make_check(
        check_id="A13",
        result="fail" if problems else "pass",
        severity="error" if problems else "info",
        scope="manifest" if len(problems) == 1 else "global",
        message=(
            f"{sum(distribution.values())} label(s): {dict(sorted(distribution.items()))}"
            if not problems
            else f"{len(problems)} label(s) violate the closed set or the reasons invariant"
        ),
        observed={"distribution": dict(sorted(distribution.items()))},
        expected={"statuses": list(STATUSES)},
        evidence=[message for _, message in problems[:20]],
        members=[path for path, _ in problems],
    )


def check_probe_separation(
    loaded: list[dict[str, Any]], *, protocol: dict[str, Any]
) -> dict[str, Any]:
    """A15: probe batches stay out of the formal total, and no non-runnable row ran.

    Two facts, both decidable here.  The coverage split is carried by A02's
    accounting -- probe batches never enter the formal total -- and restated as its
    own check so a reader sees the separation stated rather than inferred from
    another check's arithmetic.  The second is the protocol's own ``runnable=false``
    rows: a delivered row naming one of those is a manifest claiming a run the
    protocol forbids, whatever its numbers say.
    """
    probe = sorted({entry["batch"] for entry in loaded if entry.get("role") == PROBE_ROLE})
    formal = sorted({entry["batch"] for entry in loaded if entry.get("role", "formal") == "formal"})
    forbidden = {
        (unit["method"], unit["dataset"], unit["training_path"])
        for unit in expected_result_units(protocol)
        if unit["labeling_mechanism"] == "non_runnable"
    }
    impostors = sorted(
        f"{entry['path']}: {entry['payload']['execution_unit']['method']}"
        f"/{entry['payload']['execution_unit']['dataset']}/{entry['payload']['training_path']}"
        for entry in loaded
        if (
            entry["payload"]["execution_unit"]["method"],
            entry["payload"]["execution_unit"]["dataset"],
            entry["payload"]["training_path"],
        )
        in forbidden
    )
    if not probe and not impostors:
        return make_check(
            check_id="A15",
            result="not_applicable",
            severity="info",
            scope="batch",
            message="no probe batch is whitelisted and no non-runnable row is delivered",
            observed={"formal_batches": formal, "probe_batches": probe},
        )
    return make_check(
        check_id="A15",
        result="fail" if impostors else "pass",
        severity="error" if impostors else "info",
        scope="batch",
        message=(
            f"formal batches {formal}; probe batches {probe or 'none'}"
            + (
                f"; {len(impostors)} run(s) claim a row the protocol marks non-runnable"
                if impostors
                else ""
            )
        ),
        observed={
            "formal_batches": formal,
            "probe_batches": probe,
            "non_runnable_rows": sorted("/".join(item) for item in forbidden),
        },
        evidence=impostors[:20],
        members=sorted({entry["batch"] for entry in loaded if entry.get("role") == PROBE_ROLE}),
        reasons=["probe_only"] if probe else [],
    )


def empty_report(generated_at: str | None = None) -> dict[str, Any]:
    """The report skeleton every audit returns, so the schema has one home.

    ``generated_at`` stays ``None`` inside :func:`build_audit`: a report builder
    that read a clock could not be compared against itself, and the determinism
    check is exactly that comparison.  The CLI stamps the field on the way out.
    """
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "protocol_version": None,
        "protocol_sha256": None,
        "batches": [],
        "coverage": {},
        "checks": [],
        "overall": "pass",
    }


def build_audit(
    config: dict[str, Any], *, entries: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Run every check this entry point can decide, and name the ones it cannot.

    ``entries`` is accepted so a caller that also summarizes does not walk and
    load the tree twice; the audit is a pure function of what it is handed.
    """
    protocol = load_protocol()
    frozen = digest(protocol)
    report = empty_report()
    report["protocol_version"] = protocol.get("protocol_version")
    report["protocol_sha256"] = frozen

    try:
        entries = discover_batches(config) if entries is None else entries
    except ConfigError as exc:
        report["batches"] = [{"name": b["name"], "root": b["root"]} for b in config["batches"]]
        report["checks"] = [
            make_check(
                check_id="A01",
                result="fail",
                severity="error",
                scope="batch",
                message=str(exc),
            )
        ]
        report["overall"] = overall_status(report["checks"])
        return report

    loaded = [entry for entry in entries if "payload" in entry]
    report["batches"] = sorted({entry["batch"] for entry in loaded})
    report["coverage"] = {
        "manifests": dict(Counter(entry["batch"] for entry in loaded)),
        "total": len(loaded),
    }

    checks = [
        check_whitelist(config, entries),
        check_coverage(config, entries),
        check_protocol(loaded, frozen_sha256=frozen),
    ]

    # Phase A: per-manifest structure, grouped so one check names every offender.
    # The frozen matrix is loaded once and handed to the preflight: the loader
    # re-validates every mapping and anchor, which is too much work to repeat 610
    # times for the same answer.
    comparison = load_comparison_protocol()
    unusable: list[str] = []
    for entry in loaded:
        status, reasons, missing = preflight_status(entry["payload"], comparison=comparison)
        entry["status"] = status
        entry["reasons"] = reasons
        if status == "not_reproducible":
            unusable.append(f"{entry['path']}: {', '.join(missing) or ', '.join(reasons)}")
    entry_reasons = sorted({code for entry in loaded for code in entry.get("reasons", [])})
    checks.append(
        make_check(
            check_id="A12",
            result="pass" if not unusable else "fail",
            severity="error" if unusable else "info",
            scope="manifest" if len(unusable) == 1 else "global",
            message=(
                f"{len(loaded) - len(unusable)}/{len(loaded)} manifests carry their identity"
                if not unusable
                else f"{len(unusable)} manifest(s) are not reproducible"
            ),
            observed={"not_reproducible": len(unusable)},
            evidence=unusable[:20],
            members=[
                str(entry["path"]) for entry in loaded if entry.get("status") == "not_reproducible"
            ],
            reasons=entry_reasons,
        )
    )
    checks.extend(
        [
            check_selection(loaded),
            check_view_mechanism(loaded),
            check_result_completeness(loaded, protocol=protocol),
            check_reclaim(loaded),
            check_status_labels(loaded),
            check_probe_separation(loaded, protocol=protocol),
        ]
    )

    # Phase B: one gate call per group, so a refusal names its own members.
    usable = [entry for entry in loaded if entry.get("status") != "not_reproducible"]
    buckets: dict[tuple, list[dict[str, Any]]] = {}
    for entry in usable:
        buckets.setdefault(group_key(entry["payload"]), []).append(entry)

    failures: list[tuple[list, str]] = []
    refused_total = 0
    blocked_groups = 0
    for members in sorted(buckets.values(), key=lambda group: str(group[0]["path"])):
        paths = [entry["path"] for entry in members]
        try:
            outcome = gate.aggregate(paths, protocol=protocol, require_formal=False)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            failures.append((paths, str(exc)))
            continue
        refused_total += len(outcome["refused"])
        # A unit whose manifests carry formal blockers is what makes the gate's
        # batch-wide `formal_ready` false.  Recorded here because a reader would
        # otherwise see per-row "formal" verdicts beside a batch the gate holds
        # back, and read one of the two as wrong.
        if any(unit["blockers"] for unit in outcome["groups"][0]["units"]):
            blocked_groups += 1
    formal_ready = not failures and not refused_total and not blocked_groups
    checks.append(
        rollup_gate_check(check_id="A08", failures=failures)
        if failures
        else make_check(
            check_id="A08",
            result="pass",
            severity="info",
            scope="batch",
            message=(
                f"fairness gates ran over {len(buckets)} group(s); "
                f"formal_ready={formal_ready}"
                + (f" ({blocked_groups} group(s) carry blocked units)" if blocked_groups else "")
            ),
            observed={
                "groups": len(buckets),
                "groups_with_blocked_units": blocked_groups,
                "refused_runs": refused_total,
                "formal_ready": formal_ready,
            },
        )
    )

    for check_id, why in sorted(_UNWIRED_CHECKS.items()):
        checks.append(
            make_check(
                check_id=check_id,
                result="not_run",
                severity="warning",
                scope="global",
                message=why,
            )
        )

    report["checks"] = checks
    report["overall"] = overall_status(checks)
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P2.1 批次审计",
        "",
        f"- 协议：`{report['protocol_version']}` / `{report['protocol_sha256']}`",
        f"- 制品：{report['coverage'].get('total', 0)} 份 manifest",
        f"- 结论：**{report['overall']}**",
        "",
        "| check | 结果 | 作用域 | reasons | 说明 |",
        "|---|---|---|---|---|",
    ]
    for check in report["checks"]:
        reasons = ", ".join(check["reasons"]) or "—"
        lines.append(
            f"| {check['check_id']} | {check['result']} | {check['scope']} | "
            f"{reasons} | {check['message']} |"
        )
    for check in report["checks"]:
        if not check["evidence"]:
            continue
        lines.extend(["", f"## {check['check_id']} 证据（{len(check['evidence'])} 条）", ""])
        lines.extend(f"- `{item}`" for item in check["evidence"])
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write the audit artifacts and return their paths."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written = []
    json_path = target / "audit.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    written.append(json_path)
    md_path = target / "audit.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    written.append(md_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="batch-root whitelist JSON")
    parser.add_argument("--out-dir", required=True, help="where audit.json / audit.md go")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        report = build_audit(config)
    except (AuditError, ConfigError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    written = write_report(report, args.out_dir)
    for path in written:
        print(f"wrote {path}")
    print(f"overall: {report['overall']}")
    return 0 if report["overall"] != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
