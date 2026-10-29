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
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping
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

from pu_toolbox.experiment import batch_evidence  # noqa: E402
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
    comparison_digest,
    comparison_for_version,
    expected_result_units,
)
from pu_toolbox.experiment.survey_profile_identity import (  # noqa: E402
    valid_manifest_profile_identity,
)
from pu_toolbox.experiment.survey_profile_map import profile_registration_status  # noqa: E402
from pu_toolbox.experiment.survey_protocol import (  # noqa: E402
    digest,
    load_protocol,
    resolve_protocol_path,
)
from pu_toolbox.experiment.survey_provenance import (  # noqa: E402
    build_provenance,
    input_result_roots,
    recorded_source_roots,
    render_provenance_lines,
    resolve_code_commit,
    with_code_commit,
)
from pu_toolbox.experiment.survey_recipe_audit import (  # noqa: E402
    BATCH_SCHEMA_VERSION,
    audit_recipe_batches,
    recipe_audit_lines,
    validate_recipe_config,
)
from pu_toolbox.experiment.survey_recipe_registry import (  # noqa: E402
    load_registry,
    validate_registry,
)
from pu_toolbox.experiment.survey_summary import (  # noqa: E402
    STATUSES,
    group_key,
    matrix_selection_protocol,
    mechanism_of,
    normalize_c_token,
    result_labeling_mechanism,
)
from pu_toolbox.utils.serialization import file_hash  # noqa: E402

CONFIG_SCHEMA_VERSION = "survey-batch-roots-1"

# Historical global progress notes are not evidence against an individual completed row.
# Keep this allowlist exact: future or run-specific blockers require their own adjudication.
_REASSESSABLE_BLOCKERS = frozenset(
    {
        (
            "CVIR, KLDCE, P3MIX, PAN, PUET, RP, VPU, the GradPU native-2d path, and the "
            "cifar10 rows for LaGAM and PULNS still lack a frozen main-chain budget or runner."
        ),
        (
            "Formal admission still awaits a multi-seed batch; the P4.1-B budget freeze, the "
            "versioned profile-to-comparison mapping and the comparison matrix "
            "(PA re-adjudication, P4.2a/P4.2b entries and the added datasets) "
            "landed on 2026-10-25 (D40/D41). "
            "Per-run per-epoch checkpoint independence is no longer a static protocol blocker: "
            "the runner adds and discharges it from the actual checkpoint trajectories (D41)."
        ),
        "profile_identity_extension_engineering_only",
    }
)
REASSESSMENT_CHECKS = (
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "A06",
    "A07",
    "A09",
    "A10",
    "A12",
    "A14",
    "A17",
    "E01",
)

#: Optional per-batch inputs, each a path.  A check whose key is absent reports
#: ``not_run`` rather than passing, so adding a key can only add evidence.
_PATH_KEYS = ("plan", "checkpoint_inventory", "run_log", "archive_digests", "exit_code_file")

#: Directory names that hold a derived copy rather than a batch of its own.
#: A manifest under one of these would be counted twice, once as itself and once
#: as its flattened copy, so discovery refuses the whole run instead.
WORKING_COPY_DIRS = ("B3ab_merged",)


class ConfigError(ValueError):
    """A batch-root config that cannot be used as given."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Read and validate the batch-root whitelist."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {CONFIG_SCHEMA_VERSION, BATCH_SCHEMA_VERSION}:
        raise ConfigError(
            f"config schema_version must be {CONFIG_SCHEMA_VERSION!r} or {BATCH_SCHEMA_VERSION!r}, "
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
        for key in _PATH_KEYS:
            if key not in batch:
                continue
            if not isinstance(batch[key], str) or not batch[key]:
                raise ConfigError(f"batch {batch.get('name')!r} needs {key!r} as a path string")
            # Relative to the config, so a config and its inputs can move together.
            batch[key] = str((Path(path).resolve().parent / batch[key]).resolve())
    try:
        validate_recipe_config(payload, base_dir=Path(path).resolve().parent)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
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


def check_batch_evidence(config: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    """E01: every cited batch packet exists beside its run trees and still matches.

    Protocol §5 clause 6 carries the code commit and the dependency lock in one
    packet per batch (D44, D52), and a manifest only copies the packet's name and
    digest.  Nothing in the manifest's own fields can tell whether that packet
    still exists, so a results tree copied without ``batch_evidence/`` -- or a
    packet edited afterwards -- is detectable here and nowhere else.

    Its own id prefix rather than an A-number: the P4.1 delivery record closed
    that range ("不新增 A18") and this check belongs to a separate extension, the
    way the recipe audit did.

    Manifests that cite no packet are counted, not failed: everything written
    before D52 predates the packet, is not rewritten, and would otherwise turn
    every historical tree into a finding.  The count is what tells a reader how
    much of a tree cannot cite evidence for its code.
    """
    roots = {batch["name"]: Path(batch["root"]) for batch in config["batches"]}
    cited = uncited = 0
    missing: list[str] = []
    mismatched: list[str] = []
    members: list[str] = []
    for entry in entries:
        payload = entry.get("payload")
        if payload is None:
            # A missing root is A02's finding; there is nothing to cite here.
            continue
        block = payload.get("batch")
        if not isinstance(block, Mapping):
            uncited += 1
            continue
        cited += 1
        name = block.get("evidence_file")
        root = roots.get(entry["batch"])
        if not isinstance(name, str) or root is None:
            mismatched.append(f"{entry['path']}: the batch block names no packet")
            members.append(str(entry["path"]))
            continue
        packet = root / "batch_evidence" / name
        if not packet.is_file():
            missing.append(f"{entry['path']}: {name}")
            members.append(str(entry["path"]))
            continue
        if file_hash(packet) != block.get("evidence_sha256"):
            mismatched.append(f"{entry['path']}: {name}")
            members.append(str(entry["path"]))
    problems = missing + mismatched
    if problems:
        result, severity = "fail", "error"
        message = (
            f"{len(missing)} cited packet(s) missing and {len(mismatched)} no longer "
            f"matching their manifest"
        )
    elif cited:
        result, severity = "pass", "info"
        message = f"{cited} manifest(s) cite a batch packet that is present and unchanged"
    else:
        result, severity = "not_applicable", "info"
        message = "no manifest cites a batch packet; every run here predates D52"
    return make_check(
        check_id="E01",
        result=result,
        severity=severity,
        scope="global",
        message=message,
        observed={"cited": cited, "uncited": uncited},
        evidence=problems[:20],
        members=members,
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


def _plan_run_key(run: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        run.get(field)
        for field in ("dataset", "method", "training_path", "mechanism", "c_token", "seed", "view")
    )


def _manifest_run_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    """A manifest's run, spelled the way a plan spells it.

    The plan writes the PN oracle with no mechanism and no c, while its manifest
    records ``pn_oracle`` and a nominal ``c``; the translation is made here, once,
    so neither side is read literally.
    """
    unit = payload["execution_unit"]
    oracle = bool(payload.get("c_independent"))
    return (
        unit["dataset"],
        unit["method"],
        payload["training_path"],
        None if oracle else mechanism_of(payload),
        None if oracle else normalize_c_token(payload),
        payload["seed"],
        payload["run_view"],
    )


def _compare_plan(
    plan_path: str, delivered: list[dict[str, Any]], *, frozen_sha256: str
) -> tuple[list[str], list[str], set[str]]:
    """``(findings, member_paths, reason_codes)`` for one batch against its plan."""
    try:
        document = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        runs = document["runs"]
        planned = [_plan_run_key(run) for run in runs]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return [f"cannot read the plan {plan_path}: {exc}"], [], set()

    findings: list[str] = []
    reasons: set[str] = set()
    if document.get("source_protocol_sha256") != frozen_sha256:
        findings.append(
            f"plan protocol {document.get('source_protocol_sha256')!r} is not the frozen "
            f"protocol {frozen_sha256!r}"
        )
        reasons.add("superseded_protocol")
    total = (document.get("totals") or {}).get("planned")
    if total != len(planned):
        findings.append(f"totals.planned is {total!r} but the plan lists {len(planned)} run(s)")
    repeated = sorted(
        {key for key in planned if planned.count(key) > 1}, key=lambda key: tuple(map(str, key))
    )
    findings.extend(f"plan repeats run {key}" for key in repeated)

    by_key: dict[tuple[Any, ...], list[str]] = {}
    for entry in delivered:
        try:
            key = _manifest_run_key(entry["payload"])
        except (KeyError, TypeError, ValueError):
            findings.append(f"{entry['path']}: identity fields cannot be read against the plan")
            continue
        by_key.setdefault(key, []).append(str(entry["path"]))
    planned_set = set(planned)
    members: list[str] = []
    for key in sorted(planned_set - set(by_key), key=lambda key: tuple(map(str, key))):
        findings.append(f"planned but not delivered: {key}")
    for key in sorted(set(by_key) - planned_set, key=lambda key: tuple(map(str, key))):
        for path in by_key[key]:
            findings.append(f"delivered but not planned: {path} {key}")
            members.append(path)
    return findings, members, reasons


def check_plan(
    config: dict[str, Any], loaded: list[dict[str, Any]], *, frozen_sha256: str
) -> dict[str, Any]:
    """A04: each planned batch delivered exactly the runs its plan listed.

    Compared as sets of parsed runs rather than by file hash or by count: the same
    plan has been stored twice with different bytes and identical content, and a
    count cannot see a run swapped for another.  Three honest outcomes: ``fail`` if
    any configured batch disagrees with its plan; ``not_run`` while any batch has no
    plan configured, with the verified ones named so a verified half is not read as a
    verified whole; ``pass`` only when every batch was compared and agreed.
    """
    verified: dict[str, int] = {}
    unverified: list[str] = []
    findings: list[str] = []
    members: list[str] = []
    reasons: set[str] = set()
    for batch in config["batches"]:
        name = batch["name"]
        if not batch.get("plan"):
            unverified.append(name)
            continue
        delivered = [entry for entry in loaded if entry["batch"] == name]
        batch_findings, batch_members, batch_reasons = _compare_plan(
            batch["plan"], delivered, frozen_sha256=frozen_sha256
        )
        findings.extend(f"{name}: {item}" for item in batch_findings)
        members.extend(batch_members)
        reasons.update(batch_reasons)
        if not batch_findings:
            verified[name] = len(delivered)
    if findings:
        result, severity = "fail", "error"
        message = f"{len(findings)} disagreement(s) between a batch and its plan"
    elif unverified:
        result, severity = "not_run", "warning"
        message = f"no plan configured for {unverified}" + (
            f"; verified {sorted(verified)}" if verified else ""
        )
    else:
        result, severity = "pass", "info"
        message = f"every delivered run is a planned run, in {len(verified)} batch(es)"
    return make_check(
        check_id="A04",
        result=result,
        severity=severity,
        scope="batch",
        message=message,
        observed={"verified_batches": verified, "unverified_batches": sorted(unverified)},
        evidence=findings[:40],
        members=members,
        reasons=sorted(reasons),
    )


_ARROW_LINE = re.compile(r"->\s+(\S+/manifest\.json)\s*$")
_COMPLETION_LINE = re.compile(
    r"^completed (\d+) of (\d+) run\(s\)(?: in [^;]+)?; (\d+) still pending"
)
_ASSIGNED_COMPLETION_LINE = re.compile(
    r"^TRAINING_COMPLETE (\d+)/(\d+); formal admission requires audit$"
)


def _relative_key(path: Path, root: str) -> str:
    """A manifest's path under its batch root, the part a log shares with the host."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _compare_log(log_path: str, delivered: list[dict[str, Any]], root: str) -> list[str]:
    """Disagreements between one batch's run log and the manifests it delivered."""
    try:
        lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return [f"cannot read the log {log_path}: {exc}"]

    keys = {_relative_key(Path(entry["path"]), root) for entry in delivered}
    logged: Counter[str] = Counter()
    unmatched: list[str] = []
    for line in lines:
        arrow = _ARROW_LINE.search(line)
        if not arrow:
            continue
        logged_path = arrow.group(1)
        match = next((key for key in sorted(keys) if logged_path.endswith("/" + key)), None)
        if match is None:
            unmatched.append(logged_path)
        else:
            logged[match] += 1

    findings: list[str] = []
    completion = []
    for line in lines:
        if match := _COMPLETION_LINE.match(line):
            completion.append(tuple(int(value) for value in match.groups()))
        elif match := _ASSIGNED_COMPLETION_LINE.fullmatch(line):
            completion.append((*map(int, match.groups()), 0))
    if not completion:
        findings.append("the log has no completion line")
    else:
        done, total, pending = completion[-1]
        if done != total:
            findings.append(f"the completion line reports {done} of {total} run(s)")
        if pending:
            findings.append(f"the completion line reports {pending} still pending")
        if total != len(delivered):
            findings.append(
                f"the completion line totals {total} run(s) but {len(delivered)} manifest(s) "
                "were delivered"
            )
    findings.extend(
        f"run logged more than once: {key}" for key, n in sorted(logged.items()) if n > 1
    )
    findings.extend(f"logged but not delivered: {item}" for item in sorted(set(unmatched)))
    findings.extend(f"delivered but not logged: {key}" for key in sorted(keys - set(logged)))
    return findings


def _exit_code_findings(exit_code_path: str) -> list[str]:
    """Why an exit-code file cannot vouch for success; empty when it holds a single ``0``."""
    try:
        text = Path(exit_code_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"cannot read the exit code file {exit_code_path}: {exc}"]
    tokens = text.split()
    if len(tokens) != 1 or not re.fullmatch(r"-?\d+", tokens[0]):
        return [f"the exit code file is not a single integer: {text.strip()[:40]!r}"]
    if int(tokens[0]) != 0:
        return [f"the exit code is {tokens[0]}, not 0"]
    return []


def check_completion(config: dict[str, Any], loaded: list[dict[str, Any]]) -> dict[str, Any]:
    """A05: each logged batch completed, and its log names exactly the delivered runs.

    The P2.1 logs carry a closing ``completed X of Y run(s); Z still pending`` line and
    one ``-> .../manifest.json`` line per run, and **no exit code**; the runner's exit
    status sits in a separate ``<batch>_exit_code.txt``.  So the check matches each logged
    run to a manifest by its path under the batch root (the absolute prefix belongs to the
    execution host), and reads the exit code only from a configured ``exit_code_file``:
    ``0`` verifies it, anything else is a ``fail``, and a batch without the file is named
    as unverified instead of guessed.  Any disagreement is a ``fail``; a batch with no log
    is ``not_run``, and the verified batches are named so a verified half is not read as a
    verified whole.
    """
    verified: dict[str, int] = {}
    unverified: list[str] = []
    exit_unverified: list[str] = []
    findings: list[str] = []
    for batch in config["batches"]:
        name = batch["name"]
        if batch.get("exit_code_file"):
            exit_findings = _exit_code_findings(batch["exit_code_file"])
            findings.extend(f"{name}: {item}" for item in exit_findings)
            if exit_findings:
                exit_unverified.append(name)
        else:
            exit_unverified.append(name)
        if not batch.get("run_log"):
            unverified.append(name)
            continue
        delivered = [entry for entry in loaded if entry["batch"] == name]
        batch_findings = _compare_log(batch["run_log"], delivered, batch["root"])
        findings.extend(f"{name}: {item}" for item in batch_findings)
        if not batch_findings:
            verified[name] = len(delivered)
    exit_state = (
        "unverified"
        if len(exit_unverified) == len(config["batches"])
        else "partial"
        if exit_unverified
        else "verified"
    )
    caveat = (
        "the exit code was read from its file in every batch"
        if exit_state == "verified"
        else f"the exit code is unverified for {sorted(exit_unverified)}"
    )
    if findings:
        result, severity = "fail", "error"
        message = f"{len(findings)} disagreement(s) between a batch and its run log; {caveat}"
    elif unverified:
        result, severity = "not_run", "warning"
        message = f"no run log configured for {unverified}" + (
            f"; verified {sorted(verified)}" if verified else ""
        )
    else:
        result, severity = "pass", "info"
        message = (
            f"every delivered run is logged and every log reports completion, in "
            f"{len(verified)} batch(es); {caveat}"
        )
    return make_check(
        check_id="A05",
        result=result,
        severity=severity,
        scope="batch",
        message=message,
        observed={
            "verified_batches": verified,
            "unverified_batches": sorted(unverified),
            "exit_code": exit_state,
            "exit_code_unverified_batches": sorted(exit_unverified),
        },
        evidence=findings[:40],
    )


_DIGEST_LINE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")


def _sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _compare_archives(digest_path: str, batch: str) -> tuple[list[str], int, int]:
    """``(findings, archives, recomputed)`` for one batch's ``sha256sum`` file."""
    try:
        lines = Path(digest_path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return [f"cannot read the digest file {digest_path}: {exc}"], 0, 0

    findings: list[str] = []
    seen: Counter[str] = Counter()
    archives = recomputed = 0
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        match = _DIGEST_LINE.match(line)
        if not match:
            findings.append(f"line {number} is not '<sha256>  <path>'")
            continue
        digest, recorded = match.groups()
        archives += 1
        seen[recorded] += 1
        basename = re.split(r"[\\/]", recorded)[-1]
        if not basename.startswith(f"{batch}_"):
            findings.append(f"line {number}: {basename} does not belong to batch {batch}")
        local = Path(digest_path).resolve().parent / basename
        if local.is_file():
            recomputed += 1
            actual = _sha256_of(local)
            if actual != digest:
                findings.append(
                    f"line {number}: recomputed {actual} but the file records {digest} "
                    f"for {basename}"
                )
    findings.extend(
        f"archive listed more than once: {recorded}"
        for recorded, count in sorted(seen.items())
        if count > 1
    )
    if not archives and not findings:
        findings.append("the digest file lists no archive")
    return findings, archives, recomputed


def check_archives(config: dict[str, Any], loaded: list[dict[str, Any]]) -> dict[str, Any]:
    """A11: each batch's archive digest file is well formed and names its own archive.

    Only the *form* and the *name* are judged everywhere; the digest is recomputed
    where the archive happens to sit beside its digest file.  Where it does not the
    check says so, and it never says an archive was recovered -- that stays a
    reviewer's decision.  A batch with no digest file is ``not_run``.
    """
    del loaded  # the evidence is per batch and does not depend on the manifests
    verified: dict[str, dict[str, int]] = {}
    unverified: list[str] = []
    findings: list[str] = []
    for batch in config["batches"]:
        name = batch["name"]
        if not batch.get("archive_digests"):
            unverified.append(name)
            continue
        batch_findings, archives, recomputed = _compare_archives(batch["archive_digests"], name)
        findings.extend(f"{name}: {item}" for item in batch_findings)
        if not batch_findings:
            verified[name] = {"archives": archives, "recomputed": recomputed}
    total = sum(item["archives"] for item in verified.values())
    recomputed_total = sum(item["recomputed"] for item in verified.values())
    if findings:
        result, severity = "fail", "error"
        message = f"{len(findings)} problem(s) in the archive digest files"
    elif unverified:
        result, severity = "not_run", "warning"
        message = f"no archive digest file configured for {unverified}" + (
            f"; verified {sorted(verified)}" if verified else ""
        )
    else:
        result, severity = "pass", "info"
        message = (
            f"digest files well formed and archives named for their batch in {len(verified)} "
            f"batch(es); {total - recomputed_total} archive(s) not recomputed on this host; "
            "recovery is not tested"
        )
    return make_check(
        check_id="A11",
        result=result,
        severity=severity,
        scope="batch",
        message=message,
        observed={"verified_batches": verified, "unverified_batches": sorted(unverified)},
        evidence=findings[:40],
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
                (
                    normalize_c_token(payload),
                    "pa" if name == "PA+clean-support" else matrix_selection_protocol(name),
                    int(payload["seed"]),
                )
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


def load_checkpoint_inventory(path: str | Path) -> set[str]:
    """The checkpoint paths a batch's ``checkpoint_inventory.tsv`` lists as on disk.

    One ``<path>\\t<bytes>`` line per file left after reclamation.  Only the path is
    read; the size is not evidence of anything the reclaim accounting claims.
    """
    paths: set[str] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise ConfigError(f"{path}: line {number} is not '<path><TAB><bytes>'")
        paths.add(fields[0])
    return paths


def _inventory_findings(refs: list[dict[str, Any]], inventory: set[str], *, path: str) -> list[str]:
    """File-by-file disagreements between a manifest's references and an inventory."""
    findings = []
    for ref in refs:
        recorded = str(ref.get("path"))
        present = recorded in inventory
        if ref.get("reclaimed") and present:
            findings.append(f"{path}: {recorded} is marked reclaimed but still on disk")
        elif not ref.get("reclaimed") and not present:
            findings.append(f"{path}: {recorded} is marked kept but missing from the inventory")
    return findings


def check_reclaim(
    loaded: list[dict[str, Any]], *, inventories: dict[str, set[str]] | None = None
) -> dict[str, Any]:
    """A10: ``reclaimed_true + files_on_disk == refs``, run by run.

    Only batches that actually reclaimed can be judged, so a tree whose manifests
    carry no ``reclaimed`` field at all is ``not_applicable`` rather than ``fail``
    (B1-B3b predate the field) or ``pass`` (nothing was checked).  The count is per
    manifest: one run's references cannot be reconciled against a whole result tree's
    file count.  Where the recorded checkpoint paths do not resolve on this host the
    count is refused rather than guessed, because a local success against a foreign
    path would be an accident.

    A batch may instead come with its ``checkpoint_inventory.tsv`` (*inventories*,
    keyed by batch name), which stands in for the directory listing.  It is judged
    file by file -- a reclaimed reference must be absent, a kept one present, and an
    inventory file no manifest references is an orphan -- because a lost file and a
    stray file would otherwise cancel in a count.
    """
    applicable = {str(entry["path"]): _reclaim_refs(entry["payload"]) for entry in loaded}
    applicable = {path: refs for path, refs in applicable.items() if refs}
    batch_of = {str(entry["path"]): entry["batch"] for entry in loaded}
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
    referenced: dict[str, set[str]] = {}
    for path, refs in sorted(applicable.items()):
        inventory = (inventories or {}).get(batch_of[path])
        if inventory is not None:
            referenced.setdefault(batch_of[path], set()).update(
                str(ref.get("path")) for ref in refs
            )
            mismatched.extend(_inventory_findings(refs, inventory, path=path))
            continue
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
    orphans = [
        f"{batch}: {recorded} is in the inventory but no manifest references it (orphan)"
        for batch, listed in sorted((inventories or {}).items())
        if batch in referenced
        for recorded in sorted(listed - referenced[batch])
    ]
    # A fault found outranks a gap: a batch that could not be judged must not hide
    # one that was judged and does not balance.
    if mismatched or orphans:
        unbalanced = len({item.split(": ", 1)[0] for item in mismatched})
        observed = {"manifests_with_reclaim": len(applicable), "unbalanced": unbalanced}
        if inventories:
            observed["orphans"] = len(orphans)
        return make_check(
            check_id="A10",
            result="fail",
            severity="error",
            scope="group",
            message=f"{unbalanced} manifest(s) do not balance; {len(orphans)} orphan file(s)",
            observed=observed,
            evidence=(mismatched + orphans)[:20],
            members=sorted(applicable),
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
    observed = {"manifests_with_reclaim": len(applicable), "unbalanced": 0}
    if inventories:
        observed["orphans"] = 0
    return make_check(
        check_id="A10",
        result="pass",
        severity="info",
        scope="group",
        message=(
            f"reclaimed + on disk balances against references in all {len(applicable)} manifest(s)"
        ),
        observed=observed,
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
        #: Filled by :func:`build_audit`; ``None`` only for a report built by hand.
        "provenance": None,
        "batches": [],
        "coverage": {},
        "checks": [],
        "overall": "pass",
    }


REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERY_DIR = "docs/research/pu_survey/delivery"

#: The stage snapshot that carries each batch's boundary wording.  B3a and B3b share
#: one because the adapter group is analysed as a whole.
SNAPSHOT_FILES = {
    "B1": "p2_1_b1_snapshot.md",
    "B2": "p2_1_b2_snapshot.md",
    "B3a": "p2_1_b3a_snapshot.md",
    "B3b": "p2_1_b3ab_snapshot.md",
    "B4": "p2_1_b4_snapshot.md",
}
ABLATION_SNAPSHOT = SNAPSHOT_FILES["B3b"]

#: Tests that keep the test set out of selection: a test view cannot be handed to a
#: selection protocol, and PA never sees clean labels.
ISOLATION_TESTS = (
    ("tests/unit/experiment/test_bundle.py", "test_edge_test_for_selection_false"),
    (
        "tests/unit/experiment/test_strategies_selection.py",
        "test_param_protocolpa_rejects_clean_view",
    ),
    ("tests/unit/experiment/test_runner.py", "test_pa_never_receives_clean_labels"),
)

_BOUNDARY_HEADING = re.compile(r"^#+ .*本快照不含", re.MULTILINE)


def _read_text(path: Path) -> tuple[str | None, str | None]:
    """``(text, problem)``: exactly one of the two is ``None``."""
    try:
        return path.read_text(encoding="utf-8"), None
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"cannot read {path.name}: {exc}"


def _documentation_check(
    check_id: str, findings: list[str], *, ok_message: str, fail_message: str, observed: dict
) -> dict[str, Any]:
    failed = bool(findings)
    return make_check(
        check_id=check_id,
        result="fail" if failed else "pass",
        severity="error" if failed else "info",
        scope="global",
        message=fail_message if failed else ok_message,
        observed=observed,
        evidence=findings[:40],
    )


def _packet_snapshot_problem(batch: dict[str, Any], entries: list[dict[str, Any]]) -> str | None:
    """A new batch replaces a legacy stage document only with verified D52 packets."""
    members = [entry for entry in entries if entry.get("batch") == batch["name"]]
    if not members:
        return "no stage snapshot or cited batch evidence is available"
    for entry in members:
        payload = entry.get("payload") or {}
        block = payload.get("batch")
        if not isinstance(block, Mapping):
            return "every new-batch manifest must cite its evidence packet"
        name = block.get("evidence_file")
        if not isinstance(name, str) or Path(name).name != name:
            return "the cited evidence packet must be a filename"
        path = Path(batch["root"]) / batch_evidence.DIRECTORY / name
        try:
            packet = batch_evidence.load(path)
            if batch_evidence.manifest_block(packet, path) != block:
                return "the evidence citation no longer matches its packet"
            unit = payload["execution_unit"]
            oracle = bool(payload.get("c_independent"))
            batch_evidence.require_matches(
                packet,
                unit={
                    "dataset": unit["dataset"],
                    "method": unit["method"],
                    "training_path": payload["training_path"],
                    "mechanism": None if oracle else mechanism_of(payload),
                },
                seeds=packet["seeds"],
                c_tokens=packet["c_tokens"],
                protocol_sha256=payload["protocol_sha256"],
            )
            if payload["seed"] not in packet["seeds"] or (
                not oracle and normalize_c_token(payload) not in packet["c_tokens"]
            ):
                return "the evidence packet does not cover this manifest's seed or c"
            if packet["protocol_version"] != payload["protocol_version"]:
                return "the evidence packet names another protocol version"
        except (KeyError, OSError, ValueError, TypeError) as exc:
            return f"cannot verify cited batch evidence: {exc}"
    return None


def check_snapshots(
    config: dict[str, Any], repo_root: Path, *, entries: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """A14: legacy batches keep stage snapshots; new batches cite verified D52 evidence.

    Only the file and the "本快照不含" heading are looked for; whether the wording under
    it is right, or the snapshot final (B4's title still says draft), is a reviewer's call.
    A new batch has no historical stage document, so every delivered manifest
    must instead cite an unchanged content-addressed packet describing that run.
    """
    findings: list[str] = []
    drafts: list[str] = []
    found: list[str] = []
    packet_batches: list[str] = []
    for batch in config["batches"]:
        name = batch["name"]
        filename = SNAPSHOT_FILES.get(name)
        if filename is None:
            problem = _packet_snapshot_problem(batch, entries or [])
            if problem:
                findings.append(f"{name}: {problem}")
            else:
                packet_batches.append(name)
            continue
        text, problem = _read_text(repo_root / DELIVERY_DIR / filename)
        if problem:
            findings.append(f"{name}: {problem}")
        elif not _BOUNDARY_HEADING.search(text):
            findings.append(f"{name}: {filename} has no '本快照不含' section")
        else:
            found.append(name)
            if "草稿" in text.splitlines()[0]:
                drafts.append(name)
    return _documentation_check(
        "A14",
        findings,
        ok_message=(
            f"{len(found)} batch snapshot(s) present with their boundary section; "
            f"{len(packet_batches)} new batch(es) carry verified D52 evidence; "
            "the wording itself is not judged"
            + (f"; still titled draft: {drafts}" if drafts else "")
        ),
        fail_message=f"{len(findings)} snapshot problem(s)",
        observed={
            "snapshots_found": sorted(found),
            "titled_draft": sorted(drafts),
            "packet_batches": sorted(packet_batches),
        },
    )


def check_ablation_disclosure(loaded: list[dict[str, Any]], repo_root: Path) -> dict[str, Any]:
    """A16: a delivered ``self_pu`` run is disclosed as an ablation variant without review.

    Both halves are required: the variant is named, and the snapshot says the
    methodological review of it was not obtained.  Not applicable when no ``self_pu``
    run is delivered.
    """
    if not any(entry["payload"]["execution_unit"]["method"] == "self_pu" for entry in loaded):
        return make_check(
            check_id="A16",
            result="not_applicable",
            severity="info",
            scope="global",
            message="no self_pu run is delivered, so there is no ablation to disclose",
        )
    text, problem = _read_text(repo_root / DELIVERY_DIR / ABLATION_SNAPSHOT)
    findings: list[str] = []
    if problem:
        findings.append(problem)
    else:
        if "消融变体" not in text:
            findings.append(f"{ABLATION_SNAPSHOT} does not name the ablation variant")
        if not any("未获" in line and "复核" in line for line in text.splitlines()):
            findings.append(f"{ABLATION_SNAPSHOT} does not say the review was not obtained")
    return _documentation_check(
        "A16",
        findings,
        ok_message=(
            "the snapshot names the ablation variant and says its review was not obtained; "
            "that the disclosure is accurate is not judged"
        ),
        fail_message=f"{len(findings)} gap(s) in the self_pu disclosure",
        observed={"snapshot": ABLATION_SNAPSHOT},
    )


def check_isolation_tests(repo_root: Path) -> dict[str, Any]:
    """A17: the tests that keep the test set out of selection are present.

    The audit does not run them: that they pass is the test suite's verdict.
    """
    findings: list[str] = []
    for relative, test_name in ISOLATION_TESTS:
        text, problem = _read_text(repo_root / relative)
        if problem:
            findings.append(f"{relative}: {problem}")
        elif f"def {test_name}(" not in text:
            findings.append(f"{relative}: no test named {test_name}")
    return _documentation_check(
        "A17",
        findings,
        ok_message=(
            f"{len(ISOLATION_TESTS)} isolation guard test(s) present; "
            "whether they pass is the test suite's verdict, not this audit's"
        ),
        fail_message=f"{len(findings)} isolation guard test(s) missing",
        observed={"tests": [f"{path}::{name}" for path, name in ISOLATION_TESTS]},
    )


def _checkpoint_reassessment_problems(payload: dict[str, Any], protocol: dict) -> list[str]:
    """Require every frozen epoch and component, including warmup checkpoints."""
    profile_id = payload.get("profile_id")
    if profile_id in {"nnpu", "dist_pu"}:
        return _neural_checkpoint_reassessment_problems(payload, protocol)
    if profile_id not in {"lagam", "pulns", "self_pu_full_meta"}:
        return ["profile_outside_reassessment_scope"]
    profile = protocol["method_profiles"][profile_id]
    if profile_id == "self_pu_full_meta" and (
        payload.get("run_view") == "ts-compatible"
        or payload.get("extra_supervision", {}).get("training_view") == "ts"
    ):
        return ["full_meta_ts_adapter_not_admitted"]
    budget = protocol["budgets"][profile["budget"]]
    recorded_params = payload.get("estimator_parameters", {})
    if any(recorded_params.get(key) != value for key, value in profile["params"].items()):
        return ["estimator_parameters_not_frozen"]
    candidates = payload.get("candidate_runs", [])
    if payload.get("selection_checkpoint_scope") != "independent_per_epoch":
        return ["checkpoint_scope_not_independent"]
    if len(candidates) != budget["outer_candidates"]:
        return ["candidate_pool_incomplete"]
    if any(payload.get("budget", {}).get(key) != value for key, value in budget.items()):
        return ["frozen_budget_mismatch"]
    count = budget["epochs"]
    components = {"teacher_1", "teacher_2"} if profile_id == "self_pu_full_meta" else {"model"}
    start = 1 if profile_id == "self_pu_full_meta" else 0
    expected = {
        (label, component) for label in range(start, start + count) for component in components
    }
    for candidate in candidates:
        params = candidate.get("params")
        if candidate.get("status") != "succeeded" or not isinstance(params, dict):
            return ["candidate_not_succeeded"]
        if any(profile["params"].get(key) != value for key, value in params.items()):
            return ["candidate_params_not_frozen"]
        refs = candidate.get("epoch_checkpoints", [])
        observed = {(ref.get("epoch_label"), ref.get("component")) for ref in refs}
        if len(refs) != len(expected) or observed != expected:
            return ["checkpoint_trajectory_incomplete"]
        provenance_key = "captured_episodes" if profile_id == "pulns" else "captured_epochs"
        captured = candidate.get("training_provenance", {}).get(provenance_key, [])
        if captured != list(range(start, start + count)):
            return ["captured_epoch_provenance_incomplete"]
        if any(
            not {"OA", "PA+clean-support"} <= set(ref.get("validation_metrics", {})) for ref in refs
        ):
            return ["checkpoint_validation_incomplete"]
    return []


def _neural_checkpoint_reassessment_problems(payload: dict[str, Any], protocol: dict) -> list[str]:
    """Verify the frozen native-2d nnPU/Dist-PU trajectory and selection evidence."""
    if payload.get("training_path") != "native_2d":
        return ["profile_outside_reassessment_scope"]
    profile_id = payload["profile_id"]
    profile = protocol["method_profiles"][profile_id]
    budget = protocol["budgets"][profile["budget"]]
    params = payload.get("estimator_parameters", {})
    if any(params.get(key) != value for key, value in profile["params"].items()):
        return ["estimator_parameters_not_frozen"]
    if profile_id == "dist_pu" and params.get("training_recipe", "clean_room") != "clean_room":
        return ["estimator_parameters_not_frozen"]
    if any(payload.get("budget", {}).get(key) != value for key, value in budget.items()):
        return ["frozen_budget_mismatch"]
    candidates = payload.get("candidate_runs", [])
    pool = protocol["candidate_pool"]
    if len(candidates) != budget["outer_candidates"] or len(candidates) != len(pool):
        return ["candidate_pool_incomplete"]
    if payload.get("selection_checkpoint_scope") != "independent_per_epoch":
        return ["checkpoint_scope_not_independent"]
    count = budget["epochs"]
    expected = {(epoch + 1, epoch, "model") for epoch in range(count)}
    required_metrics = {"OA"} if mechanism_of(payload) != "scar" else {"PA", "OA"}
    for candidate, frozen_params in zip(candidates, pool, strict=True):
        if candidate.get("status") != "succeeded":
            return ["candidate_not_succeeded"]
        if candidate.get("params") != frozen_params:
            return ["candidate_params_not_frozen"]
        refs = candidate.get("epoch_checkpoints", [])
        observed = {
            (ref.get("epoch_position"), ref.get("epoch_label"), ref.get("component"))
            for ref in refs
        }
        if len(refs) != count or observed != expected:
            return ["checkpoint_trajectory_incomplete"]
        if any(not required_metrics <= set(ref.get("validation_metrics", {})) for ref in refs):
            return ["checkpoint_validation_incomplete"]
        if profile_id == "dist_pu":
            provenance = candidate.get("training_provenance", {})
            if (
                provenance.get("implementation_path") != "clean_room_single_stage_full_batch"
                or provenance.get("epochs_completed") != count
                or provenance.get("optimizer_steps") != count
            ):
                return ["training_budget_provenance_incomplete"]
    return []


def _partitioned_grid_problems(
    entries: list[dict[str, Any]], *, protocol: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    """Prove a single execution unit's complete grid and its batch identities."""
    if not entries:
        return ["partition_grid_empty"]
    roots = {item["name"]: Path(item["root"]) for item in config.get("batches", [])}
    problems: set[str] = set()
    identities = []
    cells = []
    first = entries[0]["payload"]
    unit = first["execution_unit"]
    mechanism = mechanism_of(first)
    expected_unit = {
        "dataset": unit["dataset"],
        "method": unit["method"],
        "training_path": first["training_path"],
        "mechanism": mechanism,
    }
    expected_seeds = set(protocol["seeds"])
    expected_tokens = set(protocol["c_tokens"].get(mechanism, []))
    if not expected_seeds or not expected_tokens:
        return ["partition_grid_unsupported"]
    expected = {(seed, token) for seed in expected_seeds for token in expected_tokens}
    for entry in entries:
        payload = entry["payload"]
        try:
            cell = (payload["seed"], normalize_c_token(payload))
            cells.append(cell)
            if payload["execution_unit"] != unit or mechanism_of(payload) != mechanism:
                problems.add("partition_unit_mismatch")
            if payload.get("run_view") != first.get("run_view"):
                problems.add("partition_view_mismatch")
            if not set(payload.get("protocol_deviation", [])) <= {"seed_subset", "c_grid"}:
                problems.add("partition_other_deviation")
            block = payload["batch"]
            name = block["evidence_file"]
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("invalid packet name")
            path = roots[entry["batch"]] / batch_evidence.DIRECTORY / name
            packet = batch_evidence.load(path)
            if batch_evidence.manifest_block(packet, path) != block:
                raise ValueError("packet citation mismatch")
            batch_evidence.require_matches(
                packet,
                unit=expected_unit,
                seeds=packet["seeds"],
                c_tokens=packet["c_tokens"],
                protocol_sha256=digest(protocol),
            )
            seeds, tokens = packet["seeds"], packet["c_tokens"]
            if (
                not seeds
                or len(seeds) != len(set(seeds))
                or not set(seeds) <= expected_seeds
                or not tokens
                or len(tokens) != len(set(tokens))
                or not set(tokens) <= expected_tokens
                or cell[0] not in seeds
                or cell[1] not in tokens
                or packet["run_count"] != len(seeds) * len(tokens)
                or packet["protocol_version"] != protocol["protocol_version"]
            ):
                raise ValueError("packet grid mismatch")
            expected_deviations = set()
            if seeds != protocol["seeds"]:
                expected_deviations.add("seed_subset")
            if tokens != protocol["c_tokens"][mechanism]:
                expected_deviations.add("c_grid")
            if set(payload.get("protocol_deviation", [])) != expected_deviations:
                raise ValueError("recorded partition deviations disagree with the packet")
            if expected_deviations and "protocol_deviation" not in payload.get(
                "formal_blockers", []
            ):
                raise ValueError("partition deviation has no recorded blocker")
            env = packet["environment"]
            hardware = env["hardware"]
            threads = env["thread_environment"]
            required_env = (
                "python_version",
                "python_implementation",
                "platform",
                "torch_version",
                "cuda_available",
                "cuda_runtime_version",
                "cudnn_version",
                "gpu_devices",
                "gpu_driver_version",
                "numpy_version",
                "scikit_learn_version",
            )
            if (
                env.get("state") != "recorded"
                or any(k not in env for k in required_env)
                or not hardware.get("machine")
                or not hardware.get("processor")
                or not hardware.get("cpu_count")
                or not all(
                    env.get(key)
                    for key in (
                        "python_version",
                        "python_implementation",
                        "platform",
                        "numpy_version",
                        "scikit_learn_version",
                    )
                )
                or (
                    env.get("cuda_available")
                    and (
                        not env.get("gpu_devices")
                        or not env.get("gpu_driver_version")
                        or not env.get("cuda_runtime_version")
                    )
                )
                or set(threads) != {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
                or packet["code_commit_state"] != "recorded"
                or not packet["code_commit"]
                or packet["code_commit_dirty"] is not False
                or packet["dependency_lock"].get("state") != "recorded"
                or not packet["dependency_lock"].get("sha256")
            ):
                raise ValueError("incomplete runtime identity")
            identities.append(
                (
                    packet["code_commit"],
                    packet["dependency_lock"],
                    env,
                    packet.get("recipe_registry"),
                )
            )
        except (KeyError, TypeError, ValueError, OSError):
            problems.add("partition_batch_evidence_invalid")
    if len(cells) != len(expected) or set(cells) != expected:
        problems.add("partition_grid_incomplete_or_duplicate")
    if (
        not identities
        or len(identities) != len(entries)
        or any(identity != identities[0] for identity in identities[1:])
    ):
        problems.add("partition_runtime_identity_mismatch")
    return sorted(problems)


def reassess_admission(
    entries: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    checks: list[dict[str, Any]],
    fairness_passed: bool = True,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-adjudicate exact historical blockers without changing recorded evidence.

    Registry validation here is posthoc protocol/budget verification. It never
    claims that a legacy run had a runtime recipe binding.
    """
    canonical = load_protocol(resolve_protocol_path("survey-v3.0"))
    canonical_sha = digest(canonical)
    results = {item["check_id"]: item["result"] for item in checks}
    problems = [
        f"audit_{name}_not_passed" for name in REASSESSMENT_CHECKS if results.get(name) != "pass"
    ]
    if any(item.get("result") == "fail" for item in checks) or not fairness_passed:
        problems.append("audit_or_fairness_failed")
    if digest(protocol) != canonical_sha or protocol.get("review_status") != "accepted":
        problems.append("protocol_not_canonical_accepted_v3")
    registry_path = REPO_ROOT / "pu_toolbox/experiment/survey_recipe_registry_v1.json"
    registry = load_registry(registry_path)
    if registry.get("recipe_lifecycle") != "locked" or any(
        item.severity == "error" for item in validate_registry(registry, canonical)
    ):
        problems.append("locked_registry_invalid")
    try:
        if check_result_completeness(entries, protocol=canonical)["result"] != "pass":
            problems.append("five_seed_c_grid_incomplete")
    except (KeyError, TypeError, ValueError):
        problems.append("five_seed_c_grid_invalid")
    partition_problems: dict[tuple, list[str]] = {}
    partition_groups: dict[tuple, list[dict[str, Any]]] = {}
    for entry in entries:
        payload = entry["payload"]
        unit = payload["execution_unit"]
        key = (
            unit["dataset"],
            unit["method"],
            payload["training_path"],
            mechanism_of(payload),
            payload.get("run_view"),
        )
        partition_groups.setdefault(key, []).append(entry)
    for key, members in partition_groups.items():
        if any(member["payload"].get("protocol_deviation") for member in members):
            partition_problems[key] = (
                _partitioned_grid_problems(members, protocol=canonical, config=config)
                if config is not None
                else ["partition_config_missing"]
            )
    per_run: dict[str, list[str]] = {}
    for entry in entries:
        payload = entry["payload"]
        local = []
        profile_id = payload.get("profile_id")
        unit = payload["execution_unit"]
        partition_key = (
            unit["dataset"],
            unit["method"],
            payload["training_path"],
            mechanism_of(payload),
            payload.get("run_view"),
        )
        local.extend(partition_problems.get(partition_key, []))
        if (
            not valid_manifest_profile_identity(payload)
            or payload.get("protocol_sha256") != canonical_sha
            or unit not in canonical["execution_units"]
        ):
            local.append("manifest_protocol_identity_or_deviation")
        try:
            registration = profile_registration_status(profile_id, protocol=canonical)
            if registration["comparison_registration_status"] == "not_in_matrix" or any(
                payload.get(key) != value for key, value in registration.items()
            ):
                local.append("profile_registration_mismatch")
            recipe = registry["profiles"][profile_id]
            if recipe.get("recipe_lifecycle") != "locked":
                local.append("profile_recipe_not_locked")
            local.extend(_checkpoint_reassessment_problems(payload, canonical))
        except (KeyError, TypeError, ValueError):
            local.append("profile_or_checkpoint_evidence_invalid")
        per_run[str(entry["path"])] = local
    # A completed row needs every member's trajectory, not only its own seed.
    incomplete_groups = {
        group_key(entry["payload"]) for entry in entries if per_run[str(entry["path"])]
    }
    runs = {}
    for entry in entries:
        path = str(entry["path"])
        payload = entry["payload"]
        unit = payload["execution_unit"]
        partition_key = (
            unit["dataset"],
            unit["method"],
            payload["training_path"],
            mechanism_of(payload),
            payload.get("run_view"),
        )
        recorded = list(payload.get("formal_blockers", []))
        reasons = sorted(set(problems + per_run[path]))
        if group_key(payload) in incomplete_groups:
            reasons.append("row_member_evidence_incomplete")
        discharged = [
            value
            for value in recorded
            if not reasons
            and (
                value in _REASSESSABLE_BLOCKERS
                or (
                    value == "protocol_deviation"
                    and payload.get("protocol_deviation")
                    and partition_key in partition_problems
                )
            )
        ]
        remaining = [value for value in recorded if value not in discharged]
        source_sha = file_hash(Path(path)) if Path(path).is_file() else None
        if source_sha is None:
            reasons.append("manifest_source_bytes_missing")
            discharged, remaining = [], recorded
        runs[path] = {
            "manifest_sha256": source_sha,
            "recorded_formal_eligible": payload.get("formal_eligible"),
            "recorded_blockers": recorded,
            "discharged_blockers": discharged,
            "remaining_blockers": remaining,
            "eligible": not remaining and not reasons,
            "requirements_not_met": reasons,
            "partition_grid_checked": partition_key in partition_problems,
            "registry_verification": "posthoc_canonical_profile_and_budget",
            "runtime_recipe_binding": payload.get("recipe_binding_status", "legacy_unbound"),
        }
    return {
        "schema_version": "survey-admission-reassessment-1",
        "protocol_version": canonical["protocol_version"],
        "protocol_sha256": canonical_sha,
        "recipe_registry_sha256": file_hash(registry_path),
        "required_checks": list(REASSESSMENT_CHECKS),
        "runs": runs,
    }


def verify_support_receipts(config: dict, entries: list[dict]) -> dict[str, bool]:
    """Re-read the five-role products without asserting upstream source replay."""
    from pu_toolbox.experiment.support_data import load_support_dataset

    roots = {}
    for batch in config["batches"]:
        if batch.get("plan"):
            try:
                plan = json.loads(Path(batch["plan"]).read_text(encoding="utf-8"))
                if plan.get("support_root"):
                    roots[batch["name"]] = Path(plan["support_root"])
            except (OSError, ValueError):
                pass
    cache = {}
    verified = {}
    for entry in entries:
        payload = entry["payload"]
        unit = payload.get("execution_unit", {})
        root = roots.get(entry["batch"])
        key = (root, unit.get("dataset"), payload.get("seed"))
        if key not in cache:
            try:
                if root is None:
                    raise ValueError("no support root in the audited plan")
                _, _, receipt = load_support_dataset(root / str(key[1]) / f"split_{key[2]}")
                cache[key] = receipt
            except (OSError, ValueError, KeyError, TypeError):
                cache[key] = None
        verified[str(entry["path"])] = cache[key] is not None and cache[key] == payload.get(
            "split_ref"
        )
    return verified


def build_audit(
    config: dict[str, Any],
    *,
    entries: list[dict[str, Any]] | None = None,
    protocol: dict[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run every check this entry point can decide, and name the ones it cannot.

    ``entries`` is accepted so a caller that also summarizes does not walk and
    load the tree twice; the audit is a pure function of what it is handed.
    ``protocol`` selects the matrix the tree was run under, so a v3 tree is
    audited against v5 instead of being failed against the v2 default.
    """
    if protocol is None:
        protocol = load_protocol()
    frozen = digest(protocol)
    report = empty_report()
    report["protocol_version"] = protocol.get("protocol_version")
    report["protocol_sha256"] = frozen
    # Stamped before discovery so the refusal path below carries the same
    # identity a successful audit does: the protocol and the roots are known
    # whether or not the tree can be walked.  The comparison fields are added
    # further down, once that matrix is loaded.
    report["provenance"] = build_provenance(
        protocol_version=protocol.get("protocol_version"),
        protocol_sha256=frozen,
        result_roots=input_result_roots(config),
        source_roots=recorded_source_roots(config),
    )

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
    recipe_audit = audit_recipe_batches(config, entries, protocol)
    if recipe_audit is not None:
        report["recipe_audit"] = recipe_audit
        if recipe_audit["overall"] == "fail":
            # This is an independent, versioned admission extension, not A18 or
            # a changed meaning of A01. Do not call numeric gates on bad inputs.
            report["overall"] = "fail"
            return report

    inventories = {
        batch["name"]: load_checkpoint_inventory(batch["checkpoint_inventory"])
        for batch in config["batches"]
        if batch.get("checkpoint_inventory")
    }
    checks = [
        check_whitelist(config, entries),
        check_coverage(config, entries),
        check_protocol(loaded, frozen_sha256=frozen),
        check_plan(config, loaded, frozen_sha256=frozen),
        check_completion(config, loaded),
        check_archives(config, loaded),
    ]

    # Phase A: per-manifest structure, grouped so one check names every offender.
    # The frozen matrix is loaded once and handed to the preflight: the loader
    # re-validates every mapping and anchor, which is too much work to repeat 610
    # times for the same answer.
    comparison = comparison_for_version(protocol["protocol_version"])
    # The matrix is part of this report's identity: the preflight verdicts below
    # depend on it, so which matrix produced them belongs next to the protocol.
    report["provenance"] = build_provenance(
        protocol_version=protocol.get("protocol_version"),
        protocol_sha256=frozen,
        comparison_version=comparison.get("comparison_version"),
        comparison_sha256=comparison_digest(comparison),
        result_roots=input_result_roots(config),
        source_roots=recorded_source_roots(config),
    )
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
            check_batch_evidence(config, loaded),
            check_view_mechanism(loaded),
            check_result_completeness(loaded, protocol=protocol),
            check_reclaim(loaded, inventories=inventories),
            check_status_labels(loaded),
            check_probe_separation(loaded, protocol=protocol),
            check_snapshots(config, repo_root, entries=loaded),
            check_ablation_disclosure(loaded, repo_root),
            check_isolation_tests(repo_root),
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
            kwargs = {}
            if recipe_audit is not None and recipe_audit["method_bindings"]:
                kwargs["expected_recipe_bindings"] = recipe_audit["method_bindings"]
            outcome = gate.aggregate(paths, protocol=protocol, require_formal=False, **kwargs)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            failures.append((paths, str(exc)))
            continue
        refused_total += len(outcome["refused"])
    reassessment = reassess_admission(
        loaded,
        protocol=protocol,
        checks=checks,
        fairness_passed=not failures and not refused_total,
        config=config,
    )
    receipts = verify_support_receipts(config, loaded)
    for path, decision in reassessment["runs"].items():
        decision["support_receipt_verified"] = receipts[path]
    report["admission_reassessment"] = reassessment
    for members in buckets.values():
        if any(
            (decision := reassessment["runs"][str(entry["path"])])["remaining_blockers"]
            or (decision["partition_grid_checked"] and not decision["eligible"])
            for entry in members
        ):
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

    report["checks"] = checks
    report["overall"] = overall_status(checks)
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P2.1 批次审计",
        "",
        f"- 协议：`{report['protocol_version']}` / `{report['protocol_sha256']}`",
    ]
    # Identity before findings: a reader has to know which checkout and input
    # tree this audit is about before the verdict means anything.
    lines.extend(render_provenance_lines(report.get("provenance")))
    lines.extend(
        [
            f"- 制品：{report['coverage'].get('total', 0)} 份 manifest",
            f"- 结论：**{report['overall']}**",
            "",
            "| check | 结果 | 作用域 | reasons | 说明 |",
            "|---|---|---|---|---|",
        ]
    )
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
    lines.extend(recipe_audit_lines(report))
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
    parser.add_argument(
        "--protocol",
        default="survey-v3.0",
        help="protocol matrix the tree was run under (shipped name or path)",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        report = build_audit(config, protocol=load_protocol(resolve_protocol_path(args.protocol)))
    except (AuditError, ConfigError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    # The only reading that needs a process: kept out of build_audit so two
    # audits over one tree stay equal.
    commit, dirty = resolve_code_commit()
    report["provenance"] = with_code_commit(report.get("provenance"), commit, dirty=dirty)
    written = write_report(report, args.out_dir)
    for path in written:
        print(f"wrote {path}")
    print(f"overall: {report['overall']}")
    return 0 if report["overall"] != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
