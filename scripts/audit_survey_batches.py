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
    SCHEMA_VERSION as AUDIT_SCHEMA_VERSION,
)
from pu_toolbox.experiment.survey_audit import (  # noqa: E402
    AuditError,
    make_check,
    overall_status,
    preflight_status,
    rollup_gate_check,
)
from pu_toolbox.experiment.survey_protocol import digest, load_protocol  # noqa: E402
from pu_toolbox.experiment.survey_summary import group_key  # noqa: E402

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

    checks = [check_coverage(config, entries), check_protocol(loaded, frozen_sha256=frozen)]

    # Phase A: per-manifest structure, grouped so one check names every offender.
    unusable: list[str] = []
    for entry in loaded:
        status, reasons, missing = preflight_status(entry["payload"])
        if status == "not_reproducible":
            entry["status"] = status
            entry["reasons"] = reasons
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
            members=[str(entry["path"]) for entry in loaded if entry.get("status")],
            reasons=entry_reasons,
        )
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
