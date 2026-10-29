"""Read-only support trajectory repair planning; never grants batch admission."""

from __future__ import annotations

from collections import Counter, defaultdict

from .pilot_plan import manifest_identity, planned_runs
from .survey_protocol import digest

SUPPORT_PROFILES = frozenset({"lagam", "pulns", "self_pu_full_meta"})


def plan_support_repairs(protocol, entries, *, checkpoint_check):
    """Reuse the official auditor's trajectory predicate, not a second standard.

    Entries are immutable source snapshots with ``path``, ``payload`` and
    ``manifest_sha256``. Complete trajectories are NOT formal approvals. A
    duplicate identity is unresolved, never silently replaced by the latest.
    """
    expected = {r.key: r for r in planned_runs(protocol) if r.method in SUPPORT_PROFILES}
    grouped = defaultdict(list)
    excluded = []
    for entry in entries:
        payload = entry["payload"]
        if payload.get("profile_id") not in SUPPORT_PROFILES:
            continue
        key = manifest_identity(payload)
        if key is None or payload.get("failures") or not payload.get("test_results"):
            excluded.append({"path": str(entry["path"]), "reason": "not_a_completed_support_fit"})
            continue
        if key not in expected or payload.get("protocol_sha256") != digest(protocol):
            excluded.append({"path": str(entry["path"]), "reason": "outside_bound_support_plan"})
            continue
        if not entry.get("manifest_sha256"):
            raise ValueError("every source entry requires its manifest digest")
        grouped[key].append(entry)
    rows = []
    for key in sorted(expected, key=str):
        dataset, method, path, mechanism, token, seed = key
        row = dict(
            dataset=dataset,
            profile=method,
            training_path=path,
            mechanism=mechanism,
            c_token=token,
            seed=seed,
            fit_identity=list(key),
            formal_admission=False,
        )
        matches = grouped.get(key, [])
        if not matches:
            row.update(status="missing", findings=["missing_completed_fit"], sources=[])
        elif len(matches) > 1:
            row.update(
                status="conflict",
                findings=["multiple_source_attempts_require_reconciliation"],
                sources=[{"path": str(e["path"]), "sha256": e["manifest_sha256"]} for e in matches],
            )
        else:
            entry = matches[0]
            m = entry["payload"]
            findings = checkpoint_check(m, protocol)
            profile = protocol["method_profiles"][method]
            budget = protocol["budgets"][profile["budget"]]
            start = 1 if method == "self_pu_full_meta" else 0
            components = ("teacher_1", "teacher_2") if method == "self_pu_full_meta" else ("model",)
            wanted = {
                (epoch, component)
                for epoch in range(start, start + budget["epochs"])
                for component in components
            }
            refs = [
                ref for c in m.get("candidate_runs", []) for ref in c.get("epoch_checkpoints", [])
            ]
            observed = {(ref.get("epoch_label"), ref.get("component")) for ref in refs}
            missing = sorted(wanted - observed)
            row.update(
                status="rerun_required" if findings else "trajectory_complete",
                findings=findings,
                expected_checkpoint_count=len(wanted),
                recorded_checkpoint_count=len(refs),
                missing_epoch_components=[
                    {"epoch": epoch, "component": component} for epoch, component in missing
                ],
                sources=[{"path": str(entry["path"]), "sha256": entry["manifest_sha256"]}],
                run_view=m.get("run_view"),
                admission_notes={
                    "recorded_formal_eligible": m.get("formal_eligible", False),
                    "recorded_blockers": m.get("formal_blockers", []),
                    "recorded_deviations": m.get("protocol_deviation", []),
                    "batch_packet_present": bool(m.get("batch")),
                    "requires_full_batch_audit": True,
                },
            )
        rows.append(row)
    return {
        "schema_version": "survey-support-repair-plan-1",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": digest(protocol),
        "scope": "trajectory_readiness_only_not_formal_admission",
        "counts": dict(Counter(row["status"] for row in rows)),
        "expected_fits": len(expected),
        "fits": rows,
        "excluded_sources": excluded,
        "original_sources_modified": False,
    }
