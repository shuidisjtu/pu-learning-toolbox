#!/usr/bin/env python3
"""Inspect old support results and prepare NEW-root commands, without training.

Uses the same D53 trajectory checker as audit_survey_batches.py. Missing or
shortened trajectories require reruns; complete PULNS episodes do not become
formal approvals merely by passing this narrow check. Nothing is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_survey_batches import _checkpoint_reassessment_problems  # noqa: E402

from pu_toolbox.experiment.support_repair import (  # noqa: E402
    SUPPORT_PROFILES,
    plan_support_repairs,
)
from pu_toolbox.experiment.survey_protocol import load_protocol  # noqa: E402


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(target):
    """Commands may only name clean code containing the audited epoch fix."""
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=target).strip():
        raise ValueError("target source must be a clean committed checkout")
    relative = Path("pu_toolbox/experiment/support_checkpoints.py")
    if file_hash(target / relative) != file_hash(SCRIPTS.parent / relative):
        raise ValueError("target support recorder differs from the inspected D53 implementation")
    return {
        "path": str(target),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=target, text=True
        ).strip(),
        "support_recorder_sha256": file_hash(target / relative),
    }


def collect_sources(roots):
    """Refuse overlapping roots, and bind every source to its original bytes."""
    roots = [Path(root).resolve() for root in roots]
    for i, root in enumerate(roots):
        if not root.is_dir():
            raise ValueError(f"result root is not a directory: {root}")
        if any(
            root == other or root.is_relative_to(other) or other.is_relative_to(root)
            for other in roots[:i]
        ):
            raise ValueError("result roots must not overlap")
    entries, unreadable = [], []
    for root in roots:
        for path in sorted(root.rglob("manifest.json")):
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                unreadable.append({"path": str(path), "reason": "concurrent_manifest_write"})
                continue
            try:
                payload = json.loads(raw)
            except ValueError:
                unreadable.append({"path": str(path), "reason": "unreadable_manifest"})
                continue
            if payload.get("profile_id") in SUPPORT_PROFILES:
                entries.append(
                    {
                        "path": path,
                        "payload": payload,
                        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
    return entries, unreadable


def pilot_commands(plan, *, source, support_root, split_map, results_root, report_root):
    """Rerun only whole affected units, so each command covers a full seed/c grid.

    D52 packets must be written by the pilot at execution time, not fabricated
    during planning. Results may NOT share a root with any historical input.
    """
    units = defaultdict(list)
    for fit in plan["fits"]:
        units[(fit["dataset"], fit["profile"], fit["training_path"])].append(fit)
    commands = []
    for (dataset, method, path), fits in sorted(units.items()):
        if not any(f["status"] in {"rerun_required", "missing"} for f in fits):
            continue
        if any(f["status"] == "conflict" for f in fits):
            continue
        split_dir = Path(split_map[dataset]).resolve()
        if split_dir.name != dataset or not split_dir.is_dir():
            raise ValueError(f"split map does not identify dataset directory: {dataset}")
        if not (support_root / dataset).is_dir():
            raise ValueError(f"missing support products: {dataset}")
        label = f"{dataset}__{method}__{path}"
        out = results_root / label
        argv = [
            sys.executable,
            str(source / "scripts/run_survey_pilot.py"),
            "--protocol",
            "survey-v3.0",
            "--datasets",
            dataset,
            "--methods",
            method,
            "--training-paths",
            path,
            "--splits",
            str(split_dir.parent),
            "--support-root",
            str(support_root),
            "--results",
            str(out),
            "--recipe-registry",
            str(source / "pu_toolbox/experiment/survey_recipe_registry_v1.json"),
            "--recipe-binding-schema-version",
            "survey-recipe-binding-2",
            "--device",
            "cuda",
            "--reclaim-unselected-checkpoints",
        ]
        if method == "self_pu_full_meta":
            argv += ["--os-or-ts", "os"]
        dry = list(argv)
        dry[dry.index("--device") + 1] = "cpu"
        dry += ["--dry-run", "--plan-json", str(report_root / (label + ".plan.json"))]
        commands.append(
            {
                "unit": {"dataset": dataset, "profile": method, "training_path": path},
                "whole_unit_fits": len(fits),
                "already_complete_trajectories_repeated_for_coherent_new_unit": sum(
                    f["status"] == "trajectory_complete" for f in fits
                ),
                "results_root": str(out),
                "pilot_argv": argv,
                "cpu_preflight_argv": dry,
                "launched": False,
            }
        )
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, action="append", required=True)
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument(
        "--split-map",
        type=Path,
        required=True,
        help="JSON dataset-directory mapping, or a queue evidence file with source_data",
    )
    parser.add_argument("--new-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.out_dir.resolve()
        fresh = args.new_results.resolve()
        roots = [p.resolve() for p in args.results_root]
        if output.exists() or fresh.exists():
            raise ValueError("report and new results roots must both be fresh")
        if output == fresh or output.is_relative_to(fresh) or fresh.is_relative_to(output):
            raise ValueError("report and new results roots must be separate")
        if any(
            fresh == root
            or fresh.is_relative_to(root)
            or root.is_relative_to(fresh)
            or output == root
            or output.is_relative_to(root)
            or root.is_relative_to(output)
            for root in roots
        ):
            raise ValueError("new outputs must not overlap historical input roots")
        source = args.target_source.resolve()
        identity = source_identity(source)
        protocol = load_protocol(source / "pu_toolbox/experiment/survey_protocol_v3.json")
        entries, unreadable = collect_sources(roots)
        plan = plan_support_repairs(
            protocol, entries, checkpoint_check=_checkpoint_reassessment_problems
        )
        split_map = json.loads(args.split_map.read_text())
        split_map = split_map.get("source_data", split_map)
        commands = pilot_commands(
            plan,
            source=source,
            support_root=args.support_root.resolve(),
            split_map=split_map,
            results_root=fresh,
            report_root=output,
        )
        plan.update(
            source=identity,
            result_roots=list(map(str, roots)),
            unreadable_sources=unreadable,
            commands=commands,
            launched=False,
            full_batch_admission="not_assessed_use_audit_survey_batches",
        )
        output.mkdir(parents=True)
        (output / "support_repair_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "cpu_preflight_commands.txt").write_text(
            "\n".join(shlex.join(c["cpu_preflight_argv"]) for c in commands) + "\n",
            encoding="utf-8",
        )
        (output / "README.txt").write_text(
            "Read-only D53 trajectory plan. Original manifests unchanged; no training launched.\n"
            "Complete trajectory metadata is not formal acceptance or a weight replay.\n"
            "Audit D54/code/environment/receipts/archive evidence separately.\n"
            "Reruns use NEW roots, frozen budgets and the existing D52 pilot.\n"
            "Do not run overlapping commands twice. cpu_preflight_commands.txt only plans.\n"
            "pilot_argv in JSON trains and needs GPU resources.\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "out_dir": str(output),
                    "counts": plan["counts"],
                    "whole_unit_commands": len(commands),
                    "launched": False,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
