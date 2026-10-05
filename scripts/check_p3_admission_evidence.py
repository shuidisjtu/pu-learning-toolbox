#!/usr/bin/env python3
"""Check P1-1 handoff facts without approving candidates or running training.

Offline checks compare defaults, calibration and references with this checkout.
Optional --source-dir method=path rechecks a local author checkout's commit and
file digests; neither mode downloads or executes author code. Passing proves
engineering consistency, not human paper readings or numeric reproduction.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_REF = "docs/research/pu_survey/data/p3_admission_evidence_20261005.json"
METHODS = ("pulda", "puet", "gradpu", "robust_pu", "split_pu")
SELECTION = {
    "training_role": "train",
    "PA_role": "pu_val",
    "OA_role": "clean_val",
    "test_role": "test",
    "test_used_for_selection": False,
    "fit_receives_validation_labels": False,
    "checkpoint_eligibility": "pending_method_owner_review",
}


def require(condition, message):
    """Reject inconsistent handoff facts rather than silently repairing them."""
    if not condition:
        raise ValueError(message)


def repo_file(reference, root):
    """Resolve only existing repository files, never an external/path-traversal ref."""
    require(isinstance(reference, str) and bool(reference), "missing repository reference")
    path = (root / reference).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f"bad reference: {reference}")
    return path


def constructor_facts(path, class_name):
    """Read literal constructor defaults and property returns without importing torch."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    args = init.args
    pairs = (
        list(zip(args.args[-len(args.defaults) :], args.defaults, strict=True))
        if args.defaults
        else []
    )
    pairs += list(zip(args.kwonlyargs, args.kw_defaults, strict=True))
    defaults = {arg.arg: ast.literal_eval(value) for arg, value in pairs if value is not None}
    properties = {
        node.name: ast.unparse(node.body[-1].value).replace("self.", "")
        for node in klass.body
        if isinstance(node, ast.FunctionDef)
        and node.decorator_list
        and any(isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list)
        and isinstance(node.body[-1], ast.Return)
    }
    return defaults, properties


def canonical(value):
    """Compare JSON types too: True must never pass as the integer one."""
    return json.dumps(value, sort_keys=True, allow_nan=False)


def validate_evidence(evidence, ledger, protocol, *, root=ROOT):
    """Validate technical consistency only; leave all formal decisions pending."""
    from pu_toolbox.experiment.survey_protocol import digest

    require(evidence["schema_version"] == "survey-p3-admission-evidence-draft-1", "bad schema")
    require(evidence["review_status"] == "pending_method_owner_review", "review must be pending")
    require(evidence["owner_signatures"] == [], "technical checker must not accept signatures")
    require(evidence["frozen_protocol_sha256"] == digest(protocol), "frozen protocol mismatch")
    require(
        canonical(evidence["selection_contract"]) == canonical(SELECTION), "selection role leak"
    )
    storage = evidence["storage_contract"]
    require(storage["snapshots_are"] == "inference_only_not_training_resume", "resume overclaim")
    require(storage["reclaim_reduces_training_peak"] is False, "reclaim peak overclaim")
    require(storage["formal_resource_approval"] is False, "resource approval overclaim")
    require(set(evidence["methods"]) == set(METHODS), "missing or unexpected methods")
    for method, row in evidence["methods"].items():
        entry = ledger["methods"][method]
        require(method not in protocol["method_profiles"], f"old pilot admission: {method}")
        require(entry["admission_evidence_ref"] == EVIDENCE_REF, f"ledger reference: {method}")
        require(row["class"] == entry["class"], f"class mismatch: {method}")
        defaults, properties = constructor_facts(
            repo_file(row["implementation_ref"], root), row["class"]
        )
        require(
            canonical(defaults) == canonical(row["constructor_defaults"]),
            f"default drift: {method}",
        )
        for key in ("run_view", "calibration_applied"):
            require(
                canonical(row["calibration"][key]) == canonical(entry[key]), f"view drift: {method}"
            )
        require(row["calibration"]["source_equivalence_claim"] is False, "calibration overclaim")
        budget = row["budget"]
        require(budget["formal_budget_approved"] is False, "budget approval overclaim")
        for key in ("approved_candidate_pool", "approved_storage_bytes"):
            require(budget[key] is None, f"unapproved {key}: {method}")
        expected_property = (
            "checkpoint_epoch_count" if "checkpoint_epoch_count" in properties else None
        )
        require(budget["snapshot_property"] == expected_property, "snapshot property drift")
        if expected_property is not None:
            require(
                properties[budget["snapshot_property"]] == budget["snapshot_bound"],
                "snapshot drift",
            )
        else:
            require(
                budget["snapshot_bound"] == ("max_epochs" if method == "gradpu" else None),
                "snapshot drift",
            )
        source = row["source"]
        require(source["repository"] == entry["code_version"]["upstream_url"], "upstream URL drift")
        if method == "gradpu":
            require(source["commit"] is None and source["files"] == [], "invented GradPU code")
            repo_file(source["paper_review_ref"], root)
        else:
            require(re.fullmatch(r"[0-9a-f]{40}", source["commit"]) is not None, "unpinned commit")
            require(entry["code_version"]["upstream_commit"] == source["commit"], "commit drift")
            require(bool(source["files"]), "missing source files")
            names = [f["path"] for f in source["files"]]
            require(len(set(names)) == len(names), "duplicate source files")
            for record in source["files"]:
                require(
                    re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None, "bad source digest"
                )
            for diff in row["adaptation_differences"]:
                location = diff["source_location"]
                if location is not None:
                    require(location["path"] in names, "unrecorded source file")
                    start, end = location["line_start"], location["line_end"]
                    require(
                        type(start) is int and type(end) is int and 0 < start <= end,
                        "bad source lines",
                    )
        require(
            bool(row["adaptation_differences"]) and bool(row["remaining_blockers"]),
            "missing boundaries",
        )
        require(bool(row["source_behavior_tests"]), "missing behavior tests")
        for reference in row["source_behavior_tests"]:
            file_ref, _, name = reference.partition("::")
            tree = ast.parse(repo_file(file_ref, root).read_text(encoding="utf-8"))
            require(
                any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body),
                "bad test ref",
            )
    return {"ok": True, "methods_checked": list(METHODS), "formal_admission": False}


def validate_source_checkout(source, directory):
    """Check locked commit and file bytes; do not import or run author modules."""
    require(source["commit"] is not None, "no verified author checkout for this method")
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    require(result.stdout.strip() == source["commit"], "source checkout commit mismatch")
    for record in source["files"]:
        path = repo_file(record["path"], directory)
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"],
            "source byte mismatch",
        )
    return len(source["files"])


def main(argv=None):
    """Emit a read-only JSON consistency receipt or an explicit error and exit one."""
    from pu_toolbox.experiment.method_ledger import load_ledger
    from pu_toolbox.experiment.survey_protocol import load_protocol

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / EVIDENCE_REF)
    parser.add_argument("--source-dir", action="append", default=[], metavar="METHOD=PATH")
    args = parser.parse_args(argv)
    try:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        receipt = validate_evidence(evidence, load_ledger(), load_protocol())
        checked = {}
        for item in args.source_dir:
            method, separator, directory = item.partition("=")
            require(
                separator and directory and method in METHODS and method not in checked,
                "bad source-dir",
            )
            checked[method] = validate_source_checkout(
                evidence["methods"][method]["source"], Path(directory)
            )
        receipt["source_files_checked"] = checked
        print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
        return 0
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        StopIteration,
        subprocess.CalledProcessError,
    ) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
