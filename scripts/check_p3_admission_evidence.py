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
#: The dated P3 evidence bound the frozen pilot protocol.  It keeps validating
#: against survey-v1.2 after the canonical protocol moved to survey-v2.0; a
#: refresh under v2 is a separate, recorded decision rather than a silent edit.
DATED_PROTOCOL = "survey-v1.2"
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
PUET_BOOTSTRAP_RISK = {
    "scope": "per_tree_root",
    "positive_resample_count": "n_P",
    "unlabeled_resample_count": "effective_U_risk_pool_size",
    "positive_total_weight": "population_pi",
    "unlabeled_total_weight": 1.0,
    "ts_unlabeled_pool": "U_plus_positive_role_copies",
}
PUET_MASS_TEST = (
    "tests/unit/estimators/test_puet.py::test_edge_bootstrap_root_masses_and_constant_leaf"
)
ROBUST_REVIEW_REF = "docs/research/pu_survey/data/robustpu_source_review_20261018.json"
ROBUST_SOURCE_FILES = {
    "main.py",
    "spl_utills.py",
    "lossFunc.py",
    "utils.py",
    "models.py",
    "README.md",
}
ROBUST_PRIOR_CONTRACT = {
    "calibrated_ts_input": "population_pi",
    "explicit_os_input": "supplied_U_positive_fraction",
    "automatic_prior_conversion": False,
    "prior_used_in": "nnPU_pretrain_only",
    "post_pretrain_pseudo_negative_pool": "original_U_only",
}
ROBUST_BEHAVIOR_TESTS = {
    "tests/unit/estimators/test_robust_pu_source_contract.py::" + name
    for name in (
        "test_basic_temperature_hardness_weights_match_independent_scalar_formula",
        "test_edge_nnpu_optimization_branches_match_source_not_clipped_risk_gradients",
        "test_basic_actual_episode_bce_gradient_is_row_mean_not_weight_normalized",
        "test_determ_moving_weights_follow_original_rows_through_shuffle_and_inner_epochs",
        "test_edge_actual_linear_schedule_starts_at_zero_and_caps_both_directions",
        "test_param_invalid_retained_subset_contract_is_not_silently_expanded",
    )
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


def _validate_robust_source_review(row, entry, root):
    """Require the requested source/behavior/prior evidence, never owner approval."""
    source = row["source"]
    require(
        source.get("paper_review_ref")
        == entry.get("source_behavior_review_ref")
        == ROBUST_REVIEW_REF,
        "Robust-PU source review reference drift",
    )
    review = json.loads(repo_file(ROBUST_REVIEW_REF, root).read_text(encoding="utf-8"))
    require(
        review["schema_version"] == "survey-robustpu-source-engineering-review-1"
        and review["method"] == "robust_pu"
        and review["review_status"] == "pending_method_owner_review"
        and review["owner_signatures"] == [],
        "Robust-PU source review identity or approval overclaim",
    )
    facts = review["source"]
    require(
        facts["repository"] == source["repository"] and facts["commit"] == source["commit"],
        "Robust-PU source review provenance drift",
    )
    digests = {record["path"]: record["sha256"] for record in source["files"]}
    require(set(digests) >= ROBUST_SOURCE_FILES, "Robust-PU incomplete source files")
    require(canonical(digests) == canonical(facts["file_sha256"]), "Robust-PU review digest drift")
    text_digests = source.get("normalized_text_sha256", {})
    require(
        source.get("normalized_text_digest_policy")
        == facts.get("normalized_text_digest_policy")
        == "utf8_text_universal_newlines_sha256"
        and set(text_digests) == set(digests)
        and all(re.fullmatch(r"[0-9a-f]{64}", v) is not None for v in text_digests.values())
        and canonical(text_digests) == canonical(facts.get("normalized_text_sha256")),
        "Robust-PU normalized text digest drift",
    )
    require(facts["executed_upstream_code"] is False, "Robust-PU author-code execution overclaim")
    license_fact = facts["license_observation"]
    require(
        license_fact["status"] == "unknown"
        and license_fact["tracked_license_file_found"] is False
        and license_fact["explicit_code_license_notice_found"] is False
        and license_fact["permission_inferred_from_public_repository"] is False
        and license_fact["author_code_copied_executed_or_redistributed"] is False,
        "Robust-PU license observation/permission drift",
    )
    for fact in (
        row["calibration"].get("prior_input_contract"),
        entry.get("runtime_prior_contract"),
        review.get("prior_input_contract"),
    ):
        require(
            canonical(fact) == canonical(ROBUST_PRIOR_CONTRACT), "Robust-PU prior contract drift"
        )
    require(
        set(row["source_behavior_tests"]) >= ROBUST_BEHAVIOR_TESTS
        and set(review["behavior_test_refs"]) >= ROBUST_BEHAVIOR_TESTS,
        "Robust-PU missing source behavior regression",
    )
    for key in ("formal_admission", "formal_budget_approved", "source_equivalence_claim"):
        require(review["engineering_scope"][key] is False, "Robust-PU source acceptance overclaim")


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
        if method == "puet":
            for fact in (
                row["calibration"].get("bootstrap_risk_contract"),
                entry.get("bootstrap_risk_contract"),
            ):
                require(
                    canonical(fact) == canonical(PUET_BOOTSTRAP_RISK), "PUET bootstrap mass drift"
                )
            require(PUET_MASS_TEST in row["source_behavior_tests"], "missing PUET mass regression")
        if method == "robust_pu":
            _validate_robust_source_review(row, entry, root)
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
        text_digests = source.get("normalized_text_sha256")
        if text_digests is not None:
            require(
                set(text_digests) == {row["path"] for row in source["files"]}
                and source.get("normalized_text_digest_policy")
                == "utf8_text_universal_newlines_sha256",
                "source text digest coverage/policy mismatch",
            )
            require(
                hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                == text_digests[record["path"]],
                "source normalized text mismatch",
            )
    return len(source["files"])


def main(argv=None):
    """Emit a read-only JSON consistency receipt or an explicit error and exit one."""
    from pu_toolbox.experiment.method_ledger import load_ledger
    from pu_toolbox.experiment.survey_protocol import load_protocol, resolve_protocol_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / EVIDENCE_REF)
    parser.add_argument("--source-dir", action="append", default=[], metavar="METHOD=PATH")
    args = parser.parse_args(argv)
    try:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        receipt = validate_evidence(
            evidence,
            load_ledger(),
            load_protocol(resolve_protocol_path(DATED_PROTOCOL)),
        )
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
