#!/usr/bin/env python3
"""Prepare/check method-specific selection and budget facts, never executable recipes.

AST inspection avoids importing torch or running estimators. Default output is
strict JSON on stdout. --check compares the recorded draft with current code
and ledger facts; neither mode writes files, approves stages or starts training.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts.check_p3_admission_evidence import (
    EVIDENCE_REF,
    METHODS,
    ROOT,
    canonical,
    constructor_facts,
    repo_file,
    require,
    validate_evidence,
)
from scripts.check_p3_public_comparison import (
    BOUND_FILES,
    validate_draft,
)
from scripts.check_p3_public_comparison import DRAFT_REF as COMPARISON_REF

DRAFT_REF = "docs/research/pu_survey/data/p3_selection_budget_spec_20261007_draft.json"
STAGES = {
    "pulda": ("warmup", "pu_mixup"),
    "puet": (),
    "gradpu": ("pu",),
    "robust_pu": ("pretrain", "self_paced"),
    "split_pu": ("teacher", "splitter", "student"),
}
STEP_FACTS = {
    "pulda": "(warmup_epochs + pu_epochs) * ceil(n_native_U / unlabeled_batch_size)",
    "puet": None,
    "gradpu": "max_epochs * max(ceil(n_P / batch_size), ceil(n_native_U / batch_size))",
    "robust_pu": (
        "pretrain_epochs * max(ceil(n_P / batch_size), ceil(n_native_U / batch_size))"
        " + episodes * inner_epochs * ceil((n_P + n_native_U) / batch_size)"
    ),
    "split_pu": (
        "teacher_epochs * max(ceil(n_P / batch_size), ceil(n_native_U / batch_size))"
        " + actual_split_epochs * ceil((n_P + n_native_U) / batch_size)"
        " + sum_round(student_epochs * max(ceil(n_P / batch_size),"
        " ceil(n_easy / batch_size), ceil(n_hard / batch_size)))"
    ),
}
QUESTIONS = {
    "pulda": ["warmup_vs_PU_stage_eligibility", "two_stage_budget_and_cosine_horizon"],
    "puet": ["CPU_tree_budget_family_and_GPU_exemption", "raw_pixels_vs_frozen_features"],
    "gradpu": ["paper_validation_labels_and_deviation", "CNN_no_BN_and_schedule"],
    "robust_pu": ["pretrain_vs_episode_eligibility", "prior_scope_and_clean_val_source_budget"],
    "split_pu": ["teacher_splitter_student_round_eligibility", "early_stop_and_paper_code_epochs"],
}


def _interface(path, class_name):
    """Read named fit parameters and the opt-in checkpoint stage declaration."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fit = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "fit")
    require(fit.args.kwarg is None, "fit must not silently accept extra label roles")
    parameters = [arg.arg for arg in (*fit.args.posonlyargs, *fit.args.args, *fit.args.kwonlyargs)]
    require(
        not {"y_true", "X_val", "y_val", "clean_val", "test_data"} & set(parameters),
        "fit role leak",
    )
    stages = ()
    for node in klass.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "checkpoint_stages"
            for target in node.targets
        ):
            stages = ast.literal_eval(node.value)
    require(isinstance(stages, tuple), "nonliteral stage declaration")
    return parameters[1:], stages


def _budget(method, defaults):
    """Keep trees, epoch snapshots, episode snapshots and real updates distinct."""
    if method == "puet":
        count, expression = None, None
    elif method == "pulda":
        count = defaults["warmup_epochs"] + defaults["pu_epochs"]
        expression = "warmup_epochs + pu_epochs"
    elif method == "gradpu":
        count, expression = defaults["max_epochs"], "max_epochs"
    elif method == "robust_pu":
        count = defaults["pretrain_epochs"] + defaults["episodes"]
        expression = "pretrain_epochs + episodes"
    else:
        count = (
            defaults["teacher_epochs"]
            + defaults["split_epochs"]
            + defaults["rounds"] * defaults["student_epochs"]
        )
        expression = "teacher_epochs + split_epochs + rounds * student_epochs"
    return {
        "default_snapshot_count": count,
        "snapshot_bound_expression": expression,
        "bound_only_due_to_early_stop": method == "split_pu",
        "trees_per_default_fit": defaults["n_estimators"] if method == "puet" else None,
        "optimizer_steps_expression": STEP_FACTS[method],
        "counter_source": None if method == "puet" else "optimizer_steps_",
        "tree_fit_is_neural_epoch": False,
        "defaults_are_approved_budget": False,
    }


def build_spec(ledger, protocol, *, root=ROOT):
    """Source-bound pending draft; no inference of future approved values."""
    evidence = json.loads(repo_file(EVIDENCE_REF, root).read_text(encoding="utf-8"))
    comparison = json.loads(repo_file(COMPARISON_REF, root).read_text(encoding="utf-8"))
    validate_evidence(evidence, ledger, protocol, root=root)
    validate_draft(comparison, ledger, protocol, root=root)
    require(set(evidence["methods"]) == set(METHODS), "evidence scope drift")
    require(comparison["numeric_policy"]["enabled"] is False, "public comparison must stay draft")
    methods = {}
    for method in METHODS:
        entry, old = ledger["methods"][method], evidence["methods"][method]
        require(method not in protocol["method_profiles"], "new method in historical pilot")
        require(entry["selection_budget_spec_ref"] == DRAFT_REF, "ledger spec ref drift")
        require(entry["admission_evidence_ref"] == EVIDENCE_REF, "ledger evidence ref drift")
        require(entry["public_comparison_ref"] == COMPARISON_REF, "ledger comparison ref drift")
        require(old["class"] == entry["class"], "class identity drift")
        path = repo_file(old["implementation_ref"], root)
        defaults, properties = constructor_facts(path, entry["class"])
        require(canonical(defaults) == canonical(old["constructor_defaults"]), "default drift")
        parameters, stages = _interface(path, entry["class"])
        require(stages == STAGES[method], "stage declaration drift")
        require(("epoch_callback" in parameters) == bool(stages), "callback/stage mismatch")
        budget = _budget(method, defaults)
        if method not in {"puet", "gradpu"}:
            require(
                properties["checkpoint_epoch_count"] == budget["snapshot_bound_expression"],
                "snapshot bound drift",
            )
        methods[method] = {
            "class": entry["class"],
            "implementation_ref": old["implementation_ref"],
            "implementation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fit_named_parameters": parameters,
            "constructor_defaults": defaults,
            "representation_fact": entry["modality_backbone"],
            "calibration_fact": old["calibration"],
            "budget_fact": budget,
            "checkpoint_fact": {
                "stages": list(stages),
                "reference_schema": "1.1" if stages else None,
                "provenance_fields": ["stage", "stage_epoch", "round_index", "optimizer_steps"]
                if stages
                else [],
                "single_final_fit_without_epoch_callback": method == "puet",
                "training_resume_supported": False,
            },
            "current_technical_selection": {
                "scope": "all_recorded_inference_snapshots" if stages else "single_fitted_forest",
                "PA_role": "pu_val",
                "OA_role": "clean_val",
                "selection_changes_training": False,
                "current_behavior_approves_formal_stage_eligibility": False,
            },
            "formal_decisions": {
                "candidate_protocol_ref": None,
                "candidate_pool": None,
                "approved_budget": None,
                "approved_storage_bytes": None,
                "eligible_stages_PA": None,
                "eligible_stages_OA": None,
                "eligible_student_rounds": None,
                "threshold_policy_ref": None,
                "approved_representation_ref": None,
                "GPU_exemption_approved": False,
                "admitted": False,
            },
            "owner_questions": QUESTIONS[method],
            "remaining_blockers": old["remaining_blockers"],
        }
    return json.loads(
        json.dumps(
            {
                "schema_version": "survey-p3-selection-budget-spec-draft-1",
                "review_status": "pending_method_owner_review",
                "owner_signatures": [],
                "scope": (
                    "Current engineering facts only, not executable recipes"
                    " or formal stage admission."
                ),
                "evidence_ref": EVIDENCE_REF,
                "public_comparison_ref": COMPARISON_REF,
                "bound_files_sha256": {
                    ref: hashlib.sha256(repo_file(ref, root).read_bytes()).hexdigest()
                    for ref in BOUND_FILES
                },
                "checkpoint_writer_sha256": hashlib.sha256(
                    repo_file("pu_toolbox/experiment/checkpoints.py", root).read_bytes()
                ).hexdigest(),
                "storage_contract": {
                    "weights_only_inference_snapshots": True,
                    "reclaim_reduces_training_peak": False,
                    "prior_probe_numbers_are_formal_upper_bounds": False,
                    "neural_peak_basis": (
                        "bytes_per_component * actual_snapshot_count * components * candidates"
                        " * retained_attempts"
                    ),
                    "peak_before_training_uses_declared_snapshot_bound": True,
                    "tree_storage_requires_separate_measured_profile": True,
                },
                "methods": methods,
            },
            allow_nan=False,
        )
    )


def main(argv=None):
    """Print the draft or a readonly consistency receipt; no file writes."""
    from pu_toolbox.experiment.method_ledger import load_ledger
    from pu_toolbox.experiment.survey_protocol import load_protocol

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, metavar="DRAFT_JSON")
    args = parser.parse_args(argv)
    try:
        generated = build_spec(load_ledger(), load_protocol())
        if args.check is not None:
            recorded = json.loads(args.check.read_text(encoding="utf-8"))
            require(
                canonical(recorded) == canonical(generated),
                "selection/budget draft drift or false approval",
            )
            print(
                json.dumps(
                    {"ok": True, "methods_checked": list(METHODS), "formal_admission": False}
                )
            )
        else:
            print(json.dumps(generated, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, StopIteration, SyntaxError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
