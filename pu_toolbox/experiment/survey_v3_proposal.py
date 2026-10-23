"""Source-bound v3 profile/unit requests, deliberately not an execution protocol.

Keep 22 target algorithms separate from recipe variants and from the registered
extras. Constructor observations are evidence, never proposed frozen budgets.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from pathlib import Path

from ..utils.serialization import strict_canonical_hash
from .datasets import survey_dataset_catalog
from .support_data import SUPPORT_DATASETS
from .survey_protocol import digest, load_protocol

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "survey-v3-profile-unit-proposal-1"
PROTOCOL = "pu_toolbox/experiment/survey_protocol_v2.json"
LEDGER = "pu_toolbox/experiment/method_ledger.json"
RV8 = "docs/research/pu_survey/reviews/rv8_method_admission_20261021.md"
P31 = (
    "p3mix",
    "puet",
    "pulda",
    "gradpu",
    "robust_pu",
    "split_pu",
    "pan",
    "rp",
    "pulns",
    "genpu",
    "holistic_pu",
    "cvir",
    "lagam",
)
COMMON_BLOCKERS = (
    "versioned_profiles_units_and_comparison_binding",
    "formal_budget_candidates_seed_grid_and_stage_eligibility",
    "actual_runner_and_representation_binding",
)
ENGINEERING = {
    "genpu": "docs/research/pu_survey/delivery/genpu_recipe_20261023.md",
    "pulda": "docs/research/pu_survey/delivery/candidate_image_runner_20261023.md",
    "gradpu": "docs/research/pu_survey/delivery/candidate_image_runner_20261023.md",
    "cvir": "docs/research/pu_survey/delivery/cvir_calibration_20261022.md",
    "lagam": "docs/research/pu_survey/delivery/clean_support_materialization_20261022.md",
    "pulns": "docs/research/pu_survey/delivery/clean_support_materialization_20261022.md",
    "self_pu_full_meta": "docs/research/pu_survey/delivery/selfpu_ts_support_20261023.md",
}


def _file(root, reference):
    path = (root / reference).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or external proposal source: {reference}")
    return path


def _text_hash(path):
    # This planning artifact crosses Git CRLF checkouts; GPU receipts still use bytes.
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _classes(root):
    result = {}
    for path in sorted((root / "pu_toolbox/estimators").rglob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef):
                result.setdefault(node.name, []).append((path, node))
    return result


def _constructor(klass):
    init = next(
        (n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
    )
    if init is None:
        raise ValueError(f"explicit constructor required for {klass.name}")
    args = init.args
    pairs = (
        list(zip(args.args[-len(args.defaults) :], args.defaults, strict=True))
        if args.defaults
        else []
    )
    pairs += list(zip(args.kwonlyargs, args.kw_defaults, strict=True))
    literals, expressions = {}, {}
    for arg, value in pairs:
        if value is None:
            continue
        try:
            literals[arg.arg] = ast.literal_eval(value)
        except (ValueError, TypeError):
            expressions[arg.arg] = ast.unparse(value)
    return {
        "literal_defaults": literals,
        "unevaluated_expressions": expressions,
        "defaults_are_approved_budget": False,
    }


def build_proposal(*, root=ROOT):
    """Read constructor AST without instantiating estimators or filling budgets.

    The package's existing import bootstrap may import optional torch when it
    is installed; this builder does not require it or initialize any device.
    """
    root = Path(root).resolve()
    protocol = load_protocol(_file(root, PROTOCOL))
    if protocol["protocol_version"] != "survey-v2.0":
        raise ValueError("proposal must preserve the explicit survey-v2.0 basis")
    ledger = json.loads(_file(root, LEDGER).read_text(encoding="utf-8"))
    oracle_source = "pu_toolbox/experiment/survey_execution.py"
    oracle_classes = {
        node.name
        for node in ast.parse(_file(root, oracle_source).read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef)
    }
    if not {"PilotOracleMLP", "PilotOracleCNN"} <= oracle_classes:
        raise ValueError("oracle implementation class missing; do not invent CNN feasibility")
    methods = ledger["methods"]
    rv8 = _file(root, RV8).read_text(encoding="utf-8")
    reviewed = re.findall(r"^### \d+\. .*?`([^`]+)`", rv8, re.MULTILINE)
    if reviewed != [*P31, "tedn"]:
        raise ValueError("RV8 scope/order changed; do not infer a new target set")
    if set(ledger["pending_owner_review_methods"]) != set(reviewed):
        raise ValueError("historical ledger review scope and signed RV8 disagree")
    existing = list(protocol["method_profiles"])
    if set(existing) != {
        "upu",
        "nnpu",
        "kldce",
        "dist_pu",
        "pusb_kernel",
        "lbe",
        "self_pu",
        "pn_oracle",
    }:
        raise ValueError("unexpected eight-method v2 basis; do not silently redefine the target")
    targets = [*existing, *P31, "vpu"]
    if len(targets) != 22 or len(set(targets)) != 22:
        raise ValueError("22 target algorithms must be distinct from recipe variants")
    if set(targets) - {"pn_oracle"} - methods.keys():
        raise ValueError("target algorithm missing from ledger")
    classes, sources = _classes(root), {PROTOCOL, LEDGER, RV8}
    profiles = {}
    for algorithm in targets:
        inherited = algorithm in protocol["method_profiles"]
        entry = methods.get(algorithm)
        facts = None
        if entry is not None:
            matches = classes.get(entry["class"], [])
            if len(matches) != 1:
                raise ValueError(f"missing/ambiguous implementation class: {algorithm}")
            path, klass = matches[0]
            reference = path.relative_to(root).as_posix()
            sources.add(reference)
            facts = {
                "class": entry["class"],
                "implementation_ref": reference,
                "constructor_observations": _constructor(klass),
                "ledger_snapshot": copy.deepcopy(
                    {
                        key: entry[key]
                        for key in (
                            "run_view",
                            "calibration_applied",
                            "prior_semantics",
                            "adaptation_level",
                            "modality_backbone",
                            "native_sampling_assumption",
                        )
                    }
                ),
            }
        profiles[algorithm] = {
            "algorithm": algorithm,
            "profile_kind": "inherited_v2" if inherited else "new_algorithm_request",
            "inherited_profile": copy.deepcopy(protocol["method_profiles"].get(algorithm)),
            "implementation_facts": facts,
            "proposed_params": None,
            "proposed_budget": None,
            "proposed_candidates": None,
            "proposed_stage_eligibility": None,
            "formal_admission": False,
            "evidence_review_basis": RV8 if algorithm in P31 else None,
            "evidence_review_is_formal_admission": False,
            "engineering_delivery_ref": ENGINEERING.get(algorithm),
            "requires_clean_support": algorithm in ("pulns", "lagam"),
        }
    profiles["self_pu_full_meta"] = {
        **copy.deepcopy(profiles["self_pu"]),
        "profile_kind": "separate_full_meta_variant_request",
        "inherited_profile": None,
        "engineering_delivery_ref": ENGINEERING["self_pu_full_meta"],
        "requires_clean_support": True,
        "required_semantics": {
            "require_validation": True,
            "distillation_scope": "paper_population",
        },
        "available_engineering_training_views": ["os", "ts"],
        "formal_training_view": None,
    }
    catalog = survey_dataset_catalog()
    datasets = {name: {"modality": spec.modality} for name, spec in sorted(catalog.items())}
    requests = []
    for profile_id, profile in profiles.items():
        capability = (
            profile["implementation_facts"]["ledger_snapshot"]["modality_backbone"][
                "code_capability"
            ]
            if profile["implementation_facts"]
            else {"input_ndims": [2], "native_architectures": ["mlp"]}
        )
        # These are dtype/shape feasibility options, never backbone approval.
        image = ["cnn_feature_adapter"] if 2 in capability["input_ndims"] else []
        if profile_id == "pn_oracle" or (
            4 in capability["input_ndims"] and "cnn" in capability["native_architectures"]
        ):
            image.append("native_cnn")
        requests.append(
            {
                "profile_id": profile_id,
                "datasets": list(datasets),
                "path_options_by_modality": {
                    "tabular": ["native_2d"],
                    "text": ["native_2d"],
                    "image": image,
                },
                "path_options_are_code_feasibility_not_approved_architecture": True,
                "runnable": False,
            }
        )
    sources.update(ref for ref in ENGINEERING.values())
    sources.update(
        (
            "pu_toolbox/experiment/survey_v3_proposal.py",
            "scripts/prepare_survey_v3_proposal.py",
            "pu_toolbox/experiment/survey_execution.py",
            "pu_toolbox/experiment/survey_protocol.py",
            "pu_toolbox/experiment/survey_profile_identity.py",
            "pu_toolbox/experiment/survey_staged_budget.py",
            "pu_toolbox/experiment/pilot_plan.py",
            "pu_toolbox/experiment/runner.py",
            "scripts/run_survey_experiment.py",
            "scripts/run_survey_pilot.py",
            "pu_toolbox/experiment/datasets.py",
            "pu_toolbox/experiment/support_data.py",
        )
    )
    payload = {
        "schema_version": SCHEMA,
        "status": "planning_only_not_executable_protocol",
        "proposed_protocol_version": "survey-v3.0",
        "formal_admission": False,
        "owner_signatures": [],
        "source_hash_policy": "UTF8_universal_newlines_not_GPU_raw_bytes",
        "basis": {
            "protocol_ref": PROTOCOL,
            "protocol_sha256": digest(protocol),
            "inherited_review_items": copy.deepcopy(protocol.get("review_items", [])),
            "inherited_execution_units": copy.deepcopy(protocol["execution_units"]),
        },
        "scope": {
            "existing_v2_algorithms": existing,
            "p3_1_algorithms": list(P31),
            "additional_target_algorithms": ["vpu"],
            "target_algorithms": targets,
            "registered_outside_target_scope": sorted(methods.keys() - set(targets)),
            "variants_do_not_increase_algorithm_count": ["self_pu_full_meta"],
        },
        "datasets": datasets,
        "method_profile_requests": profiles,
        "execution_unit_requests": requests,
        "common_blockers": list(COMMON_BLOCKERS),
        "later_acceptance_checks_not_pretraining_approval": [
            "matching_resource_peak_D25_and_smoke_record",
            "preregistered_numerical_comparison_and_main_table_gates",
        ],
        "oracle_implementation": {
            "ref": oracle_source,
            "classes": ["PilotOracleMLP", "PilotOracleCNN"],
        },
        "support_policy_datasets": list(SUPPORT_DATASETS),
        "deferred_experiments": {
            "pn_oracle/native_cnn/five_seed": "deferred_until_protocol_final_per_user"
        },
        "specific_blockers": {
            "split_pu": ["issue_122_JSD_TS_batch_and_ledger_pointer_decisions"],
            "vpu": ["VPU_PA_selection_spec_not_variational_diagnostic"],
            "cvir": ["formal_TS_pool_and_per_view_prior_binding"],
            "gradpu": ["private_no_BN_group_and_SHOULD_shared_comparison_binding"],
        },
        "source_files_sha256": {ref: _text_hash(_file(root, ref)) for ref in sorted(sources)},
    }
    payload["proposal_sha256"] = strict_canonical_hash(payload)
    return payload


def expand_unit_requests(proposal):
    """Materialize blocked dataset/profile requests, not runnable execution rows."""
    if proposal.get("schema_version") != SCHEMA or proposal.get("formal_admission") is not False:
        raise ValueError("expected an unapproved planning proposal")
    if proposal.get("proposal_sha256") != strict_canonical_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    ):
        raise ValueError("proposal identity mismatch")
    rows, seen = [], set()
    for request in proposal["execution_unit_requests"]:
        if request["runnable"] is not False:
            raise ValueError("planning requests cannot authorize execution")
        profile_id = request["profile_id"]
        profile = proposal["method_profile_requests"][profile_id]
        if (
            any(
                profile[key] is not None
                for key in (
                    "proposed_params",
                    "proposed_budget",
                    "proposed_candidates",
                    "proposed_stage_eligibility",
                )
            )
            or profile["formal_admission"] is not False
        ):
            raise ValueError(
                "budget/parameter approval cannot be inserted into a planning artifact"
            )
        for dataset in request["datasets"]:
            identity = (profile_id, dataset)
            if identity in seen:
                raise ValueError("duplicate dataset/profile request")
            seen.add(identity)
            modality = proposal["datasets"][dataset]["modality"]
            blockers = list(proposal["common_blockers"])
            blockers += proposal["specific_blockers"].get(profile["algorithm"], [])
            if profile["requires_clean_support"]:
                blockers.append("independent_support_product_and_extra_label_budget_binding")
                if dataset not in proposal["support_policy_datasets"]:
                    blockers.append("dataset_outside_existing_support_policy")
            if dataset == "adni":
                blockers.append("subject_isolation_and_placeholder_source_scope_required")
            rows.append(
                {
                    "profile_id": profile_id,
                    "algorithm": profile["algorithm"],
                    "dataset": dataset,
                    "path_options": list(request["path_options_by_modality"][modality]),
                    "runnable": False,
                    "formal_admission": False,
                    "blockers": blockers,
                }
            )
    return rows


def validate_proposal(proposal, *, root=ROOT):
    """Reject stale sources, ghosts, missing cells, or fabricated authorizations."""
    if strict_canonical_hash(proposal) != strict_canonical_hash(build_proposal(root=root)):
        raise ValueError("proposal differs from current sources/scope or inserts unapproved values")
    return expand_unit_requests(proposal)
