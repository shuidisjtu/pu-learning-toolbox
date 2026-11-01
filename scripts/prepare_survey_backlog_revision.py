"""Materialize the owner-confirmed v3.1 recipe revision; never rewrite v3.0.

This is a deterministic generator, not a training/admission gate. All generated
artifacts are validated before publication. No historical results are rebound.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pu_toolbox.experiment.survey_backlog import ADAPTER, DEEP_METHODS
from pu_toolbox.experiment.survey_comparison import (
    expected_result_units,
    validate_comparison_coverage,
)
from pu_toolbox.experiment.survey_protocol import digest, load_protocol, resolve_protocol_path
from pu_toolbox.experiment.survey_recipe_registry import build_locked_registry

METHODS = ("kldce", "p3mix", "puet", "gradpu", "pan", "rp", "cvir", "vpu")
ROOT = Path(__file__).resolve().parents[1] / "pu_toolbox" / "experiment"


def build_protocol():
    original = load_protocol(resolve_protocol_path("survey-v3.0"))
    protocol = copy.deepcopy(original)
    protocol["protocol_version"] = "survey-v3.1"
    protocol["revision_basis"] = {
        "protocol_version": original["protocol_version"],
        "protocol_sha256": digest(original),
        "owner_decision_date": "2026-11-01",
        "decision": (
            "current provisional parameters adopted as project recipes; paper differences retained"
        ),
        "decision_ref": "docs/research/pu_survey/admission/p4_1b_registry_freeze_20261025.md",
    }
    protocol["review_status"] = "pending_collaborator_review"
    protocol["formal_blockers"] = [
        "The owner approved project recipes, not numerical reproduction "
        "or collaborator comparison signatures.",
        "Per-unit full-data resource, five-seed, selection and artifact audits remain required.",
        "New comparison mappings have no direct anchors pending independent source review.",
    ]
    for method in METHODS:
        profile = protocol["method_profiles"][method]
        profile["configuration_status"] = "owner_confirmed_project_recipe_2026-11-01"
        profile["configuration_review"]["status"] = (
            "project_recipe_confirmed_source_differences_retained"
        )
        profile["execution_adapter"] = ADAPTER
        profile["budget"] = f"backlog_{method}"
        params = profile["params"]
        # Bind omitted constructor defaults explicitly; never tune to prior scores.
        if method == "p3mix":
            params["batching_policy"] = "role_balanced"
        if method in {"vpu", "cvir"}:
            # Storage placement is an engineering choice, not a batch/epoch reduction.
            params["data_device_policy"] = "cpu_stream"
        if method == "kldce":
            from pu_toolbox.estimators.risk.kldce import KLDCEClassifier

            defaults = KLDCEClassifier(flip_probability=0.5).get_params(deep=False)
            for key in (
                "sigma",
                "reg_strength",
                "centroid_radius",
                "mom_groups",
                "covariance_ridge",
                "inner_tol",
                "tol",
            ):
                params[key] = defaults[key]
        budget = {
            "unit": "epochs"
            if method in DEEP_METHODS
            else {"kldce": "alternating_solver", "puet": "trees", "rp": "crossfit_and_refit"}[
                method
            ],
            "outer_candidates": 1,
            "parameter_values": copy.deepcopy(params),
        }
        if method in DEEP_METHODS:
            budget.update(
                epochs=params["max_epochs"],
                batch_size=params["batch_size"],
                learning_rate=params["learning_rate"],
                checkpoint_bytes_per_component=64 * 1024**2,
            )
            budget["checkpoint_epoch_count"] = params["max_epochs"] + params.get(
                "warm_start_epochs", 0
            )
        elif method == "kldce":
            budget.update(
                max_acs_iter=params["max_acs_iter"],
                max_inner_iter=params["max_inner_iter"],
                kernel_payload_cap_bytes=256 * 1024**3,
            )
        protocol["budgets"][profile["budget"]] = budget
    for row in protocol["execution_units"]:
        method = row["method"]
        if method in METHODS or (not row["runnable"] and method in {"lagam", "pulns"}):
            row["budget"] = protocol["method_profiles"][method]["budget"]
            row["runnable"] = True
            row.pop("non_runnable_reason", None)
            row["comparability_status"] = "project_recipe_not_paper_reproduction"
            row["comparability_group"] = (
                f"{row['dataset']}/{row['training_path']}/project_{method}_v3_1"
            )
    # Complete template inheritance, not merely the original three datasets.
    templates = {
        "connect_4": "spambase",
        "twenty_newsgroups": "imdb",
        "mnist": "cifar10",
        "fashion_mnist": "cifar10",
        "adni": "cifar10",
    }
    for dataset, template in templates.items():
        for row in list(protocol["execution_units"]):
            if row["dataset"] != template or row["method"] not in METHODS:
                continue
            if any(
                r["dataset"] == dataset and r["method"] == row["method"]
                for r in protocol["execution_units"]
            ):
                continue
            inherited = copy.deepcopy(row)
            inherited["dataset"] = dataset
            inherited["comparability_group"] = row["comparability_group"].replace(
                template + "/", dataset + "/", 1
            )
            protocol["execution_units"].append(inherited)
    # CIFAR is already in the approved five-support-dataset scope. Follow the
    # existing full-meta MNIST adapter (no invented native-CNN implementation).
    row = copy.deepcopy(
        next(
            r
            for r in protocol["execution_units"]
            if r["method"] == "self_pu_full_meta" and r["dataset"] == "mnist"
        )
    )
    row["dataset"] = "cifar10"
    row["comparability_group"] = row["comparability_group"].replace("mnist/", "cifar10/", 1)
    protocol["execution_units"].append(row)
    protocol["draft_scope"] = {
        "status": "owner_confirmed_project_recipes_pending_execution_audit",
        "limitations": [
            "PUET/RP image paths remain outside the declared template scope; "
            "do not invent a feature recipe.",
            "KLDCE keeps SCAR only and rejects over-cap dense kernels rather than subsampling.",
            "Support profiles remain limited to the approved five datasets and SCAR.",
            "Earlier candidate results are not rebound to this version "
            "or counted as its completed runs.",
        ],
    }
    protocol["decisions"].append(
        "2026-11-01: owner adopts current provisional project recipes, including P3MIX "
        "base/default role_balanced batching. Historical merged_shuffle candidate results "
        "remain separate. VPU/CVIR cpu_stream preserves full batches/epochs. SAR is an "
        "OA-only stress test, not a claim of new selection-bias correction."
    )
    return protocol


def build_comparison(protocol):
    comparison = json.loads((ROOT / "survey_comparison_v6.json").read_text())
    profile_map = json.loads((ROOT / "survey_comparison_profile_map_v3.json").read_text())
    comparison["comparison_version"] = "survey-comparison-v7"
    comparison["review_status"] = "pending_collaborator_review"
    comparison["bound_survey_protocol"] = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": digest(protocol),
    }
    profile_map["map_version"] = "survey-profile-map-v4"
    profile_map["bound_survey_protocol"] = copy.deepcopy(comparison["bound_survey_protocol"])
    for method in METHODS:
        profile_map["profiles"][method] = {
            "registration": "mapped",
            "inherits": method,
            "deviations": [],
            "rationale": (
                "Owner-confirmed project recipe; new result units registered "
                "without a numeric source claim."
            ),
        }
    expected = expected_result_units(protocol, profile_map=profile_map)

    def key(selector):
        return json.dumps(selector, sort_keys=True)

    expected_keys = {key(unit) for unit in expected}
    kept = []
    for mapping in comparison["mappings"]:
        if mapping["scope"] == "unit" and key(mapping["result_selector"]) not in expected_keys:
            if mapping["anchor_ids"]:
                raise ValueError("revision would discard a published anchor")
            continue
        kept.append(mapping)
    present = {key(m["result_selector"]) for m in kept if m["scope"] == "unit"}
    for i, unit in enumerate(expected):
        if key(unit) not in present:
            kept.append(
                {
                    "mapping_id": f"v3_1_backlog_unit_{i}",
                    "scope": "unit",
                    "result_selector": unit,
                    "protocol_differences": [],
                    "review_state": "pending_review",
                    "verified_by": [],
                    "eligibility": "no_direct_anchor",
                    "anchor_ids": [],
                    "rationale": (
                        "Project-recipe extension has no registered direct numeric anchor; "
                        "source review remains open."
                    ),
                }
            )
    comparison["mappings"] = kept
    profile_map["bound_comparison"] = {
        "comparison_version": comparison["comparison_version"],
        "comparison_sha256": digest(comparison),
    }
    validate_comparison_coverage(protocol, comparison, profile_map=profile_map)
    return comparison, profile_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--update", action="store_true", help="regenerate only these four v3.1 artifacts"
    )
    args = parser.parse_args()
    protocol = build_protocol()
    comparison, profile_map = build_comparison(protocol)
    registry = build_locked_registry(protocol)
    artifacts = {
        "survey_protocol_v3_1.json": protocol,
        "survey_comparison_v7.json": comparison,
        "survey_comparison_profile_map_v4.json": profile_map,
        "survey_recipe_registry_v1_1.json": registry,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        path = args.output_dir / name
        if path.exists() and not args.update and json.loads(path.read_text()) != payload:
            raise ValueError(f"refusing to overwrite differing artifact: {path}")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
