"""Add PUET/RP image feature tests as v3.2; never rewrite active v3.1 artifacts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pu_toolbox.experiment.survey_comparison import (
    expected_result_units,
    validate_comparison_coverage,
)
from pu_toolbox.experiment.survey_protocol import digest, load_protocol, resolve_protocol_path
from pu_toolbox.experiment.survey_recipe_registry import build_locked_registry

ROOT = Path(__file__).resolve().parents[1] / "pu_toolbox/experiment"
DATASETS = ("cifar10", "mnist", "fashion_mnist", "adni")


def build_protocol():
    original = load_protocol(resolve_protocol_path("survey-v3.1"))
    protocol = copy.deepcopy(original)
    protocol["protocol_version"] = "survey-v3.2"
    protocol["revision_basis"] = {
        "protocol_version": original["protocol_version"],
        "protocol_sha256": digest(original),
        "owner_decision_date": "2026-11-01",
        "decision": (
            "Owner requests determining and testing the missing PUET/RP image paths. "
            "Use the existing fixed-random feature adapter with unchanged project budgets; "
            "canary-gated execution does not certify paper reproduction or formal admission."
        ),
        "decision_ref": "docs/research/pu_survey/admission/p4_1b_registry_freeze_20261025.md",
    }
    for method in ("puet", "rp"):
        template = next(
            r
            for r in original["execution_units"]
            if r["method"] == method and r["dataset"] == "spambase"
        )
        for dataset in DATASETS:
            row = copy.deepcopy(template)
            row.update(
                dataset=dataset,
                training_path="cnn_feature_adapter",
                backbone="resnet18_random_frozen",
                representation="resnet18_random_initialization",
                comparability_group=f"{dataset}/cnn_feature_adapter/project_{method}_v3_2",
                oracle_alignment="no_matched_neural_backbone_oracle",
            )
            protocol["execution_units"].append(row)
    protocol["draft_scope"]["limitations"][0] = (
        "PUET/RP images use the existing seeded random frozen ResNet-18 feature adapter; "
        "no pretrained weights, encoder fitting, new labels or native CNN claim. "
        "Retain source-space SCAR/SAR label generation and independent PA/OA selection."
    )
    protocol["decisions"].append(
        "2026-11-01: test the remaining eight PUET/RP image cells with the existing "
        "fixed-random feature recipe. Same dataset/split/seed encoder is shared across "
        "methods and labeling mechanisms; train-only normalization, no augmentation. "
        "No changes to estimators, 100 trees, RP 3-fold/refit, or historical results. "
        "All 280 fits are canary-gated; failed families remain held."
    )
    return protocol


def build_artifacts():
    protocol = build_protocol()
    comparison = json.loads((ROOT / "survey_comparison_v7.json").read_text(encoding="utf-8"))
    profile_map = json.loads(
        (ROOT / "survey_comparison_profile_map_v4.json").read_text(encoding="utf-8")
    )
    comparison["comparison_version"] = "survey-comparison-v8"
    binding = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": digest(protocol),
    }
    comparison["bound_survey_protocol"] = binding
    profile_map["bound_survey_protocol"] = copy.deepcopy(binding)
    profile_map["map_version"] = "survey-profile-map-v5"

    def key(value):
        return json.dumps(value, sort_keys=True)

    existing = {key(m["result_selector"]) for m in comparison["mappings"] if m["scope"] == "unit"}
    for i, unit in enumerate(expected_result_units(protocol, profile_map=profile_map)):
        if key(unit) in existing:
            continue
        comparison["mappings"].append(
            {
                "mapping_id": f"v3_2_classical_image_{i}",
                "scope": "unit",
                "result_selector": unit,
                "protocol_differences": [],
                "review_state": "pending_review",
                "verified_by": [],
                "eligibility": "no_direct_anchor",
                "anchor_ids": [],
                "rationale": "Fixed random ResNet features with original classical estimator; "
                "not a native CNN or paper numeric reproduction. No direct source anchor.",
            }
        )
    profile_map["bound_comparison"] = {
        "comparison_version": comparison["comparison_version"],
        "comparison_sha256": digest(comparison),
    }
    validate_comparison_coverage(protocol, comparison, profile_map=profile_map)
    return {
        "survey_protocol_v3_2.json": protocol,
        "survey_recipe_registry_v1_2.json": build_locked_registry(protocol),
        "survey_comparison_v8.json": comparison,
        "survey_comparison_profile_map_v5.json": profile_map,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    artifacts = build_artifacts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        path = args.output_dir / name
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise ValueError(f"refusing to overwrite changed artifact: {path}")
        else:
            path.write_text(text, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
