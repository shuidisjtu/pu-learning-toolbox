"""Print a source-bound P3 extension handoff, not executable candidate recipes.

The existing eight-method admission draft remains unchanged. This separate
five-method draft quotes the ledger rather than inventing approved budgets,
selection criteria, support-label allowances or method-owner signatures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pu_toolbox.experiment.survey_protocol import digest, load_protocol

METHOD_BLOCKERS = {
    "pan": ["official_source_review", "adversarial_budget_and_inference_only_snapshot_spec"],
    "rp": ["oof_plus_refit_budget", "estimated_prior_degeneracy_handling"],
    "pulns": [
        "independent_clean_support_budget_and_five_role_isolation",
        "PA_ineligible_protocol_decision",
        "native_TS_vs_uncalibrated_OS_adapter_decision",
        "sequential_policy_probe_and_classifier_budget",
    ],
    "genpu": [
        "paper_minimax_vs_author_nonsaturating_demo_decision",
        "prior_weighted_Du_and_synthetic_PN_spec",
        "source_license_review",
        "six_network_GAN_plus_PN_selection_and_storage_spec",
    ],
    "holistic_pu": [
        "paper_pairwise_variance_vs_author_adjacent_Jenks_decision",
        "fixed_warmup_vs_LZO_and_finetune_spec",
        "source_license_review",
        "trajectory_and_model_selection_storage_spec",
    ],
}
SNAPSHOT_KEYS = (
    "native_sampling_assumption",
    "run_view",
    "calibration_applied",
    "prior_semantics",
    "adaptation_level",
    "code_version",
    "modality_backbone",
)


def build_extension(ledger, protocol):
    """Quote existing facts; refuse missing entries or admission into old pilot."""
    methods = ledger.get("methods", {})
    candidates = []
    for priority, (method, blockers) in enumerate(METHOD_BLOCKERS.items(), 1):
        if method not in methods:
            raise ValueError(f"missing ledger entry: {method}")
        if method in protocol["method_profiles"]:
            raise ValueError(f"extension method unexpectedly admitted in old protocol: {method}")
        entry = methods[method]
        missing = [key for key in SNAPSHOT_KEYS if key not in entry]
        if missing:
            raise ValueError(f"missing ledger facts for {method}: {missing}")
        if not isinstance(entry["calibration_applied"], bool):
            raise ValueError(f"calibration_applied must be boolean: {method}")
        candidates.append(
            {
                "method": method,
                "preparation_priority": priority,
                "ledger_ref": f"pu_toolbox/experiment/method_ledger.json#methods.{method}",
                "ledger_snapshot": {
                    key: entry[key]
                    for key in (
                        *SNAPSHOT_KEYS,
                        "calibration_hooked",
                        "requires_clean_support",
                        "pa_eligible",
                    )
                    if key in entry
                },
                "candidate_protocol_ref": None,
                "budget_ref": None,
                "candidate_pool": None,
                "admitted": False,
                "owner_decisions": {},
                "specific_blockers": list(blockers),
                "storage_evidence_ref": (
                    "docs/research/pu_survey/data/candidate_storage_probe_20261005.json"
                ),
                "gpu_smoke": "not_applicable_cpu"
                if method == "rp"
                else "synthetic_current_environment_only",
                "gpu_evidence_ref": None
                if method == "rp"
                else "docs/research/pu_survey/independent_progress_20261005.md",
            }
        )
    # Roundtrip yields independent plain-JSON snapshots, never ledger aliases.
    return json.loads(
        json.dumps(
            {
                "schema_version": "survey-p3-preintegration-extension-draft-1",
                "review_status": "pending_method_owner_review",
                "purpose": (
                    "Technical handoff only; no execution authorization or human signatures."
                ),
                "frozen_protocol_sha256": digest(protocol),
                "declared_scope": (
                    "synthetic dense/frozen-feature and PAN/Holistic-PU injected-CNN "
                    "engineering paths; not paper image reproduction"
                ),
                "common_blockers": [
                    "versioned_execution_and_comparison_preregistration",
                    "approved_budget_and_PA_OA_selection_spec",
                    "formal_representation_and_resource_profile",
                    "method_owner_review",
                ],
                "candidates": candidates,
                "unregistered_components": [
                    {
                        "method": "p3mix",
                        "component_ref": "pu_toolbox/estimators/deep/p3mix.py",
                        "registered": False,
                        "blockers": ["full_primary_source", "epoch_pool_schedule", "E_C_variants"],
                    }
                ],
            },
            allow_nan=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pu_toolbox/experiment/method_ledger.json",
    )
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    print(json.dumps(build_extension(ledger, load_protocol()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
