"""Recorded metadata only, not independent access to the local run products."""

import json
from pathlib import Path

import pytest

from pu_toolbox.utils.serialization import strict_canonical_hash

pytestmark = pytest.mark.contract
PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/research/pu_survey/data/selfpu_ts_support_validation_20261023.json"
)


def receipt():
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_basic_two_real_runs_keep_explicit_short_budget_and_no_formal_or_accuracy_claim():
    data = receipt()
    assert not data["formal_admission"] and not data["formal_budget_approved"]
    assert (
        not data["paper_numerical_reproduction"] and not data["calibration_effectiveness_verified"]
    )
    assert data["owner_signatures"] == []
    for view, run in data["runs"].items():
        facts = run["training_provenance"]
        assert run["training_view"] == view and run["calibration_applied"] is (view == "ts")
        assert run["selected_model_replay_verified"] and run["accepted_engineering_run"]
        assert run["warning_count"] == run["retry_count"] == 0
        assert (
            facts["calibration_mode"] == "clean_validation_meta"
            and facts["active_meta_reweight_records"] == 6
        )
        assert facts["executed_epochs"] == 3 and facts["distillation_scope"] == "paper_population"
        assert not facts["pa_eligible"] and facts["pa_clean_support_eligible"]
        assert not run["manifest_formal_eligible"] and not run["artifacts_in_git"]


def test_param_paired_data_marking_params_and_clean_label_budget_stay_equal():
    os, ts = (receipt()["runs"][view] for view in ("os", "ts"))
    for key in (
        "product_receipt_sha256",
        "parameters_effective",
        "seed",
        "c_token",
        "population_prior",
    ):
        assert os[key] == ts[key]
    assert os["generation"] == ts["generation"]
    for run in (os, ts):
        facts = run["training_provenance"]
        assert facts["support_samples"] == 100
        assert facts["support_positive_labels"] + facts["support_negative_labels"] == 100
        assert facts["original_train_count"] - facts["remaining_train_count"] == 100
        assert facts["total_observed_training_label_budget"] == 630 + 100
        assert facts["all_role_isolation_verified"] and facts["support_removed_from_train"]
        assert not facts["raw_label_mapping_verified"] and not facts["support_source_verified"]
    assert ts["training_provenance"]["student_optimizer_steps"] == {"1": 3, "2": 3}
    assert ts["training_provenance"]["optimizer_steps"] == 6
    assert not ts["training_provenance"]["population_mixture_restoration_verified"]


def test_edge_selected_storage_does_not_hide_peak_or_two_teacher_attempt_budget():
    for run in receipt()["runs"].values():
        storage = run["checkpoint_storage"]
        assert (
            storage["retained_count"] + storage["reclaimed_count"] == run["checkpoint_count"] == 6
        )
        assert storage["peak_payload_bytes"] == 17616 > storage["retained_bytes"] == 2936
        assert run["resources"]["checkpoint_disk_preflight"]["required_bytes"] == 65536 * 3 * 2 * 2
        assert not run["frozen_lock_environment_verified"]


def test_determ_canonical_receipt_identity_and_file_hashes_are_well_formed():
    data = receipt()
    digest = data.pop("receipt_sha256")
    assert strict_canonical_hash(data) == digest
    for run in data["runs"].values():
        assert len(run["plan_payload_sha256"]) == 64
        assert all(len(value) == 64 for value in run["files_sha256"].values())
