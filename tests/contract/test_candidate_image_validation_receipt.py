"""Public metadata consistency, not independent replay of local GPU artifacts."""

import json
from pathlib import Path

import pytest

from pu_toolbox.utils.serialization import strict_canonical_hash

pytestmark = pytest.mark.contract
PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/research/pu_survey/data/candidate_image_validation_20261023.json"
)


def receipt():
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_basic_full_snapshot_counts_do_not_promote_small_subset_to_formal_run():
    data = receipt()
    assert data["formal_admission"] is data["formal_budget_approved"] is False
    assert not data["paper_numerical_reproduction"] and data["owner_signatures"] == []
    assert not data["input_receipt"]["full_survey_split"]
    assert not data["input_receipt"]["label_based_resampling"]
    for method, epochs, steps in (("pulda", 120, 360), ("gradpu", 200, 600)):
        profile = data["profiles"][method]
        assert profile["completed_epochs"] == sum(profile["stage_counts"].values()) == epochs
        assert (
            profile["optimizer_steps"]
            == profile["last_training_context"]["optimizer_steps"]
            == steps
        )
        assert (
            not profile["manifest_formal_eligible"]
            and not profile["formal_resource_profile_complete"]
        )
        assert profile["selected_models_replayed"] and profile["accepted_engineering_run"]
        assert profile["warning_count"] == profile["retry_count"] == 0
        assert profile["calibration_applied"] and profile["run_view"] == "ts-compatible"
        assert not profile["external_validation_forwarded_to_fit"]
        assert profile["role_sizes"] == dict(train=24, pu_val=8, clean_val=8, test=16)


def test_param_distinct_native_backbones_prior_routing_and_budget_semantics():
    data = receipt()
    pulda, grad = (data["profiles"][method] for method in ("pulda", "gradpu"))
    assert pulda["comparison_group"] != grad["comparison_group"]
    assert pulda["image_setup"]["image_preprocessing"]["backbone"]["name"] == "resnet18"
    assert grad["image_setup"]["image_preprocessing"]["backbone"]["name"] == "cnn13_no_bn"
    assert pulda["stage_counts"] == dict(warmup=60, pu_mixup=60)
    assert grad["stage_counts"] == dict(pu=200)
    assert grad["selection_prior_not_forwarded_to_GradPU"]
    batch = grad["native_GradPU_training"]["batch"]
    assert batch["n_positive"] == 4 and batch["n_native_unlabeled"] == 20
    assert batch["configured_per_branch_size"] == 8 and batch["steps_per_epoch"] == 3
    assert batch["observed_max_rows"]["positive"] == 4
    assert not batch["paper_batch_250_semantics_resolved"]
    assert (
        pulda["input_files_sha256"]
        == grad["input_files_sha256"]
        == {role: value["sha256"] for role, value in data["input_receipt"]["roles"].items()}
    )


def test_edge_post_selection_retention_is_not_peak_or_full_data_gpu_bound():
    for profile in receipt()["profiles"].values():
        histogram = profile["checkpoint_bytes_histogram"]
        epochs = profile["completed_epochs"]
        assert sum(histogram.values()) == epochs
        assert (
            sum(int(size) * n for size, n in histogram.items())
            == profile["checkpoint_peak_payload_bytes"]
        )
        assert (
            profile["retained_checkpoint_count"] + profile["reclaimed_checkpoint_count"] == epochs
        )
        assert 1 <= profile["retained_checkpoint_count"] <= 2
        assert profile["checkpoint_peak_payload_bytes"] > profile["retained_checkpoint_bytes"] > 0
        bound = profile["declared_bytes_per_component"]
        assert all(0 < int(size) <= bound for size in histogram)
        assert (
            profile["resources"]["checkpoint_disk_preflight"]["required_bytes"]
            == bound * epochs * 2
        )
        assert profile["resources"]["peak_gpu_memory_bytes"] > 0
        assert profile["process_lifetime_peak_rss_bytes"] > 0
        assert not profile["frozen_lock_environment_verified"] and not profile["artifacts_in_git"]


def test_determ_public_receipt_identity_binds_recorded_metadata_not_current_code():
    data = receipt()
    digest = data.pop("receipt_sha256")
    assert strict_canonical_hash(data) == digest
    for profile in data["profiles"].values():
        assert len(profile["plan_sha256"]) == 64
        assert all(len(value) == 64 for value in profile["source_files_sha256"].values())
        assert all(len(value) == 64 for value in profile["files_sha256"].values())
