"""Metadata consistency only; CI does not have the private local GPU artifacts."""

import json
from pathlib import Path

import pytest

from pu_toolbox.utils.serialization import strict_canonical_hash

pytestmark = pytest.mark.contract
PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/research/pu_survey/data/genpu_recipe_validation_20261023.json"
)


def receipt():
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_basic_gpu_evidence_is_explicitly_short_subset_and_not_formal_admission():
    data = receipt()
    assert data["formal_admission"] is data["formal_budget_approved"] is False
    assert data["owner_signatures"] == [] and data["artifacts_in_git"] is False
    assert data["full_dataset_or_multiseed_effectiveness_verified"] is False
    assert data["input_receipt"]["label_based_resampling"] is False
    assert [
        data["input_receipt"]["roles"][role]["count"]
        for role in ("train", "pu_val", "clean_val", "test")
    ] == [24, 8, 8, 16]
    assert data["recipe"]["parameters"]["device"] == "cuda"
    assert data["recipe"]["image_preprocessing"]["backbone"]["name"] == "resnet18"
    assert data["measurement_environment"]["frozen_lock_environment_verified"] is False
    assert data["measurement_environment"]["environment"]["cuda_available"] is True


def test_param_two_batch_profiles_preserve_input_identity_and_pay_all_six_stages():
    data = receipt()
    other = data["batch256_profile"]
    for profile, steps in ((data, 3), (other, 1)):
        parameters = profile["recipe"]["parameters"]
        assert parameters["max_epochs"] == 1 and parameters["classifier_epochs"] == 3
        assert (
            sum(profile["stage_optimizer_steps"].values())
            == profile["optimizer_steps"]
            == steps * 8
        )
        assert profile["stage_optimizer_steps"]["synthetic_pn"] == steps * 3
        assert profile["selected_models_replayed"] and profile["accepted_engineering_run"]
        assert profile["warning_count"] == profile["retry_count"] == 0
    assert data["recipe"]["parameters"]["batch_size"] == 8
    assert other["recipe"]["parameters"]["batch_size"] == 256
    assert other["input_files_sha256"] == {
        role: value["sha256"] for role, value in data["input_receipt"]["roles"].items()
    }


def test_edge_selected_weight_storage_does_not_replace_peak_or_gpu_profile_scope():
    data = receipt()
    assert len(data["six_network_state_bytes"]) == 6
    assert sum(data["checkpoint_bytes"]) == data["checkpoint_peak_payload_bytes"]
    for profile in (data, data["batch256_profile"]):
        assert profile["checkpoint_peak_payload_bytes"] > profile["retained_checkpoint_bytes"] > 0
        assert profile["retained_checkpoint_count"] + profile["reclaimed_checkpoint_count"] == 3
        assert profile["retained_checkpoint_count"] <= 2
        assert profile["fit_peak_cuda_allocated_bytes"] > 0
        assert profile["process_lifetime_peak_rss_bytes"] > 0
    assert data["disk_preflight"]["required_bytes"] == 64 * 2**20 * 3 * 2
    assert "torch_allocated_not_reserved" in data["peak_scope"]


def test_determ_receipt_and_recipe_canonical_digests_are_self_consistent():
    data = receipt()
    identity = data.pop("receipt_sha256")
    assert strict_canonical_hash(data) == identity
    for recipe in (data["recipe"], data["batch256_profile"]["recipe"]):
        identity = recipe.pop("recipe_sha256")
        assert strict_canonical_hash(recipe) == identity
