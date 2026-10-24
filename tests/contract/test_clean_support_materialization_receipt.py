"""Metadata regressions only; raw products are local and unavailable in CI."""

import json
import re
from pathlib import Path

import pytest

from pu_toolbox.utils.serialization import strict_canonical_hash

pytestmark = pytest.mark.contract
ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "docs/research/pu_survey/data/clean_support_materialization_20261022.json"
REBUILD = ROOT / "docs/research/pu_survey/data/clean_support_materialization_20261024.json"


def receipt():
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def rebuild_receipt():
    return json.loads(REBUILD.read_text(encoding="utf-8"))


def test_basic_coverage_is_exactly_five_datasets_times_five_seeds():
    data = receipt()
    expected = {(dataset, seed) for dataset in data["datasets"] for seed in range(5)}
    observed = [(row["dataset"], row["seed"]) for row in data["products"]]
    assert len(data["datasets"]) == len(set(data["datasets"])) == 5
    assert len(observed) == len(set(observed)) == 25
    assert set(observed) == expected
    assert data["coverage"] == {
        "expected_products": 25,
        "observed_products": 25,
        "complete": True,
        "missing": [],
    }
    assert data["product_total_bytes"] == sum(row["bytes"] for row in data["products"])
    assert data["artifact_location"] == "this_server_only_not_in_git"


def test_param_role_sizes_label_budgets_and_seed_policy_are_consistent():
    data = receipt()
    assert data["support_sizes"] == {
        "spambase": 100,
        "connect_4": 500,
        "mnist": 500,
        "fashion_mnist": 500,
        "cifar10": 500,
    }
    for row in data["products"]:
        assert row["seed"] == row["support_seed"]
        assert set(row["role_sizes"]) == {"train", "pu_val", "clean_val", "test", "support"}
        assert all(type(count) is int and count > 0 for count in row["role_sizes"].values())
        size = data["support_sizes"][row["dataset"]]
        assert row["role_sizes"]["support"] == size
        assert row["support_positive"] > 0 and row["support_negative"] > 0
        assert row["support_positive"] + row["support_negative"] == size
    assert data["validation"]["canonical_cli_reproduction"].startswith("spambase seed0:")


def test_edge_real_cpu_smokes_never_become_formal_gpu_or_pure_pa_evidence():
    data = receipt()
    assert data["formal_admission"] is data["formal_budget_approved"] is False
    assert data["owner_signatures"] == []
    assert data["protocol_binding"] == "policy_basis_only_not_a_versioned_execution_unit"
    assert set(data["protocol_pending_review_items"]) == {
        "v2_p4_clean_support_policy",
        "v2_p10_self_pu_scope",
    }
    smoke = data["real_cpu_smoke"]
    assert smoke["formal_admission"] is smoke["formal_budget_approved"] is False
    assert smoke["paper_numerical_comparison"] is False
    assert {row["method"] for row in smoke["results"]} == {"self_pu", "lagam", "pulns"}
    for row in smoke["results"]:
        assert set(row["protocols"]) == {"PA+clean-support", "OA"}
        assert row["parameters"]["device"] == "cpu"
        assert row["formal_admission"] is False
        assert row["accepted_engineering_run"] and row["selected_model_replay_verified"]
        assert row["warning_count"] == row["retry_count"] == 0


def test_determ_canonical_receipt_digest_and_relative_artifact_paths():
    data = receipt()
    identity = data.pop("receipt_sha256")
    assert strict_canonical_hash(data) == identity
    for row in data["products"]:
        directory = Path(row["directory"])
        assert not directory.is_absolute() and ".." not in directory.parts
        assert directory == Path(data["artifact_root"]) / row["dataset"] / f"split_{row['seed']}"
        for name in ("receipt_sha256", "support_manifest_file_sha256"):
            assert re.fullmatch(r"[0-9a-f]{64}", row[name])
    assert set(data["source_readers"]) == set(data["datasets"])
    cifar = data["additional_downloads"]["cifar10"]
    assert cifar["bytes"] == 170498071
    assert cifar["publisher_checksum"] == "c58f30108f718f92721af3b95e74349a"
    assert cifar["publisher_checksum_verified"] is True


def test_basic_rebuild_reproduces_membership_but_carries_new_receipts():
    baseline = receipt()
    data = rebuild_receipt()
    identity = data.pop("receipt_sha256")
    assert strict_canonical_hash(data) == identity
    assert data["rebuild_of"] == (
        "docs/research/pu_survey/data/clean_support_materialization_20261022.json"
    )
    assert data["rebuild_of_receipt_sha256"] == baseline["receipt_sha256"]
    assert data["formal_admission"] is data["formal_budget_approved"] is False
    assert data["owner_signatures"] == []
    assert data["protocol_pending_review_items"] == []
    assert "accepted in survey-v3.0" in data["protocol_pending_resolution"]
    assert data["validation"]["canonical_cli_reproduction"].startswith("spambase seed0:")
    assert data["coverage"] == {
        "expected_products": 25,
        "observed_products": 25,
        "complete": True,
        "missing": [],
    }
    assert data["product_total_bytes"] == sum(row["bytes"] for row in data["products"])
    recorded = {(row["dataset"], row["seed"]): row for row in baseline["products"]}
    observed = {(row["dataset"], row["seed"]): row for row in data["products"]}
    assert set(observed) == set(recorded) and len(observed) == 25
    for key, row in observed.items():
        old = recorded[key]
        assert row["support_seed"] == row["seed"]
        assert row["role_sizes"] == old["role_sizes"]
        assert (row["support_positive"], row["support_negative"]) == (
            old["support_positive"],
            old["support_negative"],
        )
        assert row["receipt_sha256"] != old["receipt_sha256"]
        directory = Path(row["directory"])
        assert not directory.is_absolute() and ".." not in directory.parts
        assert directory == Path(data["artifact_root"]) / row["dataset"] / f"split_{row['seed']}"
