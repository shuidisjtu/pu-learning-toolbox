"""Preparation facts must not silently become an executable survey protocol."""

import json
from pathlib import Path

import pytest

from pu_toolbox.experiment.survey_protocol import digest, load_protocol

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]


def test_basic_p3_admission_draft_matches_ledger_and_remains_unadmitted():
    proposal = json.loads(
        (ROOT / "docs/research/pu_survey/data/p3_candidate_admission_v1_draft.json").read_text()
    )
    ledger = json.loads((ROOT / "pu_toolbox/experiment/method_ledger.json").read_text())
    protocol = load_protocol()
    candidates = proposal["candidates"]
    methods = [row["method"] for row in candidates]
    assert len(methods) == len(set(methods)) == 8
    assert proposal["review_status"] == "pending_method_owner_review"
    assert proposal["frozen_protocol_sha256"] == digest(protocol)
    assert sorted(row["preparation_priority"] for row in candidates) == list(range(1, 9))
    assert proposal["common_blockers"]
    for row in candidates:
        assert row["method"] not in protocol["method_profiles"]
        assert row["admitted"] is False
        assert row["candidate_protocol_ref"] is None
        assert row["budget_ref"] is None
        assert row["candidate_pool"] is None
        assert row["specific_blockers"]
        assert row["ledger_snapshot"] == {
            k: ledger["methods"][row["method"]][k] for k in row["ledger_snapshot"]
        }


def test_p3_special_label_and_view_restrictions_remain_explicit():
    proposal = json.loads(
        (ROOT / "docs/research/pu_survey/data/p3_candidate_admission_v1_draft.json").read_text()
    )
    rows = {row["method"]: row for row in proposal["candidates"]}
    assert "D22_PA_selection_decision" in rows["vpu"]["specific_blockers"]
    assert "alpha_U_provenance_and_finite_sample_conversion" in rows["cvir"]["specific_blockers"]
    assert "PA_ineligible_label_budget_decision" in rows["lagam"]["specific_blockers"]
    assert rows["cvir"]["ledger_snapshot"]["calibration_applied"] is False
    assert rows["lagam"]["ledger_snapshot"]["calibration_applied"] is False
    gradpu_review = json.loads(
        (ROOT / rows["gradpu"]["ledger_snapshot"]["source_protocol_review_ref"]).read_text()
    )
    # This checks handoff state, not whether the human paper readings are correct.
    assert gradpu_review["method"] == "gradpu"
    assert gradpu_review["review_status"] == "pending_method_owner_review"
    assert gradpu_review["owner_signatures"] == []
    assert gradpu_review["source_status"] == "not_found"
    decisions = gradpu_review["engineering_decisions"]
    assert decisions["retain_batchnorm_rejection"] is True
    assert decisions["TS_run_is_paper_experiment_replay"] is False
    assert decisions["numeric_anchors_added"] is False
    assert decisions["admitted"] is False
    assert decisions["candidate_protocol_ref"] is None and decisions["budget_ref"] is None
    for method in ("robust_pu", "split_pu"):
        accounting = rows[method]["ledger_snapshot"]["checkpoint_accounting"]
        assert accounting["formal_budget_approved"] is False
        assert accounting["optimizer_steps_attribute"] == "optimizer_steps_"
        assert accounting["bound_is_upper_limit"] is True


def test_basic_cnn_storage_evidence_remains_separate_from_formal_budget():
    ledger = json.loads((ROOT / "pu_toolbox/experiment/method_ledger.json").read_text())
    record = ledger["technical_resource_evidence"][0]
    assert record["formal_budget_approved"] is False and record["protocol_binding"] is None
    methods = set(record["methods"])
    assert methods == {"pulda", "gradpu", "robust_pu", "split_pu", "pan"}
    profiles = json.loads((ROOT / record["cpu_record"]).read_text())["profiles"]
    assert len(profiles) == 15
    assert {(p["method"], p["seed"]) for p in profiles} == {
        (method, seed) for method in methods for seed in (0, 1, 2)
    }
    for profile in profiles:
        assert profile["formal_budget_approved"] is False
        assert profile["device"] == "cpu" and profile["cuda_peak"] is None
        assert profile["encoder"]["matches_toolbox_default_width"] is True
        assert profile["epoch_weights_bytes"]["all_snapshots_replayed"]
        assert profile["method"] not in load_protocol()["method_profiles"]
    cuda = json.loads((ROOT / record["cuda_record"]).read_text())["profiles"]
    assert len(cuda) == 1 and cuda[0]["method"] == "pulda"
    assert record["cuda_methods"] == ["pulda"]
    assert cuda[0]["device"] == "cuda" and cuda[0]["formal_budget_approved"] is False
    assert cuda[0]["cuda_peak"]["allocated_bytes"] > 0
    assert (ROOT / record["review_ref"]).is_file()
