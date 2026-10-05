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
