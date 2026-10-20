"""Offline registration contracts for the unsigned LaGAM full-paper follow-up."""

import json
from pathlib import Path

import pytest

from pu_toolbox.experiment.method_ledger import load_ledger

pytestmark = pytest.mark.contract
_ROOT = Path(__file__).resolve().parents[2]
_REF = "docs/research/pu_survey/data/lagam_paper_review_20261020.json"


@pytest.fixture
def receipt():
    return json.loads((_ROOT / _REF).read_text(encoding="utf-8"))


def test_paper_followup_preserves_prior_retrieval_and_consistent_pinned_code_identity(receipt):
    previous = json.loads((_ROOT / receipt["previous_review_ref"]).read_text(encoding="utf-8"))
    assert "HTTP403" in previous["source"]["paper_retrieval_this_review"]
    assert previous["paper_followup_ref"] == _REF
    upstream = receipt["sources"]["upstream"]
    assert upstream["commit"] == previous["source"]["commit"]
    assert upstream["repository"] == previous["source"]["repository"]
    for name, digest in upstream["file_sha256"].items():
        assert digest == previous["source"]["file_sha256"][name]
    assert not upstream["executed_upstream_code"] and not upstream["copied_source_code"]
    paper = receipt["sources"]["paper"]
    supplement = receipt["sources"]["supplement"]
    assert paper["pdf_pages"] == 10 and supplement["pdf_pages"] == 7
    assert paper["sha256"] == "31c76526cfa5797d836e2b0108e385334902d8bdc77043658efaea4f887f1420"
    assert (
        supplement["sha256"] == "989d49b00699e6d1796c5498a2ff5e9a3a3aa96bea29183c65231e3cc2ea6378"
    )


def test_basic_ledger_and_method_card_resolve_to_unsigned_paper_crosswalk(receipt):
    entry = load_ledger()["methods"]["lagam"]
    assert entry["source_behavior_review_ref"] == receipt["previous_review_ref"]
    assert entry["source_protocol_review_ref"] == _REF
    review = (_ROOT / receipt["review_ref"]).read_text(encoding="utf-8")
    assert "pending_method_owner_review" in review
    assert "式 (10)" in review and "Table 15" in review and "README.md:9,15" in review
    assert (_ROOT / "docs/research/method_cards/LaGAM.md").read_text(encoding="utf-8").find(
        "lagam_paper_review_20261020.md"
    ) >= 0
    assert receipt["review_status"] == "pending_method_owner_review"
    assert receipt["owner_signatures"] == []
    assert entry["run_view"] == receipt["boundaries"]["run_view"] == "os-compatible"
    for field in ("calibration_applied", "calibration_hooked"):
        assert entry[field] is receipt["boundaries"][field] is False
    assert receipt["boundaries"]["requires_clean_support"] is True
    for field in (
        "formal_admission",
        "pa_eligible_under_frozen_protocol",
        "frozen_comparison_modified",
        "historical_results_modified",
        "protocol_draft_accepted",
        "numerical_reproduction_claimed",
    ):
        assert receipt["boundaries"][field] is False
    assert receipt["boundaries"]["candidate_protocol_ref"] is None
    assert receipt["boundaries"]["budget_ref"] is None


def test_param_zero_boundary_and_recipe_conflicts_remain_distinct_from_defaults(receipt):
    findings = {item["id"]: item for item in receipt["findings"]}
    assert len(findings) == len(receipt["findings"]) == 7
    zero = findings["zero_gradient_projection"]
    assert "delta>=0" in zero["paper_value"] and "delta>0" in zero["paper_value"]
    assert "zero chooses0" in zero["toolbox_value"]
    beta = findings["contrastive_coefficient"]
    assert "beta=0" in beta["paper_value"]
    assert "coefficient1" in beta["code_value"]
    assert "no_inferred_beta_mapping" in beta["verdict"]
    clusters = findings["cluster_count"]
    assert "100 clusters" in clusters["paper_value"]
    assert "override5" in clusters["code_value"]
    assert "default5" in clusters["toolbox_value"]
    assert (
        "method_owner_resolution_of_zero_boundary_beta_and_cluster_recipe"
        in receipt["remaining_blockers"]
    )
    previous = json.loads((_ROOT / receipt["previous_review_ref"]).read_text(encoding="utf-8"))
    assert previous["toolbox_defaults"]["contrastive_weight"] == 1
    assert previous["toolbox_defaults"]["num_clusters"] == 5


def test_cifar_split_arithmetic_is_conditional_derivation_not_execution(receipt):
    split = receipt["cifar10_released_split"]
    assert split["support_per_original_class"] * split["original_class_count"] == 500
    assert split["support_positive_count"] == 50 * len(split["positive_original_classes"]) == 200
    assert split["support_negative_count"] == 500 - 200 == 300
    assert split["labeled_per_positive_original_class"] * 4 == split["labeled_positive_count"]
    assert (
        split["original_train_count"] - split["support_count"] == split["remaining_pu_train_count"]
    )
    assert (
        split["remaining_pu_train_count"] - split["labeled_positive_count"]
        == split["unlabeled_count"]
    )
    assert split["observed_pilot_or_author_result"] is False
    assert split["view"] == "os"
