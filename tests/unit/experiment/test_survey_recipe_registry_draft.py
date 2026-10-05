"""Preparation must register existing declarations without promoting their status."""

import copy
import json
from pathlib import Path

import pytest

from pu_toolbox.experiment.survey_protocol import digest, load_protocol
from pu_toolbox.experiment.survey_recipe_registry import (
    build_draft_registry,
    canonical_digest,
    resolve_candidate,
    validate_manifest_binding,
    validate_registry,
)

pytestmark = pytest.mark.unit


def test_draft_covers_every_declared_profile_and_validates():
    protocol = load_protocol()
    registry = build_draft_registry(protocol)
    assert set(registry["profiles"]) == set(protocol["method_profiles"])
    assert validate_registry(registry, protocol) == []
    assert registry["protocol_basis"]["protocol_sha256"] == digest(protocol)


def test_draft_does_not_mutate_or_promote_protocol():
    protocol = load_protocol()
    original = copy.deepcopy(protocol)
    registry = build_draft_registry(protocol)
    assert protocol == original
    assert registry["recipe_lifecycle"] == "draft"
    for method, profile in registry["profiles"].items():
        assert profile["recipe_lifecycle"] == ("excluded" if method == "kldce" else "draft")
        assert not profile["review"]["reviewers"]
        for candidate in profile["candidate_pool"] or []:
            assert candidate["resolved_snapshot"] is None
            assert candidate["overrides"] == {}


def test_edge_excluded_method_has_reason_and_none_candidate_pool():
    profile = build_draft_registry(load_protocol())["profiles"]["kldce"]
    assert profile["candidate_pool"] is None
    assert "flip_probability" in profile["exclusion_reason"]


def test_inner_search_is_separate_from_outer_candidate():
    protocol = load_protocol()
    profile = build_draft_registry(protocol)["profiles"]["pusb_kernel"]
    assert len(profile["candidate_pool"]) == 1
    assert profile["inner_search"]["grid_ref"].endswith("pusb_kernel.params")
    resolved = resolve_candidate(profile, profile["candidate_pool"][0], protocol)
    assert resolved["budget_declaration"]["internal_hyperparameter_pairs"] == 72


def test_oracle_is_separate_and_historical_manifests_remain_unbound():
    registry = build_draft_registry(load_protocol())
    assert registry["profiles"]["pn_oracle"]["recipe_family"] == "oracle"
    manifest = {"method": "nnpu", "execution_mode": "versioned_pilot"}
    before = copy.deepcopy(manifest)
    assert validate_manifest_binding(manifest, registry) == []
    assert manifest == before


def test_determ_generated_draft_matches_checked_in_artifact():
    root = Path(__file__).resolve().parents[3]
    recorded = json.loads(
        (root / "docs/research/pu_survey/data/survey_recipe_registry_v0_1_draft.json").read_text(
            encoding="utf-8"
        )
    )
    generated = build_draft_registry(load_protocol())
    assert generated == recorded
    assert canonical_digest(generated) == canonical_digest(recorded)
    for evidence in recorded["source_evidence"].values():
        assert (root / evidence["file"]).is_file()


@pytest.mark.parametrize("candidates", [[], [{"learning_rate": 0.01}], [None], [False]])
def test_param_unsupported_candidates_fail_closed(candidates):
    protocol = copy.deepcopy(load_protocol())
    protocol["candidate_pool"] = candidates
    with pytest.raises(ValueError, match="override-free"):
        build_draft_registry(protocol)


def test_method_without_scope_cannot_be_invented():
    protocol = copy.deepcopy(load_protocol())
    protocol["execution_units"] = [r for r in protocol["execution_units"] if r["method"] != "nnpu"]
    with pytest.raises(ValueError, match="no declared execution scope"):
        build_draft_registry(protocol)


def test_budget_candidate_mismatch_is_not_silently_registered():
    protocol = copy.deepcopy(load_protocol())
    protocol["budgets"]["minibatch"]["outer_candidates"] = 2
    with pytest.raises(ValueError, match="failed validation"):
        build_draft_registry(protocol)
