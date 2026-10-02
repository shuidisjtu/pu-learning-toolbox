# tests/unit/experiment/test_survey_recipe_registry.py
"""Schema, candidate-pool and digest rules for the P4.1 recipe registry.

The registry's whole claim is that identity is registered *before* results and
cannot be reworded afterwards, so these tests are about what the schema refuses
-- a parameter override, a scope dimension the matrix does not declare, a test
split leaking into a candidate -- and about the digest moving for exactly the
fields that carry identity.
"""

from __future__ import annotations

import json

import pytest
from _survey_recipe_registry_helpers import (
    base_registry,
    candidate,
    codes,
    excluded_profile,
    inner_search,
    profile,
    protocol_with_outer_candidates,
)

from pu_toolbox.experiment.survey_protocol import load_protocol
from pu_toolbox.experiment.survey_recipe_registry import (
    CANDIDATE_ROLES,
    SCHEMA_VERSION,
    canonical_digest,
    load_registry,
    validate_registry,
)

pytestmark = pytest.mark.unit


def test_basic_single_default_candidate_validates_clean():
    assert validate_registry(base_registry(), load_protocol()) == []


def test_basic_multiple_candidates_validate_and_keep_declaration_order():
    protocol = protocol_with_outer_candidates(2)
    value = base_registry(
        protocol=protocol,
        profiles={
            "nnpu": profile(
                "nnpu",
                candidate_pool=[candidate("nnpu_default_v1"), candidate("nnpu_alt_v1")],
            )
        },
    )
    assert validate_registry(value, protocol) == []
    pool = value["profiles"]["nnpu"]["candidate_pool"]
    assert [item["recipe_candidate_id"] for item in pool] == [
        "nnpu_default_v1",
        "nnpu_alt_v1",
    ]


def test_basic_registry_without_profiles_is_not_a_registry(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid survey recipe registry"):
        load_registry(path)


def test_param_inner_search_is_registered_as_identity_and_enforced_both_ways():
    # 72 is the protocol's number, declared and checked there; the registry names
    # the search and points at the grid rather than restating either.
    params = load_protocol()["method_profiles"]["pusb_kernel"]["params"]
    assert (len(params["sigma_grid"]), len(params["reg_grid"])) == (9, 8)
    assert load_protocol()["budgets"]["kernel_cv"]["internal_hyperparameter_pairs"] == 72

    clean = base_registry(
        profiles={"pusb_kernel": profile("pusb_kernel", inner_search=inner_search())}
    )
    assert validate_registry(clean, load_protocol()) == []

    wrong_grid = base_registry(
        profiles={
            "pusb_kernel": profile(
                "pusb_kernel",
                inner_search={
                    "inner_search_id": "pusb_kernel_sigma_reg_cv",
                    "grid_ref": "survey_protocol_v1.method_profiles.lbe.params",
                },
            )
        }
    )
    assert codes(validate_registry(wrong_grid, load_protocol())) == ["unknown_grid_ref"]

    unregistered = base_registry(profiles={"pusb_kernel": profile("pusb_kernel")})
    assert codes(validate_registry(unregistered, load_protocol())) == ["missing_inner_search"]


def test_param_candidate_count_must_match_the_declared_outer_candidates():
    value = base_registry(
        profiles={
            "nnpu": profile(
                "nnpu",
                candidate_pool=[candidate("nnpu_default_v1"), candidate("nnpu_alt_v1")],
            )
        }
    )
    assert "candidate_count_mismatch" in codes(validate_registry(value, load_protocol()))


def test_param_candidate_role_is_closed_to_the_v1_value():
    assert CANDIDATE_ROLES == ("current_protocol_binding",)
    value = base_registry(
        profiles={
            "nnpu": profile(
                "nnpu",
                candidate_pool=[candidate("nnpu_default_v1", candidate_role="source_recipe")],
            )
        }
    )
    assert codes(validate_registry(value, load_protocol())) == ["unknown_candidate_role"]


def test_edge_duplicate_recipe_candidate_id_is_reported():
    value = base_registry(
        profiles={
            "nnpu": profile("nnpu", candidate_pool=[candidate("shared_id_v1")]),
            "lbe": profile("lbe", candidate_pool=[candidate("shared_id_v1")]),
        }
    )
    assert "duplicate_candidate_id" in codes(validate_registry(value, load_protocol()))


def test_edge_parameter_overrides_are_rejected_at_both_levels():
    fixed = base_registry(profiles={"nnpu": profile("nnpu", fixed_overrides={"max_epochs": 999})})
    assert "fixed_overrides_forbidden" in codes(validate_registry(fixed, load_protocol()))
    per_candidate = base_registry(
        profiles={
            "nnpu": profile(
                "nnpu",
                candidate_pool=[candidate("nnpu_default_v1", overrides={"max_epochs": 999})],
            )
        }
    )
    assert "candidate_overrides_forbidden" in codes(
        validate_registry(per_candidate, load_protocol())
    )


def test_edge_test_truth_key_in_a_candidate_is_reported():
    value = base_registry(
        profiles={
            "nnpu": profile(
                "nnpu",
                candidate_pool=[
                    candidate("nnpu_default_v1", resolved_snapshot={"test_truth": [1, 0]})
                ],
            )
        }
    )
    assert "test_truth_in_candidate" in codes(validate_registry(value, load_protocol()))


def test_edge_unknown_schema_version_is_reported():
    value = base_registry(schema_version="survey-recipe-registry-9.9")
    assert "unsupported_schema_version" in codes(validate_registry(value, load_protocol()))


def test_edge_null_candidate_pool_and_exclusion_reason_move_together():
    # A null pool outside the excluded lifecycle, with no recorded reason.
    unjustified = base_registry(profiles={"nnpu": profile("nnpu", candidate_pool=None)})
    assert codes(validate_registry(unjustified, load_protocol())) == [
        "candidate_pool_null_not_excluded",
        "missing_exclusion_reason",
    ]
    # An excluded profile whose protocol rows are still runnable.
    contradictory = base_registry(profiles={"nnpu": excluded_profile("nnpu")})
    assert codes(validate_registry(contradictory, load_protocol())) == [
        "excluded_method_still_runnable"
    ]
    # A reason left behind on a profile that went back to carrying candidates.
    stale = base_registry(
        profiles={"nnpu": profile("nnpu", exclusion_reason="was excluded last month")}
    )
    assert codes(validate_registry(stale, load_protocol())) == ["unexpected_exclusion_reason"]
    # And the legal combination -- excluded, no pool, with its reason -- is clean.
    excluded = base_registry(profiles={"kldce": excluded_profile("kldce")})
    assert validate_registry(excluded, load_protocol()) == []
    assert excluded["profiles"]["kldce"]["candidate_pool"] is None


def test_edge_scope_outside_the_execution_matrix_is_reported():
    outside = base_registry(
        profiles={
            "nnpu": profile(
                "nnpu",
                scope={"datasets": ["spambase", "mnist"], "training_paths": ["native_2d"]},
            )
        }
    )
    assert "scope_not_in_protocol" in codes(validate_registry(outside, load_protocol()))
    with_run_view = base_registry(
        profiles={
            "nnpu": profile(
                "nnpu",
                scope={
                    "datasets": ["spambase"],
                    "training_paths": ["native_2d"],
                    "run_view": "ts-compatible",
                },
            )
        }
    )
    assert "scope_contains_run_view" in codes(validate_registry(with_run_view, load_protocol()))


def test_determ_canonical_digest_survives_a_serialization_round_trip():
    value = base_registry()
    assert canonical_digest(value) == canonical_digest(json.loads(json.dumps(value)))


def test_determ_digest_moves_for_identity_and_ignores_provenance():
    value = base_registry()
    baseline = canonical_digest(value)

    provenance = json.loads(json.dumps(value))
    provenance["source_evidence"]["project_method_ledger_nnpu"]["file"] = "elsewhere.json"
    provenance["profiles"]["nnpu"]["review"]["reviewers"] = ["someone"]
    provenance["profiles"]["nnpu"]["source_refs"] = []
    assert canonical_digest(provenance) == baseline

    lifecycle = json.loads(json.dumps(value))
    lifecycle["profiles"]["nnpu"]["recipe_lifecycle"] = "under_review"
    assert canonical_digest(lifecycle) != baseline

    candidate_id = json.loads(json.dumps(value))
    candidate_id["profiles"]["nnpu"]["candidate_pool"][0]["recipe_candidate_id"] = "renamed_v1"
    assert canonical_digest(candidate_id) != baseline
