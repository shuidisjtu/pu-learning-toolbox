"""The frozen registry is the runnable scope of v3, and it must stay recomputable."""

import json
from pathlib import Path

import pytest

from pu_toolbox.experiment.survey_protocol import load_protocol, resolve_protocol_path
from pu_toolbox.experiment.survey_recipe_registry import (
    bind_recipe_candidate,
    build_draft_registry,
    build_locked_registry,
    canonical_digest,
    profile_id,
    resolve_candidate,
    validate_registry,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "pu_toolbox/experiment/survey_recipe_registry_v1.json"


def protocol():
    return load_protocol(resolve_protocol_path("survey-v3.0"))


def committed():
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def runnable_methods():
    return {row["method"] for row in protocol()["execution_units"] if row["runnable"]}


def methods_with_lifecycle(registry, state):
    return {
        key for key, value in registry["profiles"].items() if value["recipe_lifecycle"] == state
    }


def test_basic_committed_registry_locks_exactly_the_runnable_profiles():
    registry = committed()
    assert registry["recipe_registry_version"] == "survey-recipes-v1.0"
    assert registry["recipe_lifecycle"] == "locked"
    locked = methods_with_lifecycle(registry, "locked")
    excluded = methods_with_lifecycle(registry, "excluded")
    assert locked == runnable_methods()
    assert excluded == set(registry["profiles"]) - locked


def test_param_every_locked_candidate_carries_a_recomputable_snapshot():
    registry = committed()
    loaded = protocol()
    for profile in registry["profiles"].values():
        if profile["recipe_lifecycle"] != "locked":
            continue
        for candidate in profile["candidate_pool"]:
            assert candidate["resolved_snapshot"] == resolve_candidate(profile, candidate, loaded)


def test_param_excluded_profiles_state_why_and_carry_no_pool():
    for profile in committed()["profiles"].values():
        if profile["recipe_lifecycle"] != "excluded":
            continue
        assert profile["candidate_pool"] is None
        assert profile["exclusion_reason"]


def test_determ_generated_registry_matches_the_committed_artifact():
    generated = build_locked_registry(protocol())
    recorded = committed()
    assert generated == recorded
    assert canonical_digest(generated) == canonical_digest(recorded)
    assert validate_registry(recorded, protocol()) == []


def test_basic_locked_registry_binds_a_runnable_profile():
    loaded = protocol()
    registry = build_locked_registry(loaded)
    binding = bind_recipe_candidate(
        registry, loaded, method="genpu", dataset="spambase", training_path="native_2d"
    )
    assert binding["recipe_binding_status"] == "registry_bound"
    assert binding["recipe_profile_id"] == profile_id(registry["profiles"]["genpu"])
    assert binding["resolved_params_sha256"]


def test_param_draft_registry_cannot_bind_at_runtime():
    loaded = protocol()
    with pytest.raises(ValueError, match="locked"):
        bind_recipe_candidate(
            build_draft_registry(loaded),
            loaded,
            method="genpu",
            dataset="spambase",
            training_path="native_2d",
        )


def test_edge_hand_edited_snapshot_parameter_is_refused():
    loaded = protocol()
    registry = build_locked_registry(loaded)
    snapshot = registry["profiles"]["genpu"]["candidate_pool"][0]["resolved_snapshot"]
    snapshot["params"]["hidden_dim"] = 128

    codes = {finding.code for finding in validate_registry(registry, loaded)}

    assert "resolved_snapshot_mismatch" in codes
