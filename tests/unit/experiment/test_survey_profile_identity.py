# ruff: noqa: N803
"""Versioned identity mappings, true native classes and fail-closed protocols."""

import copy

import pytest
from _profile_identity_helpers import draft, write_matrix

from pu_toolbox.core.exceptions import RegistryError
from pu_toolbox.experiment.survey_execution import assemble_model
from pu_toolbox.experiment.survey_profile_identity import (
    FIELD,
    SCHEMA,
    profile_algorithm,
    profile_manifest_identity,
    require_four_role_profile,
    validate_profile_identities,
)
from pu_toolbox.experiment.survey_protocol import digest, load_protocol
from pu_toolbox.experiment.survey_recipe_binding import bind_recipe
from pu_toolbox.experiment.survey_recipe_registry import (
    build_draft_registry,
    resolve_candidate,
    validate_registry,
)
from pu_toolbox.registry import get_algorithm, get_algorithm_registry

pytestmark = pytest.mark.unit


def test_basic_explicit_mapping_is_not_a_registered_algorithm_alias(tmp_path):
    protocol = draft()
    assert load_protocol(write_matrix(tmp_path, protocol)) == protocol
    assert profile_algorithm(protocol, "upu_recipe") == "upu"
    assert profile_manifest_identity(protocol, "upu_recipe") == {
        FIELD: SCHEMA,
        "profile_id": "upu_recipe",
        "algorithm_id": "upu",
    }
    assert "upu_recipe" not in get_algorithm_registry()
    with pytest.raises(RegistryError, match="Unknown"):
        get_algorithm("upu_recipe")


def test_determ_legacy_paths_keep_identity_and_protocol_bytes_without_extra_fields():
    protocol = load_protocol()
    original = copy.deepcopy(protocol)
    assert profile_algorithm(protocol, "self_pu") == "self_pu"
    assert profile_algorithm({}, "nnpu") == "nnpu"
    assert profile_manifest_identity(protocol, "self_pu") == {}
    validate_profile_identities(protocol)
    assert protocol == original and digest(protocol) == digest(original)


@pytest.mark.parametrize(
    "fault",
    [
        "no_marker",
        "old_version",
        "bad_schema",
        "missing_algorithm",
        "unknown_algorithm",
        "shadow",
        "oracle_alias",
        "path",
        "prior_estimator",
        "shadow_alias",
    ],
)
def test_param_invalid_or_ambiguous_new_mappings_fail_before_assembly(fault, tmp_path):
    protocol = draft()
    profile = protocol["method_profiles"]["upu_recipe"]
    if fault == "no_marker":
        del protocol[FIELD]
    elif fault == "old_version":
        protocol["protocol_version"] = "survey-v2.0"
    elif fault == "bad_schema":
        protocol[FIELD] = True
    elif fault == "missing_algorithm":
        del profile["algorithm"]
    elif fault == "unknown_algorithm":
        profile["algorithm"] = "ghost"
    elif fault == "shadow":
        protocol["method_profiles"] = {"nnpu": profile}
    elif fault == "oracle_alias":
        profile["algorithm"] = "pn_oracle"
    elif fault == "path":
        protocol["method_profiles"] = {"../escape": profile}
    elif fault == "prior_estimator":
        profile["algorithm"] = "tice"
    else:
        profile["algorithm"] = "nnpu"
        protocol["method_profiles"] = {"unbiased_pu": profile}
    with pytest.raises(ValueError):
        load_protocol(write_matrix(tmp_path, protocol))


@pytest.mark.parametrize("seed", [0, 7])
def test_determ_nnpu_recipe_assembly_has_the_exact_legacy_network_weights(seed):
    torch = pytest.importorskip("torch")
    legacy = load_protocol()
    old_row = next(
        row
        for row in legacy["execution_units"]
        if row["dataset"] == "spambase" and row["method"] == "nnpu"
    )
    new = draft("nnpu")
    left = assemble_model(legacy, old_row, 3, seed=seed, params={}, class_prior=0.4, device="cpu")
    right = assemble_model(
        new, new["execution_units"][0], 3, seed=seed, params={}, class_prior=0.4, device="cpu"
    )
    assert type(left) is type(right) is get_algorithm("nnpu")
    assert left.max_epochs == 200 and right.max_epochs == 2
    for name, value in left.model.state_dict().items():
        torch.testing.assert_close(value, right.model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "fault",
    [
        "ablation",
        "other_algorithm",
        "missing_support",
        "wrong_consistency",
        "nonbool",
        "old_profile",
    ],
)
def test_param_full_meta_name_and_semantics_cannot_masquerade_as_old_ablation(fault):
    protocol = draft("self_pu", "self_pu_full_meta")
    profile = protocol["method_profiles"]["self_pu_full_meta"]
    profile.update(supervision="independent_clean_support")
    profile["params"].update(require_validation=True, distillation_scope="paper_population")
    if fault == "ablation":
        profile["params"]["require_validation"] = False
    elif fault == "other_algorithm":
        profile["algorithm"] = "nnpu"
    elif fault == "missing_support":
        profile.pop("supervision")
    elif fault == "wrong_consistency":
        profile["params"]["distillation_scope"] = "untrusted_u"
    elif fault == "nonbool":
        profile["params"]["require_validation"] = 1
    else:
        protocol["method_profiles"] = {"self_pu": profile}
    with pytest.raises(ValueError):
        validate_profile_identities(protocol)


def test_edge_full_meta_class_can_be_constructed_but_not_run_via_four_roles():
    pytest.importorskip("torch")
    protocol = draft("self_pu", "self_pu_full_meta")
    profile = protocol["method_profiles"]["self_pu_full_meta"]
    profile["supervision"] = "independent_clean_support"
    profile["params"].update(require_validation=True, distillation_scope="paper_population")
    validate_profile_identities(protocol)
    model = assemble_model(
        protocol,
        protocol["execution_units"][0],
        3,
        seed=0,
        params={},
        class_prior=0.4,
        device="cpu",
    )
    assert type(model) is get_algorithm("self_pu") and model.require_validation
    with pytest.raises(ValueError, match="five-role"):
        require_four_role_profile(protocol, "self_pu_full_meta")


def test_edge_registry_v1_does_not_invent_v2_references_for_new_profiles():
    old = load_protocol()
    registry = build_draft_registry(old)
    new = draft()
    with pytest.raises(ValueError, match="registry v1"):
        build_draft_registry(new)
    findings = validate_registry(registry, new)
    assert len(findings) == 1 and findings[0].code == "protocol_basis_mismatch"
    with pytest.raises(ValueError, match="protocol_basis_mismatch"):
        bind_recipe(
            registry, new, method="upu_recipe", dataset="spambase", training_path="native_2d"
        )
    profile = registry["profiles"]["upu"]
    with pytest.raises(ValueError, match="registry v1"):
        resolve_candidate(profile, profile["candidate_pool"][0], new)
