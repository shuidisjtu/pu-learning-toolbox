"""Training-time recipe identities, not human sign-off or retrospective binding."""

import copy

import pytest
from _survey_recipe_registry_helpers import (
    candidate,
    locked_registry,
    protocol_with_outer_candidates,
)

from pu_toolbox.experiment.survey_protocol import load_protocol
from pu_toolbox.experiment.survey_recipe_registry import (
    bind_recipe_candidate,
    canonical_digest,
    resolve_candidate,
    validate_manifest_binding,
)

pytestmark = pytest.mark.unit


def bind(value, **kwargs):
    protocol = kwargs.pop("protocol", None) or load_protocol()
    request = {"method": "upu", "dataset": "spambase", "training_path": "native_2d"}
    request.update(kwargs)
    return bind_recipe_candidate(value, protocol, **request)


def test_basic_determ_binding_is_recomputable_and_does_not_mutate_registry():
    value = locked_registry()
    before = copy.deepcopy(value)
    result = bind(value)
    assert result == bind(value)
    assert validate_manifest_binding(result, value, protocol=load_protocol()) == []
    assert result["recipe_binding_status"] == "registry_bound"
    assert result["recipe_registry_sha256"] == canonical_digest(value)
    assert value == before


@pytest.mark.parametrize("level", ["registry", "profile"])
@pytest.mark.parametrize("status", ["draft", "under_review", "signed_off", "retired"])
def test_param_only_locked_lifecycles_can_bind(level, status):
    value = locked_registry()
    target = value if level == "registry" else value["profiles"]["upu"]
    target["recipe_lifecycle"] = status
    with pytest.raises(ValueError, match="locked"):
        bind(value)


@pytest.mark.parametrize("value", [None, [], "registry.json", {"profiles": [1]}])
def test_edge_invalid_registry_shape_is_a_readable_error(value):
    with pytest.raises(ValueError, match="object"):
        bind(value)


def test_param_scope_membership_is_checked_for_the_actual_unit():
    value = locked_registry()
    value["profiles"]["upu"]["scope"]["datasets"] = ["imdb"]
    with pytest.raises(ValueError, match="outside"):
        bind(value)


def test_param_stale_expected_digest_is_refused():
    value = locked_registry()
    assert bind(value, expected_sha256=canonical_digest(value))
    with pytest.raises(ValueError, match="changed since preflight"):
        bind(value, expected_sha256="0" * 64)


def test_param_undeclared_candidates_and_constructor_overrides_cannot_bind():
    value = locked_registry()
    with pytest.raises(ValueError, match="runtime candidates"):
        bind(value, candidates=[{"reg_lambda": 0.01}])
    with pytest.raises(ValueError, match="constructor override"):
        bind(value, constructor_overrides={"undeclared_parameter": 1})


def test_edge_singular_v1_identity_does_not_misrepresent_a_multi_candidate_search():
    protocol = protocol_with_outer_candidates(2)
    value = locked_registry("nnpu", protocol)
    prof = value["profiles"]["nnpu"]
    prof["candidate_pool"] = [candidate("first"), candidate("second")]
    for chosen in prof["candidate_pool"]:
        chosen["resolved_snapshot"] = resolve_candidate(prof, chosen, protocol)
    with pytest.raises(ValueError, match="one outer candidate"):
        bind(value, protocol=protocol, method="nnpu")


def test_basic_internal_kernel_cv_is_not_mistaken_for_an_outer_candidate_pool():
    from _survey_recipe_registry_helpers import inner_search

    protocol = load_protocol()
    value = locked_registry("pusb_kernel", protocol)
    profile = value["profiles"]["pusb_kernel"]
    profile["inner_search"] = inner_search()
    chosen = profile["candidate_pool"][0]
    chosen["resolved_snapshot"] = resolve_candidate(profile, chosen, protocol)
    result = bind(value, method="pusb_kernel")
    assert result["inner_search_id"] == "pusb_kernel_sigma_reg_cv"


def test_param_protocol_basis_mismatch_rejects_before_binding():
    # The registry binds by version; its recorded digest is provenance.
    value = locked_registry()
    value["protocol_basis"]["protocol_version"] = "survey-v1.2"
    with pytest.raises(ValueError, match="protocol_basis_mismatch"):
        bind(value)
