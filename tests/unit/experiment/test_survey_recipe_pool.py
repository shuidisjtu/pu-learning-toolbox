"""Versioned pools are identities, not permission to change a frozen search."""

import copy

import pytest
from _survey_recipe_registry_helpers import inner_search, locked_registry, pool_registry

from pu_toolbox.experiment.survey_protocol import digest, load_protocol
from pu_toolbox.experiment.survey_recipe_binding import (
    SCHEMA_VERSION,
    bind_recipe,
    bind_recipe_pool,
    candidate_binding_at,
    matches_recipe_binding,
    validate_expected_binding,
)
from pu_toolbox.experiment.survey_recipe_registry import (
    bind_recipe_candidate,
    canonical_digest,
    resolve_candidate,
    validate_manifest_binding,
)

pytestmark = pytest.mark.unit


def pool_binding(count=3):
    protocol, registry = pool_registry(count)
    return (
        protocol,
        registry,
        bind_recipe_pool(
            registry,
            protocol,
            method="upu",
            dataset="spambase",
            training_path="native_2d",
        ),
    )


def pool_manifest(binding):
    """Candidate zero fails twice; PA/OA choose different surviving candidates."""
    return {
        **copy.deepcopy(binding),
        "candidate_runs": [
            {
                **candidate_binding_at(binding, index),
                "status": "excluded" if index == 0 else "succeeded",
                "trajectory_index": None if index == 0 else index - 1,
                "params": {},
            }
            for index in range(len(binding["recipe_candidates"]))
        ],
        "selection": {
            name: {**candidate_binding_at(binding, index), "run_index": index - 1}
            for name, index in (("PA", 1), ("OA", 2))
        },
    }


def test_basic_determ_pool_recomputes_without_selecting_or_mutating():
    protocol, registry, binding = pool_binding()
    before = copy.deepcopy(registry)
    assert binding["recipe_binding_schema_version"] == SCHEMA_VERSION
    assert binding["recipe_registry_sha256"] == canonical_digest(registry)
    assert binding["recipe_candidates_sha256"] == digest(binding["recipe_candidates"])
    assert "recipe_candidate_id" not in binding
    assert "resolved_params_sha256" not in binding
    assert binding == bind_recipe_pool(
        registry,
        protocol,
        method="upu",
        dataset="spambase",
        training_path="native_2d",
    )
    assert registry == before
    assert matches_recipe_binding(pool_manifest(binding), binding)


@pytest.mark.parametrize("defect", ["draft", "pin", "override", "candidates"])
def test_param_pool_preflight_does_not_relax_approval(defect):
    protocol, registry, _ = pool_binding()
    request = {}
    if defect == "draft":
        registry["recipe_lifecycle"] = "draft"
    elif defect == "pin":
        request["expected_sha256"] = "a" * 64
    elif defect == "override":
        request["constructor_overrides"] = {"new_parameter": 1}
    else:
        request["candidates"] = [{}]
    with pytest.raises(ValueError):
        bind_recipe_pool(
            registry,
            protocol,
            method="upu",
            dataset="spambase",
            training_path="native_2d",
            **request,
        )


@pytest.mark.parametrize(
    "defect",
    ["missing", "digest", "duplicate", "order", "bool_index", "scalar", "version", "truth_extra"],
)
def test_param_invalid_external_pool_is_a_caller_error(defect):
    _, _, binding = pool_binding()
    if defect == "missing":
        binding["recipe_candidates"][1].pop("inner_search_id")
    elif defect == "digest":
        binding["recipe_candidates_sha256"] = "a" * 64
    elif defect == "duplicate":
        binding["recipe_candidates"][1]["recipe_candidate_id"] = binding["recipe_candidates"][0][
            "recipe_candidate_id"
        ]
    elif defect == "order":
        binding["recipe_candidates"].reverse()
    elif defect == "bool_index":
        binding["recipe_candidates"][1]["candidate_index"] = True
    elif defect == "scalar":
        binding["recipe_candidate_id"] = "pretend_selected"
    elif defect == "version":
        binding["recipe_binding_schema_version"] = "unknown"
    else:
        binding["recipe_candidates"][1]["test_labels"] = [1, 0]
        binding["recipe_candidates_sha256"] = digest(binding["recipe_candidates"])
    with pytest.raises(ValueError):
        validate_expected_binding(binding)


@pytest.mark.parametrize(
    "defect",
    [
        "excluded",
        "trajectory",
        "wrong_id",
        "wrong_index",
        "duplicate_run",
        "missing_run",
        "all_erased",
        "root_scalar",
        "missing_null",
        "root_index_bool",
    ],
)
def test_param_bad_candidate_selection_joins_never_match(defect):
    _, _, binding = pool_binding()
    payload = pool_manifest(binding)
    if defect == "excluded":
        payload["selection"]["PA"] = {**candidate_binding_at(binding, 0), "run_index": 0}
    elif defect == "trajectory":
        payload["selection"]["OA"]["run_index"] = 2
    elif defect == "wrong_id":
        payload["selection"]["OA"]["recipe_candidate_id"] = "wrong"
    elif defect == "wrong_index":
        payload["selection"]["OA"]["candidate_index"] = True
    elif defect == "duplicate_run":
        payload["candidate_runs"][2] = copy.deepcopy(payload["candidate_runs"][1])
    elif defect == "missing_run":
        payload["candidate_runs"].pop(0)
    elif defect == "all_erased":
        payload = {
            "candidate_runs": [{"candidate_index": 0}],
            "selection": {"OA": {"candidate_index": 0}},
        }
    elif defect == "root_scalar":
        payload["recipe_candidate_id"] = binding["recipe_candidates"][2]["recipe_candidate_id"]
    elif defect == "missing_null":
        payload["candidate_runs"][1].pop("inner_search_id")
    else:
        payload["recipe_candidates"][1]["candidate_index"] = True
    assert not matches_recipe_binding(payload, binding)


def test_edge_candidate_record_order_is_not_trajectory_identity():
    _, _, binding = pool_binding()
    payload = pool_manifest(binding)
    payload["candidate_runs"].reverse()
    assert matches_recipe_binding(payload, binding)


@pytest.mark.parametrize("index", [-1, True, 3, "0"])
def test_param_invalid_candidate_lookup_is_refused(index):
    _, _, binding = pool_binding()
    with pytest.raises(ValueError, match="index"):
        candidate_binding_at(binding, index)


def test_edge_single_pool_is_explicit_and_does_not_change_scalar_defaults():
    protocol, registry = load_protocol(), locked_registry()
    request = dict(method="upu", dataset="spambase", training_path="native_2d")
    scalar = bind_recipe(registry, protocol, **request)
    assert scalar == bind_recipe_candidate(registry, protocol, **request)
    pool = bind_recipe(registry, protocol, binding_schema_version=SCHEMA_VERSION, **request)
    assert "recipe_candidate_id" not in pool
    assert len(pool["recipe_candidates"]) == 1
    assert validate_manifest_binding(pool, registry, protocol=protocol)
    payload = {
        **scalar,
        "candidate_runs": [{"candidate_index": 0, **scalar}],
        "selection": {"OA": {"candidate_index": 0, **scalar}},
    }
    assert matches_recipe_binding(payload, scalar)
    payload["recipe_binding_schema_version"] = SCHEMA_VERSION
    assert not matches_recipe_binding(payload, scalar)


def test_param_unknown_schema_is_not_silently_a_scalar_binding():
    with pytest.raises(ValueError, match="schema"):
        bind_recipe(
            locked_registry(),
            load_protocol(),
            binding_schema_version="unknown",
            method="upu",
            dataset="spambase",
            training_path="native_2d",
        )


def test_determ_candidate_return_is_an_independent_copy():
    _, _, binding = pool_binding()
    identity = candidate_binding_at(binding, 1)
    identity["recipe_candidate_id"] = "mutated"
    assert binding["recipe_candidates"][1]["recipe_candidate_id"] != "mutated"


def test_param_pool_artifact_version_does_not_authorize_parameter_search():
    protocol, registry, _ = pool_binding()
    protocol["candidate_pool"][0] = {"reg_lambda": 0.001}
    registry["protocol_basis"]["protocol_sha256"] = digest(protocol)
    with pytest.raises(ValueError, match="parameter-level"):
        bind_recipe_pool(
            registry, protocol, method="upu", dataset="spambase", training_path="native_2d"
        )


def test_edge_kernel_inner_cv_is_one_outer_pool_candidate():
    protocol, registry = pool_registry(1, "pusb_kernel")
    profile = registry["profiles"]["pusb_kernel"]
    profile["inner_search"] = inner_search()
    chosen = profile["candidate_pool"][0]
    chosen["resolved_snapshot"] = resolve_candidate(profile, chosen, protocol)
    binding = bind_recipe_pool(
        registry, protocol, method="pusb_kernel", dataset="spambase", training_path="native_2d"
    )
    assert len(binding["recipe_candidates"]) == 1
    assert binding["recipe_candidates"][0]["inner_search_id"] == "pusb_kernel_sigma_reg_cv"
    assert protocol["budgets"]["kernel_cv"]["internal_hyperparameter_pairs"] == 72
