"""Versioned candidate-pool identity, independent of PA/OA trajectory indices.

The pool contract versions *artifacts*, not experimental permission. Registry
v1 still forbids parameter-level overrides; only an already locked protocol
and registry can supply the pool. Historical scalar bindings remain unchanged.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .survey_protocol import digest
from .survey_recipe_registry import (
    POOL_BINDING_FIELDS,
    REGISTRY_BOUND_MANIFEST_FIELDS,
    _candidate_binding,
    _prepare_recipe_profile,
    bind_recipe_candidate,
)

SCHEMA_VERSION = "survey-recipe-binding-2"
COMMON_FIELDS = (
    "recipe_binding_status",
    "recipe_registry_version",
    "recipe_registry_sha256",
    "recipe_profile_id",
)
CANDIDATE_FIELDS = ("recipe_binding_status", *REGISTRY_BOUND_MANIFEST_FIELDS)
SCALAR_ONLY_FIELDS = ("recipe_candidate_id", "resolved_params_sha256", "inner_search_id")


def bind_recipe(registry, protocol, *, binding_schema_version=None, **request):
    """Explicit artifact-version dispatch; absence preserves the scalar format."""
    if binding_schema_version is None:
        return bind_recipe_candidate(registry, protocol, **request)
    if binding_schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported recipe binding schema version")
    return bind_recipe_pool(registry, protocol, **request)


def bind_recipe_pool(registry, protocol, **request) -> dict[str, Any]:
    """Prepare an ordered pool without choosing a candidate or modifying inputs.

    Same request arguments and preflight as ``bind_recipe_candidate``. v1
    parameter-level overrides remain forbidden even in a future wider pool:
    adding a new artifact representation is not approval of a search space.
    """
    profile, sha256 = _prepare_recipe_profile(registry, protocol, **request)
    if any(params for params in protocol["candidate_pool"]):
        raise ValueError("registry v1 pool binding cannot authorize parameter-level candidates")
    identities = [
        {
            **_candidate_binding(registry, protocol, profile, chosen, sha256),
            "recipe_binding_schema_version": SCHEMA_VERSION,
            "candidate_index": index,
        }
        for index, chosen in enumerate(profile["candidate_pool"])
    ]
    binding = {
        **{field: identities[0][field] for field in COMMON_FIELDS},
        "recipe_binding_schema_version": SCHEMA_VERSION,
        "recipe_candidates": identities,
        "recipe_candidates_sha256": digest(identities),
    }
    validate_expected_binding(binding)
    return copy.deepcopy(binding)


def _flat_valid(value, *, strict: bool = False) -> bool:
    if not isinstance(value, Mapping) or value.get("recipe_binding_status") != "registry_bound":
        return False
    if any(field not in value for field in REGISTRY_BOUND_MANIFEST_FIELDS):
        return False
    if any(
        not isinstance(value[field], str) or not value[field]
        for field in REGISTRY_BOUND_MANIFEST_FIELDS
        if field != "inner_search_id"
    ):
        return False
    if strict:
        for field in ("recipe_registry_sha256", "resolved_params_sha256"):
            token = value[field]
            if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
                return False
        inner = value["inner_search_id"]
        if inner is not None and (not isinstance(inner, str) or not inner):
            return False
    return True


def validate_expected_binding(expected: Mapping[str, Any]) -> None:
    """Reject an incomplete/unknown external expectation, never infer a schema."""
    if not isinstance(expected, Mapping):
        raise ValueError("expected recipe binding must be an object")
    if not any(field in expected for field in POOL_BINDING_FIELDS):
        if not _flat_valid(expected):
            raise ValueError("expected recipe binding needs complete registry-bound identities")
        return
    if expected.get("recipe_binding_schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported recipe binding schema version")
    if any(field in expected for field in SCALAR_ONLY_FIELDS):
        raise ValueError("pool root must not claim a single selected candidate")
    pool = expected.get("recipe_candidates")
    if not isinstance(pool, list) or not pool:
        raise ValueError("pool binding needs a non-empty ordered recipe_candidates list")
    ids: set[str] = set()
    identity_keys = {*CANDIDATE_FIELDS, "recipe_binding_schema_version", "candidate_index"}
    for index, candidate in enumerate(pool):
        if (
            not _flat_valid(candidate, strict=True)
            or set(candidate) != identity_keys
            or candidate.get("recipe_binding_schema_version") != SCHEMA_VERSION
            or type(candidate.get("candidate_index")) is not int
            or candidate["candidate_index"] != index
            or any(
                field not in expected or candidate[field] != expected[field]
                for field in COMMON_FIELDS
            )
        ):
            raise ValueError("pool entries need complete consistent identities and ordered indices")
        if candidate["recipe_candidate_id"] in ids:
            raise ValueError("pool candidate identities must be unique")
        ids.add(candidate["recipe_candidate_id"])
    if expected.get("recipe_candidates_sha256") != digest(pool):
        raise ValueError("pool identity digest does not recompute")


def candidate_binding_at(binding: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Identity to write on a candidate and on its actual PA/OA selection."""
    validate_expected_binding(binding)
    if type(index) is not int or index < 0:
        raise ValueError("candidate index must be a nonnegative integer")
    if binding.get("recipe_binding_schema_version") == SCHEMA_VERSION:
        if index >= len(binding["recipe_candidates"]):
            raise ValueError("candidate index is outside the bound pool")
        return copy.deepcopy(binding["recipe_candidates"][index])
    if index != 0:
        raise ValueError("scalar recipe binding supports only candidate zero")
    return {field: copy.deepcopy(binding[field]) for field in CANDIDATE_FIELDS}


def _matches(entry, expected, fields) -> bool:
    return isinstance(entry, Mapping) and all(
        field in entry and entry[field] == expected[field] for field in fields
    )


def matches_recipe_binding(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Check root, candidate identities and selected candidate/trajectory joins.

    v2 identifies candidates by stable pool index, not list position in the
    compressed successful-trajectory list. Excluded candidates keep their
    identity but can never be selected. PA and OA need not select the same one.
    This is an identity check; numeric/role/checkpoint gates remain separate.
    """
    validate_expected_binding(expected)
    if not _matches(payload, expected, expected):
        return False
    if expected.get("recipe_binding_schema_version") != SCHEMA_VERSION:
        if any(field in payload for field in POOL_BINDING_FIELDS):
            return False
        candidates, selections = payload.get("candidate_runs"), payload.get("selection")
        return (
            isinstance(candidates, list)
            and len(candidates) == 1
            and _matches(candidates[0], expected, CANDIDATE_FIELDS)
            and not any(field in candidates[0] for field in POOL_BINDING_FIELDS)
            and type(candidates[0].get("candidate_index")) is int
            and candidates[0]["candidate_index"] == 0
            and isinstance(selections, Mapping)
            and bool(selections)
            and all(
                _matches(selection, expected, CANDIDATE_FIELDS)
                and not any(field in selection for field in POOL_BINDING_FIELDS)
                and type(selection.get("candidate_index")) is int
                and selection["candidate_index"] == 0
                for selection in selections.values()
            )
        )
    try:
        validate_expected_binding(payload)
    except (ValueError, TypeError):
        return False
    pool = expected["recipe_candidates"]
    candidates, selections = payload.get("candidate_runs"), payload.get("selection")
    if not isinstance(candidates, list) or len(candidates) != len(pool):
        return False
    seen: dict[int, Mapping[str, Any]] = {}
    trajectories: set[int] = set()
    for record in candidates:
        if not isinstance(record, Mapping) or type(record.get("candidate_index")) is not int:
            return False
        index = record["candidate_index"]
        if index in seen or index < 0 or index >= len(pool):
            return False
        if not _matches(record, pool[index], pool[index]):
            return False
        if "recipe_candidates" in record or "recipe_candidates_sha256" in record:
            return False
        seen[index] = record
        trajectory = record.get("trajectory_index")
        if record.get("status") == "excluded":
            if "trajectory_index" not in record or trajectory is not None:
                return False
        elif record.get("status") in {"succeeded", "recovered"}:
            if type(trajectory) is not int or trajectory < 0 or trajectory in trajectories:
                return False
            trajectories.add(trajectory)
        else:
            return False
    if trajectories != set(range(len(trajectories))):
        return False
    if not isinstance(selections, Mapping) or not selections:
        return False
    for selection in selections.values():
        if not isinstance(selection, Mapping) or type(selection.get("candidate_index")) is not int:
            return False
        index = selection["candidate_index"]
        if index not in seen or seen[index]["status"] == "excluded":
            return False
        if not _matches(selection, pool[index], pool[index]):
            return False
        if "recipe_candidates" in selection or "recipe_candidates_sha256" in selection:
            return False
        if (
            type(selection.get("run_index")) is not int
            or selection["run_index"] != seen[index]["trajectory_index"]
        ):
            return False
    return True
