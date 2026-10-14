"""Resume does not mistake a legacy or partially bound result for a bound run."""

import copy
import json

import pytest
from _survey_recipe_registry_helpers import locked_registry

from pu_toolbox.experiment.pilot_plan import (
    PilotScope,
    pending_runs,
    planned_runs,
    select_execution_units,
)
from pu_toolbox.experiment.survey_protocol import digest, load_protocol
from pu_toolbox.experiment.survey_recipe_registry import bind_recipe_candidate

pytestmark = pytest.mark.unit


@pytest.fixture
def plan():
    full = load_protocol()
    protocol = select_execution_units(full, PilotScope(datasets=("spambase",), methods=("upu",)))
    runs = planned_runs(protocol)
    binding = bind_recipe_candidate(
        locked_registry(), full, method="upu", dataset="spambase", training_path="native_2d"
    )
    views = {r.key: "os-compatible" for r in runs}
    protocol_identity = {
        "protocol_sha256": digest(full),
        "protocol_version": full["protocol_version"],
    }
    recipes = {r.key: {**binding, **protocol_identity} for r in runs}
    run = runs[0]
    payload = {
        "execution_mode": "versioned_pilot",
        "seed": run.seed,
        "execution_unit": {
            "dataset": run.dataset,
            "method": run.method,
            "training_path": run.training_path,
        },
        "generation": {"train": {"mechanism": run.mechanism}},
        "c_requested_token": run.c_token,
        "run_view": "os-compatible",
        "calibration_applied": False,
        "selection": {"OA": {"candidate_index": 0, **binding}},
        "candidate_runs": [{"candidate_index": 0, **binding}],
        **binding,
        **protocol_identity,
    }
    return protocol, views, recipes, payload


def write(tmp_path, name, payload):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps(payload))


def resume(plan, tmp_path):
    protocol, views, recipes, _ = plan
    return pending_runs(protocol, tmp_path, expected_views=views, expected_recipe_bindings=recipes)


def test_basic_determ_matching_bound_result_satisfies_exactly_one_run(plan, tmp_path):
    write(tmp_path, "bound", plan[3])
    pending, done = resume(plan, tmp_path)
    assert len(done) == 1 and len(pending) == 34
    assert resume(plan, tmp_path) == (pending, done)


@pytest.mark.parametrize(
    "defect",
    [
        "legacy",
        "digest",
        "candidate",
        "selection",
        "all_fields_removed",
        "null_field_missing",
        "candidate_index",
        "protocol_digest",
    ],
)
def test_param_partial_or_mismatched_binding_is_never_completed(plan, tmp_path, defect):
    payload = copy.deepcopy(plan[3])
    if defect == "legacy":
        for key in plan[2][next(iter(plan[2]))]:
            payload.pop(key, None)
    elif defect == "digest":
        payload["recipe_registry_sha256"] = "0" * 64
    elif defect == "candidate":
        payload["candidate_runs"][0]["recipe_candidate_id"] = "other"
    elif defect == "selection":
        del payload["selection"]["OA"]["resolved_params_sha256"]
    elif defect == "null_field_missing":
        del payload["inner_search_id"]
    elif defect == "candidate_index":
        payload["selection"]["OA"]["candidate_index"] = 9
    elif defect == "protocol_digest":
        payload["protocol_sha256"] = "0" * 64
    else:
        payload = {
            key: value
            for key, value in payload.items()
            if not key.startswith("recipe_")
            and key not in {"resolved_params_sha256", "inner_search_id"}
        }
    write(tmp_path, "invalid", payload)
    assert len(resume(plan, tmp_path)[1]) == 0


def test_edge_legacy_sorted_first_does_not_hide_a_later_matching_result(plan, tmp_path):
    legacy = {
        key: value for key, value in plan[3].items() if key not in next(iter(plan[2].values()))
    }
    write(tmp_path, "a_legacy", legacy)
    write(tmp_path, "z_bound", plan[3])
    assert len(resume(plan, tmp_path)[1]) == 1


def test_edge_incomplete_expectation_is_a_caller_error(plan, tmp_path):
    protocol, views, _, _ = plan
    with pytest.raises(ValueError, match="does not cover"):
        pending_runs(protocol, tmp_path, expected_views=views, expected_recipe_bindings={})


def test_param_partial_identity_expectations_cannot_downgrade_a_bound_request(plan, tmp_path):
    protocol, views, recipes, _ = plan
    incomplete = {
        key: {"recipe_candidate_id": value["recipe_candidate_id"]} for key, value in recipes.items()
    }
    with pytest.raises(ValueError, match="complete registry-bound identities"):
        pending_runs(protocol, tmp_path, expected_views=views, expected_recipe_bindings=incomplete)
