"""Result consumption checks recorded recipe joins without replaying configuration."""

import copy

import pytest
from _survey_summary_helpers import manifest, write_tree

from pu_toolbox.experiment.results import build_summary
from pu_toolbox.experiment.survey_summary import SummaryError
from pu_toolbox.utils.serialization import strict_canonical_hash

pytestmark = pytest.mark.unit


def _bound():
    payload = manifest()
    binding = {
        "recipe_binding_status": "registry_bound",
        "recipe_registry_version": "survey-recipes-v1.2",
        "recipe_registry_sha256": "a" * 64,
        "recipe_profile_id": "nnpu",
        "recipe_candidate_id": "nnpu_default",
        "resolved_params_sha256": "b" * 64,
        "inner_search_id": None,
    }
    payload.update(binding)
    for child in [*payload["candidate_runs"], *payload["selection"].values()]:
        child.update(binding)
    payload["candidate_runs"][0]["params"] = {"learning_rate": 0.001}
    payload["selection"]["OA"]["params"] = {"learning_rate": 0.001}
    return payload


@pytest.mark.parametrize("fault", ["selection", "candidate", "params", "missing", "pool"])
def test_param_recorded_recipe_or_parameter_conflicts_are_refused(tmp_path, fault):
    payload = _bound()
    if fault == "selection":
        payload["selection"]["OA"]["resolved_params_sha256"] = "c" * 64
    elif fault == "candidate":
        payload["candidate_runs"][0]["recipe_profile_id"] = "upu"
    elif fault == "params":
        payload["selection"]["OA"]["params"] = {"learning_rate": 42}
    elif fault == "missing":
        del payload["selection"]["PA"]["recipe_candidate_id"]
    else:
        payload["recipe_candidates"] = [copy.deepcopy(payload["candidate_runs"][0])]
        payload["recipe_candidates_sha256"] = "c" * 64
    tree = write_tree(tmp_path, {"run": payload})
    with pytest.raises(SummaryError, match="recipe|params"):
        build_summary([tree], expected_seeds=[0])


@pytest.mark.parametrize("candidate", [None, [], "invalid", {"candidate_index": True}])
def test_edge_malformed_candidate_is_a_summary_error(tmp_path, candidate):
    payload = manifest()
    payload["candidate_runs"] = [candidate]
    tree = write_tree(tmp_path, {"run": payload})
    with pytest.raises(SummaryError):
        build_summary([tree], expected_seeds=[0])


def test_basic_determ_self_contained_pool_remains_readable_without_registry(tmp_path):
    payload = _bound()
    candidate = payload["candidate_runs"][0]
    payload["recipe_candidates"] = [
        {key: value for key, value in candidate.items() if key != "params"}
    ]
    payload["recipe_candidates_sha256"] = strict_canonical_hash(payload["recipe_candidates"])
    tree = write_tree(tmp_path, {"run": payload})
    first = build_summary([tree], expected_seeds=[0])
    assert first == build_summary([tree], expected_seeds=[0])
    assert first["sources"][0]["selection"] == payload["selection"]


def test_edge_explicit_empty_directory_is_not_a_result_tree(tmp_path):
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SummaryError, match="manifest"):
        build_summary([tmp_path])
    assert build_summary([])["rows"] == []
    pending = build_summary([], pending_paths=[tmp_path])
    assert pending["coverage"]["pending"] == 1


def test_edge_candidate_parameter_changes_across_seeds_cannot_be_averaged(tmp_path):
    first, second = _bound(), _bound()
    second["seed"] = 1
    second["candidate_runs"][0]["params"]["learning_rate"] = 42
    second["selection"]["OA"]["params"]["learning_rate"] = 42
    tree = write_tree(tmp_path, {"first": first, "second": second})
    with pytest.raises(SummaryError, match="configurations"):
        build_summary([tree], expected_seeds=[0, 1])
