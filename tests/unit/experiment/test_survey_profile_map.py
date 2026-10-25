"""Versioned profile -> comparison mapping: inheritance, deviations and the gaps.

Two things the map must not blur, and these tests pin both:

* a profile outside the matrix (no entry was ever written) is not the same
  state as a profile whose entry has not been mapped yet -- the second is a
  defect, and the resolver raises instead of reporting ``not_in_matrix``;
* ``not_in_matrix`` is a statement about the matrix, never about runnability.
  ``kldce`` is excluded from running *and* has three entries, so it is mapped;
  turning it into ``not_in_matrix`` would make the rows unreachable to the
  reverse-coverage check, which is exactly what that check is for.
"""

import copy
import json

import pytest

from pu_toolbox.experiment.survey_comparison import run_comparison_units
from pu_toolbox.experiment.survey_profile_map import (
    digest,
    load_profile_map,
    profile_map_binds,
    profile_registration_status,
    resolve_profile_mapping,
    validate_profile_map,
)
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATHS, load_protocol

pytestmark = pytest.mark.unit

V3_PATH = PROTOCOL_PATHS["survey-v3.0"]

#: A unit ``self_pu_full_meta`` really produces: its spambase row is native_2d,
#: and 0.3/OA is one of the pre-registered SCAR cells.
IDENTITY = {
    "dataset": "spambase",
    "labeling_mechanism": "scar",
    "c_token": "0.3",
    "selection_protocol": "oa",
    "metric": "accuracy",
    "training_path": "native_2d",
}
KLDCE_IDENTITY = {
    "dataset": "spambase",
    "labeling_mechanism": "non_runnable",
    "c_token": "non_runnable",
    "selection_protocol": "non_runnable",
    "metric": "accuracy",
    "training_path": "native_2d",
}


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(V3_PATH)


def _context(method="nnpu", dataset="spambase", training_path="native_2d"):
    return {
        "execution_mode": "versioned_pilot",
        "execution_unit": {
            "dataset": dataset,
            "method": method,
            "training_path": training_path,
        },
    }


def _config(method="nnpu", dataset="spambase", mechanism="scar", c_token="0.1"):
    return {
        "c_requested_token": c_token,
        "survey_protocol": {
            "path": str(V3_PATH),
            "dataset": dataset,
            "method": method,
            "mechanism": mechanism,
        },
    }


def test_basic_map_covers_every_protocol_profile_and_the_shipped_digests(protocol):
    """v3 is the effective map: 16 profiles inherit an entry, 7 sit outside it."""
    profile_map = load_profile_map()
    counts = validate_profile_map(profile_map, protocol=protocol)
    assert profile_map["map_version"] == "survey-profile-map-v3"
    assert set(profile_map["profiles"]) == set(protocol["method_profiles"])
    assert counts == {"profiles": 23, "mapped": 16, "not_in_matrix": 7, "deviations": 1}
    assert profile_map["bound_survey_protocol"]["protocol_version"] == protocol["protocol_version"]


def test_basic_full_meta_inherits_the_self_pu_entry_and_adds_its_own_deviation():
    profile_map = load_profile_map()
    base = resolve_profile_mapping("self_pu", IDENTITY, profile_map=profile_map)
    full = resolve_profile_mapping("self_pu_full_meta", IDENTITY, profile_map=profile_map)
    assert full["status"] == "mapped"
    assert full["inherits"] == "self_pu"
    assert full["mapping_id"] == base["mapping_id"]
    assert base["deviation_count"] == 0
    assert full["deviation_count"] == 1
    profile_deviations = [
        item for item in full["protocol_differences"] if item.get("source") == "profile_map"
    ]
    assert len(profile_deviations) == 1
    assert profile_deviations[0]["against"] == "pu_bench_table1"
    assert profile_deviations[0]["axis"] == "supervision"


def test_basic_kldce_maps_onto_its_non_runnable_marker_rows():
    resolved = resolve_profile_mapping("kldce", KLDCE_IDENTITY)
    assert resolved["profile_registration"] == "mapped"
    assert resolved["inherits"] == "kldce"
    assert resolved["eligibility"] == "non_runnable"


def test_basic_not_in_matrix_returns_a_reason_and_no_entry():
    resolved = resolve_profile_mapping("gradpu", IDENTITY)
    assert resolved["status"] == "not_in_matrix"
    assert resolved["deviation_count"] == 0
    assert resolved["profile_rationale"] == "method not runnable under survey-v3.0"
    assert "mapping_id" not in resolved


def test_basic_registration_status_separates_the_three_manifest_states(protocol):
    def status(profile_id):
        return profile_registration_status(profile_id, protocol=protocol)[
            "comparison_registration_status"
        ]

    assert status("self_pu") == "mapped_inherited"
    assert status("self_pu_full_meta") == "mapped_with_profile_deviations"
    # P4.2b registered the three unbenchmarked methods, so genpu is mapped now.
    assert status("genpu") == "mapped_inherited"
    assert status("gradpu") == "not_in_matrix"
    assert status("cvir") == "not_in_matrix"


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_param_validate_refuses_a_map_that_is_not_a_total_function(mutation):
    profile_map = copy.deepcopy(load_profile_map())
    if mutation == "missing":
        profile_map["profiles"].pop("genpu")
    else:
        profile_map["profiles"]["not_a_profile"] = copy.deepcopy(profile_map["profiles"]["nnpu"])
    with pytest.raises(ValueError, match="cover exactly the protocol's profiles"):
        validate_profile_map(profile_map)


def test_param_validate_refuses_an_inherited_method_the_matrix_does_not_register():
    profile_map = copy.deepcopy(load_profile_map())
    profile_map["profiles"]["nnpu"]["inherits"] = "not_a_comparison_method"
    with pytest.raises(ValueError, match="not a comparison method"):
        validate_profile_map(profile_map)


def test_param_validate_refuses_a_comparison_method_no_profile_inherits():
    """The kldce decision in one assertion: dropping it strands three rows."""
    profile_map = copy.deepcopy(load_profile_map())
    profile_map["profiles"]["kldce"] = {
        "registration": "not_in_matrix",
        "inherits": None,
        "deviations": [],
        "rationale": "method not runnable under survey-v3.0",
    }
    with pytest.raises(ValueError, match="inherited by no profile"):
        validate_profile_map(profile_map)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("axis", "mood", "axis is not recognized"),
        ("against", "some_other_paper", "not a known anchor family"),
    ],
)
def test_param_load_refuses_a_deviation_outside_the_closed_vocabularies(
    tmp_path, field, value, match
):
    profile_map = copy.deepcopy(load_profile_map())
    profile_map["profiles"]["self_pu_full_meta"]["deviations"][0][field] = value
    path = tmp_path / "map.json"
    path.write_text(json.dumps(profile_map), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_profile_map(path)


def test_param_load_refuses_a_deviation_on_a_profile_outside_the_matrix(tmp_path):
    profile_map = copy.deepcopy(load_profile_map())
    profile_map["profiles"]["gradpu"]["deviations"] = copy.deepcopy(
        profile_map["profiles"]["self_pu_full_meta"]["deviations"]
    )
    path = tmp_path / "map.json"
    path.write_text(json.dumps(profile_map), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot carry deviations"):
        load_profile_map(path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("protocol_version", "survey-v2.0", "disagrees with the loaded file"),
    ],
)
def test_param_resolve_refuses_a_map_bound_to_another_protocol_version(field, value, match):
    profile_map = copy.deepcopy(load_profile_map())
    profile_map["bound_survey_protocol"][field] = value
    with pytest.raises(ValueError, match=match):
        resolve_profile_mapping("nnpu", IDENTITY, profile_map=profile_map)


def test_edge_profile_absent_from_the_map_is_a_defect_not_not_in_matrix():
    with pytest.raises(ValueError, match="no entry in the survey profile map"):
        resolve_profile_mapping("never_declared", IDENTITY)


def test_edge_narrowed_matrix_still_resolves_through_the_versioned_map(protocol):
    """Version is the binding: a derived matrix is described by the same map."""
    narrowed = copy.deepcopy(protocol)
    narrowed["seeds"] = [0]
    assert profile_map_binds(load_profile_map(), narrowed)
    units = run_comparison_units(_context(), _config(), survey=narrowed, profile_id="nnpu")
    assert set(units) == {"PA", "OA"}
    for entry in units.values():
        assert entry["comparison_version"] == "survey-comparison-v6"


def test_determ_resolution_and_map_digest_ignore_formatting():
    first = resolve_profile_mapping("self_pu_full_meta", IDENTITY)
    second = resolve_profile_mapping(
        "self_pu_full_meta", IDENTITY, profile_map=copy.deepcopy(load_profile_map())
    )
    assert first == second
    assert first["profile_map_sha256"] == digest(load_profile_map())
    reformatted = json.loads(json.dumps(load_profile_map(), indent=4, sort_keys=True))
    assert digest(reformatted) == first["profile_map_sha256"]
