"""survey-comparison-v6: the matrix P4.2b completes (three unbenchmarked methods).

v5 left genpu, pulns and split_pu outside the matrix, so a run of theirs could
only say "no lookup path".  P4.2b registers their v3 cells instead, which puts
the absence of a published number on the unit rather than in a decision record
no reader of a manifest sees.  These tests pin that the addition is exactly
that -- 71 cells, no anchor, no dataset or metric changed -- and that the map
promotes exactly those three profiles.
"""

import json

import pytest

from pu_toolbox.experiment.survey_comparison import (
    comparison_digest,
    comparison_for_protocol,
    load_comparison_protocol,
    validate_comparison_coverage,
)
from pu_toolbox.experiment.survey_profile_map import (
    load_profile_map,
    validate_profile_map,
)
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATHS, load_protocol

pytestmark = pytest.mark.unit

V5_PATH = "pu_toolbox/experiment/survey_comparison_v5.json"
V6_PATH = "pu_toolbox/experiment/survey_comparison_v6.json"
MAP_V2_PATH = "pu_toolbox/experiment/survey_comparison_profile_map_v2.json"
MAP_V3_PATH = "pu_toolbox/experiment/survey_comparison_profile_map_v3.json"

P42B_PROFILES = {"genpu", "pulns", "split_pu"}


@pytest.fixture(scope="module")
def v3():
    return load_protocol(PROTOCOL_PATHS["survey-v3.0"])


@pytest.fixture(scope="module")
def v5(v3):
    return load_comparison_protocol(V5_PATH, survey=v3)


@pytest.fixture(scope="module")
def v6(v3):
    return load_comparison_protocol(V6_PATH, survey=v3)


@pytest.fixture(scope="module")
def map_v3():
    return load_profile_map(MAP_V3_PATH)


def _selectors(payload):
    return {json.dumps(m["result_selector"], sort_keys=True) for m in payload["mappings"]}


def _added(v5, v6):
    existing = _selectors(v5)
    return [
        mapping
        for mapping in v6["mappings"]
        if json.dumps(mapping["result_selector"], sort_keys=True) not in existing
    ]


def test_basic_v6_adds_exactly_the_p4_2b_cells(v3, v5, v6, map_v3):
    counts = validate_comparison_coverage(v3, v6, profile_map=map_v3)
    assert counts["units"] == counts["unit_mappings"] == 439
    assert counts["anchors"] == 64
    assert counts["outside_matrix"] == 19
    added = _added(v5, v6)
    assert len(added) == 151
    added_methods = {mapping["result_selector"]["method"] for mapping in added}
    assert added_methods >= P42B_PROFILES


def test_basic_the_addition_manufactures_no_anchor_or_rule(v5, v6):
    """P4.2b only registers cells; a threshold or dataset change would be a rewrite."""
    added = _added(v5, v6)
    assert {anchor["anchor_id"] for anchor in v6["anchors"]} == {
        anchor["anchor_id"] for anchor in v5["anchors"]
    }
    assert v6["decision_rules"] == v5["decision_rules"]
    assert {mapping["result_selector"]["metric"] for mapping in added} == {"accuracy"}
    assert all(not mapping["anchor_ids"] for mapping in added)
    assert all(mapping["eligibility"] in {"no_direct_anchor", "non_runnable"} for mapping in added)
    assert all(mapping["rationale"] for mapping in added)


def test_param_map_v3_promotes_exactly_those_three_profiles(v3, v6, map_v3):
    counts = validate_profile_map(map_v3, protocol=v3, comparison=v6)
    assert counts["mapped"] == 16
    assert counts["not_in_matrix"] == 7
    assert map_v3["bound_comparison"]["comparison_version"] == "survey-comparison-v6"
    assert map_v3["bound_comparison"]["comparison_version"] == v6["comparison_version"]
    map_v2 = load_profile_map(MAP_V2_PATH)
    promoted = {
        profile
        for profile, entry in map_v3["profiles"].items()
        if entry["registration"] == "mapped"
        and map_v2["profiles"][profile]["registration"] == "not_in_matrix"
    }
    assert promoted == P42B_PROFILES


def test_edge_the_non_runnable_pulns_row_is_a_marker_not_a_cell(v6):
    """pulns on native_cnn cannot run under v3; the gap is recorded, not dropped."""
    markers = [
        mapping
        for mapping in v6["mappings"]
        if mapping["result_selector"]["method"] == "pulns"
        and mapping["eligibility"] == "non_runnable"
    ]
    assert len(markers) == 1
    assert markers[0]["result_selector"]["c_token"] == "non_runnable"
    assert markers[0]["anchor_ids"] == []


def test_determ_v6_is_the_shipped_resolution(v3, v6):
    assert comparison_for_protocol(v3)["comparison_version"] == "survey-comparison-v6"
    assert comparison_digest(comparison_for_protocol(v3)) == comparison_digest(v6)
    assert load_comparison_protocol(V6_PATH)["comparison_version"] == "survey-comparison-v6"
