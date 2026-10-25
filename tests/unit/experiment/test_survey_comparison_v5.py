"""survey-comparison-v5: the first matrix bound to survey-v3.0 (P4.2a + D41).

v5 is generated from the v3 matrix rather than hand-listed, so these tests pin
the properties a generator can get wrong quietly: coverage of the units the
protocol actually enumerates, anchors that belong to exactly one mapping, PA
units with no anchor at all, and profiles that inherit instead of duplicating.
"""

import copy
import json

import pytest

from pu_toolbox.experiment.survey_comparison import (
    load_comparison_protocol,
)
from pu_toolbox.experiment.survey_profile_map import (
    digest,
    load_profile_map,
    validate_profile_map,
)
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATHS, load_protocol

pytestmark = pytest.mark.unit

V5_PATH = "pu_toolbox/experiment/survey_comparison_v5.json"
MAP_V2_PATH = "pu_toolbox/experiment/survey_comparison_profile_map_v2.json"


@pytest.fixture(scope="module")
def v3():
    return load_protocol(PROTOCOL_PATHS["survey-v3.0"])


@pytest.fixture(scope="module")
def v5(v3):
    return load_comparison_protocol(V5_PATH, survey=v3)


@pytest.fixture(scope="module")
def map_v2():
    return load_profile_map(MAP_V2_PATH)


def test_basic_v5_carries_the_288_unit_mappings_it_was_built_with(v5):
    """v5 is a historical rule set: its own counts, not the live protocol's."""
    units = [mapping for mapping in v5["mappings"] if mapping["scope"] == "unit"]
    rows = [mapping for mapping in v5["mappings"] if mapping["scope"] == "row"]
    assert len(units) == 288
    assert len(rows) == 7
    assert len(v5["anchors"]) == 64


def test_basic_pa_units_carry_no_anchor(v5):
    """D41: neither benchmark reports a proxy-accuracy-selected number here."""
    pa_units = [
        mapping
        for mapping in v5["mappings"]
        if mapping["scope"] == "unit" and mapping["result_selector"]["selection_protocol"] == "pa"
    ]
    assert pa_units, "v5 must still enumerate PA units"
    assert all(mapping["eligibility"] == "no_direct_anchor" for mapping in pa_units)
    assert all(not mapping["anchor_ids"] for mapping in pa_units)
    assert not [
        mapping for mapping in pa_units if mapping["eligibility"] == "blocked_pending_pa_criterion"
    ]


def test_basic_promoted_profiles_anchor_their_c01_oa_cells(v5):
    """P4.2a: the four profiles get a real PU-Bench number on their OA unit."""
    expected = {
        "pulda": 3,
        "robust_pu": 3,
        "holistic_pu": 3,
        "lagam": 1,  # lagam runs on Spambase only; the other cells have no unit
    }
    for method, cells in expected.items():
        anchored = [
            mapping
            for mapping in v5["mappings"]
            if mapping["scope"] == "unit"
            and mapping["result_selector"]["method"] == method
            and mapping["result_selector"]["c_token"] == "0.1"
            and mapping["result_selector"]["selection_protocol"] == "oa"
            and mapping["eligibility"] == "magnitude_and_trend"
            and mapping["anchor_ids"]
        ]
        assert len(anchored) == cells, method


def test_param_v5_binds_v3_not_the_historical_v2(v3, v5):
    assert v5["comparison_version"] == "survey-comparison-v5"
    assert v5["bound_survey_protocol"]["protocol_version"] == v3["protocol_version"]
    assert v5["bound_survey_protocol"]["protocol_version"] == v3["protocol_version"]


def test_param_map_v2_binds_v5_and_promotes_exactly_four_profiles(map_v2, v5, v3):
    counts = validate_profile_map(map_v2, protocol=v3, comparison=v5)
    assert map_v2["bound_comparison"]["comparison_version"] == v5["comparison_version"]
    assert map_v2["bound_comparison"]["comparison_version"] == v5["comparison_version"]
    assert counts["mapped"] == 13
    assert counts["not_in_matrix"] == 10
    promoted = {
        "pulda",
        "robust_pu",
        "holistic_pu",
        "lagam",
    }
    assert {
        profile
        for profile, entry in map_v2["profiles"].items()
        if entry["registration"] == "mapped"
    } >= promoted


def test_edge_inheriting_profile_gets_no_mapping_of_its_own(v5):
    """D40: self_pu_full_meta resolves through self_pu, so v5 must not restate it."""
    methods = {mapping["result_selector"]["method"] for mapping in v5["mappings"]}
    assert "self_pu_full_meta" not in methods
    assert "self_pu" in methods


def test_edge_every_anchor_is_used_by_exactly_one_mapping(v5):
    used = [anchor_id for mapping in v5["mappings"] for anchor_id in mapping["anchor_ids"]]
    assert len(used) == len(set(used))
    assert set(used) == {anchor["anchor_id"] for anchor in v5["anchors"]}


def test_determ_v5_loading_is_stable(v3, v5):
    again = load_comparison_protocol(V5_PATH, survey=v3)
    assert digest(again) == digest(v5)
    units = [mapping for mapping in v5["mappings"] if mapping["scope"] == "unit"]
    assert len(units) == 288
    assert len({json.dumps(m["result_selector"], sort_keys=True) for m in units}) == 288
    assert copy.deepcopy(units) == units
