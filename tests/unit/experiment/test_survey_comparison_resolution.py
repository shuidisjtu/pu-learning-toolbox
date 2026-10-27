"""The shipped matrix follows the protocol version, not a hardcoded default.

One path used to serve every version, so a v3 run recorded the v2-bound
matrix's entries: the binding check passed because the loader also loaded v2 as
"the" protocol, and nothing in the manifest could show that its comparison was
made against rules the run never ran under.  These tests pin the two directions
of the replacement -- each version resolves to its own matrix, and a matrix is
never read under another version's protocol -- plus the run-side consequence:
a v3 manifest states v5, a v2 manifest keeps v4.
"""

import copy
from pathlib import Path

import pytest

from pu_toolbox.experiment.survey_comparison import (
    COMPARISON_PATHS,
    comparison_digest,
    comparison_for_protocol,
    comparison_for_version,
    load_comparison_protocol,
    resolve_comparison_path,
    run_comparison_units,
    validate_comparison_coverage,
)
from pu_toolbox.experiment.survey_profile_map import (
    MAP_PATH,
    digest,
    load_profile_map,
    profile_map_binds,
    resolve_profile_mapping,
)
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATHS, load_protocol

pytestmark = pytest.mark.unit

V5_PATH = "pu_toolbox/experiment/survey_comparison_v5.json"
MAP_V1_PATH = "pu_toolbox/experiment/survey_comparison_profile_map_v1.json"

#: A unit the promoted profile really produces: pulda runs on spambase, and
#: 0.1/OA is a pre-registered SCAR cell of its row.
IDENTITY = {
    "dataset": "spambase",
    "labeling_mechanism": "scar",
    "c_token": "0.1",
    "selection_protocol": "oa",
    "metric": "accuracy",
    "training_path": "native_2d",
}


@pytest.fixture(scope="module")
def v3():
    return load_protocol(PROTOCOL_PATHS["survey-v3.0"])


@pytest.fixture(scope="module")
def v2():
    return load_protocol()


@pytest.fixture(scope="module")
def v6(v3):
    return comparison_for_protocol(v3)


@pytest.fixture(scope="module")
def v5():
    """The superseded artifact, still loadable by path for its own reader."""
    return load_comparison_protocol(V5_PATH)


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
            "path": str(PROTOCOL_PATHS["survey-v3.0"]),
            "dataset": dataset,
            "method": method,
            "mechanism": mechanism,
        },
    }


def test_basic_each_version_resolves_to_its_own_matrix(v2, v3, v6):
    assert comparison_for_protocol(v3)["comparison_version"] == "survey-comparison-v6"
    assert comparison_for_protocol(v2)["comparison_version"] == "survey-comparison-v4"
    assert v6["bound_survey_protocol"]["protocol_version"] == v3["protocol_version"]


def test_basic_the_two_version_tables_name_the_same_versions():
    """A protocol added to one table without the other has no matrix to read."""
    assert set(COMPARISON_PATHS) == set(PROTOCOL_PATHS)
    for version, path in COMPARISON_PATHS.items():
        assert path.exists(), version
        assert resolve_comparison_path(version) == path


def test_param_a_matrix_is_refused_under_another_versions_protocol(v2, v3):
    """The binding is checked against the protocol actually in hand, both ways."""
    with pytest.raises(ValueError, match="disagrees with the execution protocol"):
        load_comparison_protocol(V5_PATH, survey=v2)
    with pytest.raises(ValueError, match="disagrees with the execution protocol"):
        load_comparison_protocol(COMPARISON_PATHS["survey-v2.0"], survey=v3)


def test_basic_a_matrix_declares_the_protocol_it_binds(v5):
    """With no protocol named, the file's own declaration is the one checked."""
    assert load_comparison_protocol(V5_PATH)["comparison_version"] == "survey-comparison-v5"
    assert digest(load_comparison_protocol(V5_PATH)) == digest(v5)


def test_basic_v3_run_records_v6_in_every_entry(v6):
    """The runner's own call: no survey and no matrix given, both resolved."""
    units = run_comparison_units(_context(), _config(), profile_id="nnpu")
    assert set(units) == {"PA", "OA"}
    for entry in units.values():
        assert entry["comparison_version"] == "survey-comparison-v6"
        assert entry["comparison_sha256"] == comparison_digest(v6)
        assert entry["profile_registration"] == "mapped"
    assert units["PA"]["eligibility"] == "no_direct_anchor"


def test_basic_v2_run_keeps_the_v2_bound_matrix(v2):
    """The legacy run has no profile, so it keeps the rows it was frozen against."""
    units = run_comparison_units(_context(method="upu"), _config(method="upu"), survey=v2)
    assert set(units) == {"PA", "OA"}
    for entry in units.values():
        assert entry["comparison_version"] == "survey-comparison-v4"


def test_basic_the_effective_profile_map_binds_the_resolved_matrix(v3, v6):
    assert MAP_PATH.name.endswith("_v3.json")
    profile_map = load_profile_map()
    assert profile_map_binds(profile_map, v3)
    assert profile_map["bound_comparison"]["comparison_version"] == v6["comparison_version"]


def test_basic_promoted_profiles_resolve_through_the_effective_map():
    """pulda is the v1 -> v2 promotion in one assertion: it now has an entry."""
    resolved = resolve_profile_mapping("pulda", IDENTITY)
    assert resolved["status"] == "mapped"
    assert resolved["eligibility"] == "magnitude_and_trend"
    historical = load_profile_map(MAP_V1_PATH)
    assert historical["profiles"]["pulda"]["registration"] == "not_in_matrix"


def test_basic_the_coverage_gate_checks_the_resolved_matrix(v3):
    """The 977 units of v6 are proven to cover v3, before any run pays for them."""
    counts = validate_comparison_coverage(
        v3, comparison_for_protocol(v3), profile_map=load_profile_map()
    )
    assert counts["units"] == counts["unit_mappings"] == 977
    assert counts["anchors"] == 64
    assert counts["outside_matrix"] == 19


def test_basic_a_derived_protocol_resolves_through_the_shipped_matrix(v3, v6):
    """A narrowed matrix still records the entry its units land on (v6)."""
    narrowed = copy.deepcopy(v3)
    narrowed["seeds"] = [0]
    assert comparison_for_version(narrowed["protocol_version"])["comparison_version"] == (
        "survey-comparison-v6"
    )
    assert digest(comparison_for_version(narrowed["protocol_version"])) == digest(v6)
    # A derived matrix of the same version still resolves: version is the binding.
    assert comparison_for_protocol(narrowed)["comparison_version"] == "survey-comparison-v6"


def test_edge_a_version_no_shipped_matrix_carries_is_read_as_a_path():
    """Only a shipped version name is translated; anything else is treated as a path."""
    assert resolve_comparison_path("survey-v9.9") == Path("survey-v9.9").resolve()


def test_determ_resolution_is_stable_and_formatting_independent(v3):
    first = comparison_for_protocol(v3)
    reformatted = copy.deepcopy(first)
    assert comparison_digest(reformatted) == comparison_digest(first)
    assert digest(load_profile_map()) == digest(load_profile_map())
