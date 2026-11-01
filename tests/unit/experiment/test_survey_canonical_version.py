# ruff: noqa: F811
"""A shipped v3 run is not a modified v2 matrix; historical versions stay pinned."""

import copy
import json

import pytest
from _survey_script_helpers import make_splits, survey_script  # noqa: F401

from pu_toolbox.experiment.survey_protocol import (
    PROTOCOL_PATH,
    _canonical_matrix_matches_version,
    load_protocol,
    resolve_protocol_path,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("version", ["survey-v1.2", "survey-v2.0", "survey-v3.0"])
def test_basic_each_shipped_version_matches_itself(version):
    protocol = load_protocol(resolve_protocol_path(version))
    assert _canonical_matrix_matches_version(protocol)
    assert PROTOCOL_PATH.name == "survey_protocol_v2.json"


@pytest.mark.parametrize("fault", ["changed_budget", "unknown_version"])
def test_param_changed_or_unknown_matrices_are_not_canonical(fault):
    protocol = copy.deepcopy(load_protocol(resolve_protocol_path("survey-v3.0")))
    if fault == "changed_budget":
        protocol["budgets"]["minibatch"]["epochs"] += 1
    else:
        protocol["protocol_version"] = "survey-v999-draft"
    assert not _canonical_matrix_matches_version(protocol)


def test_edge_real_v3_cli_keeps_subset_deviations_but_not_false_version_drift(
    survey_script, tmp_path
):
    inputs = tmp_path / "input"
    inputs.mkdir()
    make_splits(inputs)
    output = tmp_path / "results"
    assert (
        survey_script.main(
            [
                str(inputs),
                "--protocol",
                "survey-v3.0",
                "--dataset",
                "spambase",
                "--method",
                "upu",
                "--seeds",
                "0",
                "--c",
                "0.5",
                "--class-prior",
                "0.4",
                "--out-dir",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads(next(output.rglob("manifest.json")).read_text())
    assert "noncanonical_protocol_matrix" not in manifest["protocol_deviation"]
    assert "seed_subset" in manifest["protocol_deviation"]
    assert not manifest["formal_eligible"]


def test_determ_identity_check_does_not_mutate_the_matrix():
    protocol = load_protocol(resolve_protocol_path("survey-v3.0"))
    original = copy.deepcopy(protocol)
    assert _canonical_matrix_matches_version(protocol)
    assert _canonical_matrix_matches_version(protocol)
    assert protocol == original
