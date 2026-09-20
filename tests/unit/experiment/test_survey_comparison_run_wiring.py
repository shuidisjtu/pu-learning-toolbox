# ruff: noqa: N803, N806, F811
"""Run-side comparison wiring: every manifest states what its result may be compared against.

The pre-registered matrix had a resolver, a coverage gate and an aggregation
report, and nothing in the pilot called any of them -- so a run could finish
without recording which pre-registered unit it was.  These tests pin the two
halves of the wiring: the run-side resolver, and the manifest that carries it.

The shape is not one mapping per manifest.  A SCAR run selects on the PU view
*and* on the clean view, and the matrix registers those as two units under two
different mappings -- so a manifest recording a single ``mapping_id`` would
leave the other half of its own result unadjudicable.
"""

import copy
import json
from dataclasses import replace

import pytest
from _survey_script_helpers import make_splits, survey_script  # noqa: F401

from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA
from pu_toolbox.experiment.survey_comparison import (
    comparison_digest,
    load_comparison_protocol,
    run_comparison_units,
)
from pu_toolbox.experiment.survey_execution import assemble_model
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATH, load_protocol, resolve_unit

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def survey():
    return load_protocol()


@pytest.fixture(scope="module")
def comparison(survey):
    return load_comparison_protocol(survey=survey)


def _context(method="upu", training_path="native_2d", dataset="spambase"):
    """The part of a runner's protocol context the resolver reads."""
    return {
        "execution_mode": "versioned_pilot",
        "execution_unit": {
            "dataset": dataset,
            "method": method,
            "training_path": training_path,
        },
    }


def _config(mechanism="scar", c_token="0.1", method="upu", dataset="spambase"):
    return {
        "c_requested_token": c_token,
        "survey_protocol": {
            "path": str(PROTOCOL_PATH),
            "dataset": dataset,
            "method": method,
            "mechanism": mechanism,
        },
    }


# --- which units a run produces ---------------------------------------------


def test_basic_scar_run_records_one_entry_per_selection_protocol(survey, comparison):
    units = run_comparison_units(_context(), _config(), survey=survey, comparison=comparison)
    assert set(units) == {"PA", "OA"}
    assert units["PA"]["mapping_id"] != units["OA"]["mapping_id"]


def test_basic_sar_run_records_only_the_oa_unit(survey, comparison):
    """SAR rows carry an OA choice and no PA one, so they contribute one unit."""
    units = run_comparison_units(
        _context(),
        _config(mechanism="sar_lbe_a", c_token="0.05"),
        survey=survey,
        comparison=comparison,
    )
    assert set(units) == {"OA"}


def test_basic_oracle_run_records_its_c_independent_oa_unit(survey, comparison):
    """The oracle trains on real labels, so its unit has no c even though the CLI carries one.

    ``--oracle`` is required to pass ``--labeling-mechanism scar`` to satisfy
    the runner's c-token binding.  Reading the mechanism off the request would
    therefore look for a scar unit that does not exist and refuse the run.
    """
    units = run_comparison_units(
        _context(method="pn_oracle"),
        _config(method="pn_oracle"),
        survey=survey,
        comparison=comparison,
    )
    assert set(units) == {"OA"}
    assert units["OA"]["mapping_id"] == "unit_pn_oracle_spambase_c_independent_c_independent_oa"


# --- parameter and coverage errors ------------------------------------------


def test_param_off_grid_c_token_records_nothing_instead_of_refusing(survey, comparison):
    """``--c`` may leave the grid; the plan asks for a deviation flag, not a refusal.

    Refusing here would delete a documented capability -- the execution plan
    lets c, seeds, the split reference and the candidate pool be overridden,
    requiring only that the deviation be recorded and the run kept off the
    formal leaderboard.
    """
    assert (
        run_comparison_units(
            _context(), _config(c_token="0.9"), survey=survey, comparison=comparison
        )
        == {}
    )


def test_param_an_expected_unit_the_matrix_does_not_map_fails_loud(survey, comparison):
    """The other direction: a *pre-registered* unit with no mapping is a defect."""
    incomplete = copy.deepcopy(comparison)
    incomplete["mappings"] = [
        item
        for item in incomplete["mappings"]
        if item["result_selector"]["selection_protocol"] != "pa"
    ]
    with pytest.raises(ValueError, match="no comparison mapping"):
        run_comparison_units(_context(), _config(), survey=survey, comparison=incomplete)


# --- boundary between two mechanisms that share a token ----------------------


def test_edge_the_same_c_token_under_two_mechanisms_is_two_units(survey, comparison):
    """c=0.5 is a SCAR token and a SAR one; only the mechanism tells them apart."""
    scar = run_comparison_units(
        _context(), _config(c_token="0.5"), survey=survey, comparison=comparison
    )
    sar = run_comparison_units(
        _context(),
        _config(mechanism="sar_lbe_a", c_token="0.5"),
        survey=survey,
        comparison=comparison,
    )
    assert scar["OA"]["mapping_id"] != sar["OA"]["mapping_id"]


# --- determinism -------------------------------------------------------------


def test_determ_entries_are_reproducible_and_bound_to_the_matrix_digest(survey, comparison):
    first = run_comparison_units(_context(), _config(), survey=survey, comparison=comparison)
    second = run_comparison_units(_context(), _config(), survey=survey, comparison=comparison)
    assert first == second
    for entry in first.values():
        assert entry["comparison_sha256"] == comparison_digest(comparison)
        assert entry["comparison_version"] == comparison["comparison_version"]


# --- the manifest that carries it -------------------------------------------


class _LowerCasePA(ProtocolPA):
    """A PA strategy whose artifact names its protocol differently from its own name.

    ``runner_protocol_context`` checks ``{p.name for p in protocols}`` but the
    manifest's selection keys come from the artifact, so this class passes the
    preflight and still writes keys the comparison block cannot join.
    """

    def select(self, trajectories, val_part, threshold_candidates=None, *, class_prior=None):
        artifact = super().select(
            trajectories, val_part, threshold_candidates, class_prior=class_prior
        )
        return replace(artifact, protocol="pa")


def _bound_runner(survey_script, tmp_path, *, c_token="0.5", protocols=None):
    """A versioned spambase/upu run with its protocol-exact model.

    c=0.5 rather than 0.1: the synthetic train split holds six positives, and
    uPU needs at least two *labeled* ones, so the smallest SCAR token would
    exclude every candidate and never reach the manifest under test.
    """
    make_splits(tmp_path)
    parts = survey_script.load_split_parts(tmp_path)
    protocol = load_protocol()
    row = resolve_unit(protocol, "spambase", "upu")
    model = assemble_model(protocol, row, 3, seed=0, params={}, class_prior=0.3, device="cpu")
    config = {
        "architecture": "mlp",
        "c": float(c_token),
        "c_requested_token": c_token,
        "split_ref": {"dataset": "spambase", "seed": 0},
        "survey_protocol": {
            "path": str(PROTOCOL_PATH),
            "dataset": "spambase",
            "method": "upu",
            "mechanism": "scar",
            "seeds": [0],
            "c_tokens": ["0.1", "0.3", "0.5"],
        },
    }
    runner = ExperimentRunner(
        config=config,
        protocols=protocols,
        manifest_path=str(tmp_path / "manifest.json"),
        class_prior=0.3,
    )
    return runner, (model, parts)


def test_basic_versioned_manifest_joins_its_comparison_entries_to_its_selections(
    tmp_path, survey_script
):
    runner, (model, parts) = _bound_runner(survey_script, tmp_path)
    runner.fit(model, *parts)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    # P2.2 joins these by name rather than by position, so the key sets must be
    # equal -- a comparison entry with no matching selection adjudicates nothing.
    assert set(manifest["comparison"]) == set(manifest["selection"]) == {"PA", "OA"}
    assert manifest["comparison"]["PA"]["mapping_id"] == "unit_upu_spambase_scar_0.5_pa"


def test_basic_off_grid_c_run_still_runs_and_states_its_deviation(tmp_path, survey_script):
    """A c outside the grid produces a run, not a refusal -- flagged, and without a comparison."""
    runner, (model, parts) = _bound_runner(survey_script, tmp_path, c_token="0.9")
    runner.fit(model, *parts)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_mode"] == "versioned_pilot"
    assert set(manifest["selection"]) == {"PA", "OA"}
    assert manifest["formal_eligible"] is False
    assert "c" in manifest["protocol_deviation"]
    assert "comparison" not in manifest


def test_param_selection_names_that_cannot_join_the_comparison_are_refused(tmp_path, survey_script):
    """P2.2 joins the two blocks by name, so names that disagree must not reach a manifest."""
    runner, (model, parts) = _bound_runner(
        survey_script, tmp_path, protocols=[_LowerCasePA(), ProtocolOA()]
    )
    with pytest.raises(ValueError, match="disagree on the result units"):
        runner.fit(model, *parts)


def test_basic_unbound_run_manifest_states_no_comparison(tmp_path, survey_script):
    """A technical smoke run has no execution unit, so it must not read as one."""
    make_splits(tmp_path)
    parts = survey_script.load_split_parts(tmp_path)
    protocol = load_protocol()
    model = assemble_model(
        protocol,
        resolve_unit(protocol, "spambase", "upu"),
        3,
        seed=0,
        params={},
        class_prior=0.3,
        device="cpu",
    )
    runner = ExperimentRunner(
        config={"c": 0.5}, manifest_path=str(tmp_path / "manifest.json"), class_prior=0.3
    )
    runner.fit(model, *parts)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_mode"] == "technical_smoke"
    assert "comparison" not in manifest
