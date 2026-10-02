"""A01 / A06 / A07 / A09: the checks decidable from manifests alone.

Each of these reads the artifacts the audit is handed and nothing else, so each is
asserted on both sides -- the shape that is clean, and the shape that is not.  The
point of the pair is that neither verdict is reachable by accident: a check that
only ever passes is indistinguishable from one that never ran, which is the failure
mode these checks were added to close.

The checks live in the entry-point script (the library owns the schema and the
preflight, the script owns the policy), so they are reached through the same
loader the CLI fixture uses.

The reclaim, label and probe-separation checks have their own file,
``test_survey_audit_reclaim.py``.
"""

import copy
import importlib
import sys
from pathlib import Path

import pytest
from _survey_summary_helpers import SCRIPTS_DIR, config_for, load_protocol, manifest

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
checks = importlib.import_module("audit_survey_batches")

pytestmark = pytest.mark.unit


def _entry(payload, *, batch="B1", role="formal", name="manifest.json"):
    return {"batch": batch, "role": role, "path": Path(name), "payload": payload}


def _grid(*, seeds=(0, 1, 2, 3, 4)):
    """One SCAR row over the protocol's c grid: the unit A09 judges a row against."""
    c_tokens = [float(token) for token in load_protocol()["c_tokens"]["scar"]]
    return [
        _entry(manifest(c=c, seed=seed), name=f"c_{c}/seed_{seed}/manifest.json")
        for c in c_tokens
        for seed in seeds
    ]


def test_basic_a01_reports_the_whitelist_it_applied():
    check = checks.check_whitelist(
        config_for({"B1": (Path("."), 2)}), [_entry(manifest()), _entry(manifest())]
    )

    assert check["result"] == "pass"
    assert check["observed"]["batches"] == ["B1"]


def test_param_a01_reports_a_missing_root_as_an_error():
    entries = [{"batch": "B1", "missing_root": "F:/nowhere"}]
    check = checks.check_whitelist(config_for({"B1": (Path("."), 2)}), entries)

    assert check["result"] == "fail"
    assert check["evidence"] == ["F:/nowhere"]


def test_basic_a06_passes_when_every_result_names_its_candidate():
    check = checks.check_selection([_entry(manifest()), _entry(manifest(seed=1))])

    assert check["result"] == "pass"


def test_param_a06_a_result_with_no_selected_candidate_is_an_error():
    payload = manifest()
    payload["selection"] = {}
    check = checks.check_selection([_entry(payload)])

    assert check["result"] == "fail"
    # Both protocols are reported, not just the first one found.
    assert {item.split(": ", 1)[1].split()[0] for item in check["evidence"]} == {"OA", "PA"}


def test_basic_a07_reports_the_view_and_calibration_distribution():
    entries = [
        _entry(manifest(run_view="os-compatible")),
        _entry(manifest(seed=1, run_view="ts-compatible"), name="second.json"),
    ]
    check = checks.check_view_mechanism(entries)

    assert check["result"] == "pass"
    assert check["observed"]["distribution"] == {
        "scar/os-compatible/calibrated=False": 1,
        "scar/ts-compatible/calibrated=True": 1,
    }


def test_param_a07_a_manifest_without_calibration_is_an_error():
    payload = manifest()
    payload.pop("calibration_applied")
    check = checks.check_view_mechanism([_entry(payload)])

    assert check["result"] == "fail"
    assert "no calibration_applied" in check["evidence"][0]


def test_basic_a09_passes_when_every_designed_unit_of_a_row_exists():
    check = checks.check_result_completeness(_grid(), protocol=load_protocol())

    assert check["result"] == "pass"
    assert check["observed"]["missing_units"] == 0


def test_edge_a09_reports_the_seed_units_a_row_does_not_cover():
    check = checks.check_result_completeness(_grid(seeds=(0, 1, 2, 3)), protocol=load_protocol())

    assert check["result"] == "fail"
    # Four seeds ran where five were designed: the gap is named, not counted.
    assert check["observed"]["missing_units"] == 6
    assert all("/seed=4" in item for item in check["evidence"])


def test_param_a09_a_row_using_a_mechanism_the_protocol_never_defined_is_an_error():
    entry = _entry(manifest(mechanism="sar_lbe_c"))
    check = checks.check_result_completeness([entry], protocol=load_protocol())

    assert check["result"] == "fail"
    assert check["observed"]["rows_with_undefined_mechanism"] == 1


def test_determ_a_check_is_a_pure_function_of_the_entries_it_is_handed():
    entries = _grid(seeds=(0, 1))
    snapshot = copy.deepcopy(entries)
    first = checks.check_result_completeness(entries, protocol=load_protocol())

    assert checks.check_result_completeness(entries, protocol=load_protocol()) == first
    # It reads; it does not annotate the entries it was handed.
    assert entries == snapshot
