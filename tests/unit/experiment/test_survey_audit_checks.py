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
import json
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


def test_basic_support_selection_satisfies_the_pa_grid_without_renaming_results():
    protocol = checks.load_protocol(checks.resolve_protocol_path("survey-v3.0"))
    entries = []
    for token in protocol["c_tokens"]["scar"]:
        for seed in protocol["seeds"]:
            payload = manifest(method="lagam", c=float(token), seed=seed)
            payload["test_results"]["PA+clean-support"] = payload["test_results"].pop("PA")
            entries.append(_entry(payload))

    check = checks.check_result_completeness(entries, protocol=protocol)

    assert check["result"] == "pass"
    assert check["observed"]["missing_units"] == 0
    assert all("PA+clean-support" in entry["payload"]["test_results"] for entry in entries)


def _admission_entries(tmp_path):
    protocol = checks.load_protocol(checks.resolve_protocol_path("survey-v3.0"))
    unit = next(
        row
        for row in protocol["execution_units"]
        if row["method"] == "lagam" and row["dataset"] == "spambase"
    )
    entries = []
    for token in protocol["c_tokens"]["scar"]:
        for seed in protocol["seeds"]:
            payload = manifest(method="lagam", c=float(token), seed=seed)
            payload.update(
                protocol_version=protocol["protocol_version"],
                protocol_sha256=checks.digest(protocol),
                execution_unit=unit,
                profile_identity_schema_version="survey-profile-identity-1",
                profile_id="lagam",
                algorithm_id="lagam",
                estimator_parameters=protocol["method_profiles"]["lagam"]["params"],
                budget=protocol["budgets"][unit["budget"]],
                formal_eligible=False,
                formal_blockers=[
                    *protocol["formal_blockers"],
                    "profile_identity_extension_engineering_only",
                ],
                protocol_deviation=[],
                selection_checkpoint_scope="independent_per_epoch",
                candidate_runs=[
                    {
                        "status": "succeeded",
                        "params": {},
                        "training_provenance": {"captured_epochs": list(range(20))},
                        "epoch_checkpoints": [
                            {
                                "epoch_position": epoch + 1,
                                "epoch_label": epoch,
                                "component": "model",
                                "validation_metrics": {"PA+clean-support": {}, "OA": {}},
                            }
                            for epoch in range(20)
                        ],
                    }
                ],
            )
            payload.update(checks.profile_registration_status("lagam", protocol=protocol))
            path = tmp_path / f"{token}_{seed}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            entries.append(_entry(payload, name=str(path)))
    passed = [{"check_id": name, "result": "pass"} for name in checks.REASSESSMENT_CHECKS]
    return protocol, entries, passed


def test_basic_admission_reassessment_discharges_only_verified_legacy_progress(tmp_path):
    protocol, entries, passed = _admission_entries(tmp_path)
    original = copy.deepcopy(entries)

    report = checks.reassess_admission(entries, protocol=protocol, checks=passed)

    assert all(not item["remaining_blockers"] for item in report["runs"].values())
    assert all(len(item["discharged_blockers"]) == 3 for item in report["runs"].values())
    assert entries == original


@pytest.mark.parametrize("problem", ["audit", "grid", "epoch", "protocol", "unknown"])
def test_param_admission_reassessment_preserves_unsatisfied_or_unknown_blockers(tmp_path, problem):
    protocol, entries, passed = _admission_entries(tmp_path)
    if problem == "audit":
        passed[0]["result"] = "fail"
    elif problem == "grid":
        entries.pop()
    elif problem == "epoch":
        entries[0]["payload"]["candidate_runs"][0]["epoch_checkpoints"].pop()
    elif problem == "protocol":
        protocol["seeds"] = [0, 1]
    else:
        entries[0]["payload"]["formal_blockers"].append("new_unknown_gate")

    report = checks.reassess_admission(entries, protocol=protocol, checks=passed)

    first = report["runs"][str(entries[0]["path"])]
    assert first["remaining_blockers"]
    if problem == "unknown":
        assert first["remaining_blockers"] == ["new_unknown_gate"]


def test_determ_admission_reassessment_is_stable_and_records_source_bytes(tmp_path):
    protocol, entries, passed = _admission_entries(tmp_path)

    first = checks.reassess_admission(entries, protocol=protocol, checks=passed)

    assert checks.reassess_admission(entries, protocol=protocol, checks=passed) == first
    assert all(len(item["manifest_sha256"]) == 64 for item in first["runs"].values())
