"""A10 / A13 / A15: reclaim accounting, the label closed set, probe separation.

Three checks that each have a shape the artifact set cannot raise, and the point of
each is that the honest answer there is not ``pass``:

* A10 balances ``reclaimed`` against the files still on disk, per manifest.  Batches
  that never reclaimed carry no such field, and reporting them as clean would say
  the accounting was checked when nothing was.
* A13 asserts the label vocabulary and the invariant that makes it mean one thing:
  ``status == "formal"`` exactly when there are no reasons.  The gate's own
  ``comparable`` / ``blocked`` words are not labels and must not be read as any.
* A15 keeps probe batches out of the formal total and refuses a delivered row the
  protocol marks non-runnable.
"""

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


def _reclaimed_manifest(run_dir: Path, *, references: int, reclaimed: int):
    """A manifest whose checkpoint references point into *run_dir*."""
    payload = manifest()
    payload["candidate_runs"] = [
        {
            "candidate_index": 0,
            "epoch_checkpoints": [
                {
                    "epoch": index,
                    "component": "student",
                    "path": str(run_dir / f"epoch_{index:04d}_student.pt"),
                    "reclaimed": index < reclaimed,
                }
                for index in range(references)
            ],
        }
    ]
    return payload


def _write_files(run_dir: Path, count: int):
    run_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (run_dir / f"epoch_{index:04d}_student.pt").write_bytes(b"weights")


def test_basic_a10_is_not_applicable_on_batches_that_never_reclaimed():
    # The measured shape of B1-B3b: the field arrives with the reclaim mechanism, so
    # a tree that predates it has no accounting to check -- which is not a pass.
    check = checks.check_reclaim([_entry(manifest()), _entry(manifest(seed=1))])

    assert check["result"] == "not_applicable"
    assert check["observed"]["with_reclaim_references"] == 0


def test_basic_a10_balances_reclaimed_against_the_files_still_on_disk(tmp_path):
    run_dir = tmp_path / "checkpoints"
    _write_files(run_dir, 2)
    entries = [_entry(_reclaimed_manifest(run_dir, references=3, reclaimed=1))]

    check = checks.check_reclaim(entries)

    assert check["result"] == "pass"
    assert check["observed"] == {"manifests_with_reclaim": 1, "unbalanced": 0}


def test_edge_a10_a_lost_file_is_reported_as_unbalanced(tmp_path):
    run_dir = tmp_path / "checkpoints"
    _write_files(run_dir, 1)
    entries = [_entry(_reclaimed_manifest(run_dir, references=3, reclaimed=0))]

    check = checks.check_reclaim(entries)

    assert check["result"] == "fail"
    assert "reclaimed 0 + on disk 1 != 3 references" in check["evidence"][0]


def test_edge_a10_unreachable_checkpoint_paths_are_not_counted(tmp_path):
    # A foreign absolute path (the batches run on another host) would otherwise make
    # "no files found" look like everything was reclaimed.
    run_dir = tmp_path / "never-written"
    entries = [_entry(_reclaimed_manifest(run_dir, references=2, reclaimed=0))]

    check = checks.check_reclaim(entries)

    assert check["result"] == "not_run"
    assert "do not resolve locally" in check["evidence"][0]


def test_basic_a13_passes_over_labels_in_the_closed_set():
    entries = [_entry(manifest())]
    entries[0]["status"], entries[0]["reasons"] = "formal", []
    entries.append(_entry(manifest(seed=1), name="second.json"))
    entries[1]["status"], entries[1]["reasons"] = "partial", ["missing_seed"]

    check = checks.check_status_labels(entries)

    assert check["result"] == "pass"
    assert check["observed"]["distribution"] == {"formal": 1, "partial": 1}


def test_param_a13_the_gates_blocked_word_is_not_a_status():
    entry = _entry(manifest())
    entry["status"], entry["reasons"] = "blocked", ["fairness_gate_blocked"]

    check = checks.check_status_labels([entry])

    assert check["result"] == "fail"
    assert "is not in the closed set" in check["evidence"][0]


def test_param_a13_a_formal_label_carrying_reasons_is_an_error():
    entry = _entry(manifest())
    entry["status"], entry["reasons"] = "formal", ["missing_seed"]

    check = checks.check_status_labels([entry])

    assert check["result"] == "fail"
    assert "carries reasons" in check["evidence"][0]


def test_basic_a15_is_not_applicable_without_a_probe_batch():
    check = checks.check_probe_separation([_entry(manifest())], protocol=load_protocol())

    assert check["result"] == "not_applicable"
    assert check["observed"]["formal_batches"] == ["B1"]


def test_determ_a15_keeps_probe_batches_out_of_the_formal_total():
    entries = [
        _entry(manifest(), batch="B1"),
        _entry(manifest(seed=1), batch="probe", role="technical_probe", name="p.json"),
    ]
    check = checks.check_probe_separation(entries, protocol=load_protocol())

    assert check["result"] == "pass"
    assert check["observed"]["formal_batches"] == ["B1"]
    assert check["observed"]["probe_batches"] == ["probe"]
    assert check["reasons"] == ["probe_only"]


def test_param_a15_a_non_runnable_row_is_an_error():
    # Read from the protocol rather than hard-coded, so a protocol bump that changes
    # which rows are runnable moves this test with it.
    forbidden = [
        unit
        for unit in checks.expected_result_units(load_protocol())
        if unit["labeling_mechanism"] == "non_runnable"
    ]
    first = forbidden[0]
    entry = _entry(
        manifest(
            method=first["method"], dataset=first["dataset"], training_path=first["training_path"]
        )
    )
    check = checks.check_probe_separation([entry], protocol=load_protocol())

    assert check["result"] == "fail"
    assert check["evidence"] == [
        f"manifest.json: {first['method']}/{first['dataset']}/{first['training_path']}"
    ]
    assert "non-runnable" in check["message"]


def test_basic_a01_carries_no_reason_codes_of_its_own():
    # A whitelist finding is about a tree, not a manifest: borrowing a manifest-level
    # reason code would put a code on a finding it does not describe.
    check = checks.check_whitelist(config_for({"B1": (Path("."), 1)}), [_entry(manifest())])

    assert check["reasons"] == []
