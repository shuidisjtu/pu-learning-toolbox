"""A10 against a batch's ``checkpoint_inventory.tsv``.
The batches run on another host, so the recorded checkpoint paths do not resolve
here.  The run's own inventory (``<path>	<bytes>`` per file left on disk) stands in
for the directory listing, and is judged file by file: a reference marked reclaimed
must be absent from it, a kept one present, and an inventory file nothing references
is an orphan.  Counting alone would let a lost file and a stray file cancel out.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest
from _survey_summary_helpers import SCRIPTS_DIR, config_for, manifest
from _survey_summary_helpers import entry as _entry
from _survey_summary_helpers import reclaimed_manifest as _reclaimed_manifest

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
checks = importlib.import_module("audit_survey_batches")
pytestmark = pytest.mark.unit


def _inventory(run_dir: Path, indices) -> set[str]:
    return {str(run_dir / f"epoch_{index:04d}_student.pt") for index in indices}


def test_basic_a10_inventory_balances_without_any_file_on_disk(tmp_path):
    run_dir = tmp_path / "never-written"
    entries = [_entry(_reclaimed_manifest(run_dir, references=3, reclaimed=2))]
    check = checks.check_reclaim(entries, inventories={"B1": _inventory(run_dir, [2])})
    assert check["result"] == "pass"
    assert check["observed"] == {"manifests_with_reclaim": 1, "unbalanced": 0, "orphans": 0}


def test_edge_a10_inventory_names_a_reference_that_should_be_on_disk_but_is_not(tmp_path):
    run_dir = tmp_path / "never-written"
    entries = [_entry(_reclaimed_manifest(run_dir, references=3, reclaimed=1))]
    check = checks.check_reclaim(entries, inventories={"B1": _inventory(run_dir, [2])})
    assert check["result"] == "fail"
    assert any("epoch_0001_student.pt" in item and "missing" in item for item in check["evidence"])


def test_edge_a10_inventory_names_a_file_the_manifest_says_was_reclaimed(tmp_path):
    run_dir = tmp_path / "never-written"
    entries = [_entry(_reclaimed_manifest(run_dir, references=3, reclaimed=1))]
    check = checks.check_reclaim(entries, inventories={"B1": _inventory(run_dir, [0, 1, 2])})
    assert check["result"] == "fail"
    assert any(
        "epoch_0000_student.pt" in item and "still on disk" in item for item in check["evidence"]
    )


def test_edge_a10_a_lost_file_and_a_stray_file_do_not_cancel_out(tmp_path):
    run_dir = tmp_path / "never-written"
    entries = [_entry(_reclaimed_manifest(run_dir, references=3, reclaimed=1))]
    check = checks.check_reclaim(entries, inventories={"B1": _inventory(run_dir, [0, 1])})
    assert check["result"] == "fail"
    assert check["observed"]["unbalanced"] == 1


def test_edge_a10_inventory_file_no_manifest_references_is_an_orphan(tmp_path):
    run_dir = tmp_path / "never-written"
    entries = [_entry(_reclaimed_manifest(run_dir, references=2, reclaimed=1))]
    stray = _inventory(run_dir, [1]) | {str(run_dir / "epoch_9999_student.pt")}
    check = checks.check_reclaim(entries, inventories={"B1": stray})
    assert check["result"] == "fail"
    assert check["observed"]["orphans"] == 1
    assert any("epoch_9999_student.pt" in item and "orphan" in item for item in check["evidence"])


def test_param_a10_an_inventory_only_governs_its_own_batch(tmp_path):
    run_dir = tmp_path / "never-written"
    payload = _reclaimed_manifest(run_dir, references=2, reclaimed=1)
    entries = [_entry(payload, batch="B2")]
    check = checks.check_reclaim(entries, inventories={"B1": _inventory(run_dir, [1])})
    assert check["result"] == "not_run"


def test_basic_a10_inventory_leaves_a_batch_that_never_reclaimed_not_applicable():
    check = checks.check_reclaim([_entry(manifest())], inventories={"B1": set()})
    assert check["result"] == "not_applicable"


def test_basic_load_checkpoint_inventory_reads_paths_and_ignores_sizes(tmp_path):
    path = tmp_path / "checkpoint_inventory.tsv"
    path.write_text(
        "/a/epoch_0001_model.pt\t44753817\n\n/a/epoch_0002_model.pt\t1\n", encoding="utf-8"
    )
    assert checks.load_checkpoint_inventory(path) == {
        "/a/epoch_0001_model.pt",
        "/a/epoch_0002_model.pt",
    }


def test_param_load_checkpoint_inventory_rejects_a_malformed_line(tmp_path):
    path = tmp_path / "checkpoint_inventory.tsv"
    path.write_text("/a/epoch_0001_model.pt 44753817\n", encoding="utf-8")
    with pytest.raises(checks.ConfigError, match="line 1"):
        checks.load_checkpoint_inventory(path)


def test_param_load_config_rejects_an_inventory_that_is_not_a_path_string(tmp_path):
    config = config_for({"B1": (tmp_path, 1)})
    config["batches"][0]["checkpoint_inventory"] = ["x"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(checks.ConfigError, match="checkpoint_inventory"):
        checks.load_config(path)


def test_edge_load_config_resolves_a_relative_inventory_beside_the_config(tmp_path):
    config = config_for({"B1": (tmp_path, 1)})
    config["batches"][0]["checkpoint_inventory"] = "ev/checkpoint_inventory.tsv"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = checks.load_config(path)
    assert loaded["batches"][0]["checkpoint_inventory"] == str(
        tmp_path / "ev" / "checkpoint_inventory.tsv"
    )


def test_determ_a10_inventory_verdict_does_not_depend_on_inventory_order(tmp_path):
    run_dir = tmp_path / "never-written"
    entries = [_entry(_reclaimed_manifest(run_dir, references=4, reclaimed=1))]
    listed = sorted(_inventory(run_dir, [1, 2, 3]))
    forward = checks.check_reclaim(entries, inventories={"B1": set(listed)})
    backward = checks.check_reclaim(entries, inventories={"B1": set(reversed(listed))})
    assert forward == backward
