"""Repair CLI is a read-only planner, never an implicit experiment launcher."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[3] / "scripts/prepare_survey_support_repair.py"
    spec = importlib.util.spec_from_file_location("prepare_support_repair_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unit_plan():
    return dict(
        fits=[
            dict(
                dataset="spambase", profile="self_pu_full_meta", training_path="native_2d", status=s
            )
            for s in ["rerun_required"] * 15
        ]
    )


def setup_dirs(tmp_path):
    splits = tmp_path / "splits/spambase"
    splits.mkdir(parents=True)
    support = tmp_path / "support"
    (support / "spambase").mkdir(parents=True)
    return splits, support


def test_basic_determ_commands_keep_frozen_grid_os_view_and_cpu_only_preflight(script, tmp_path):
    splits, support = setup_dirs(tmp_path)
    kwargs = dict(
        source=tmp_path / "code",
        support_root=support,
        split_map={"spambase": str(splits)},
        results_root=tmp_path / "new-results",
        report_root=tmp_path / "report",
    )
    a = script.pilot_commands(unit_plan(), **kwargs)
    assert a == script.pilot_commands(unit_plan(), **kwargs) and len(a) == 1
    command = a[0]
    assert command["whole_unit_fits"] == 15 and not command["launched"]
    argv = command["pilot_argv"]
    assert argv[argv.index("--os-or-ts") + 1] == "os"
    assert argv[argv.index("--device") + 1] == "cuda"
    assert "--seeds" not in argv and "--c" not in argv
    assert "--recipe-binding-schema-version" in argv and "--reclaim-unselected-checkpoints" in argv
    dry = command["cpu_preflight_argv"]
    assert dry[dry.index("--device") + 1] == "cpu" and "--dry-run" in dry


def test_edge_complete_or_conflicting_unit_has_no_automatic_rerun_command(script, tmp_path):
    splits, support = setup_dirs(tmp_path)
    kwargs = dict(
        source=tmp_path,
        support_root=support,
        split_map={"spambase": str(splits)},
        results_root=tmp_path / "new",
        report_root=tmp_path / "report",
    )
    plan = unit_plan()
    for fit in plan["fits"]:
        fit["status"] = "trajectory_complete"
    assert script.pilot_commands(plan, **kwargs) == []
    plan["fits"][0]["status"] = "rerun_required"
    plan["fits"][1]["status"] = "conflict"
    assert script.pilot_commands(plan, **kwargs) == []


@pytest.mark.parametrize("nested", [False, True])
def test_param_overlapping_source_roots_are_refused(script, tmp_path, nested):
    root = tmp_path / "old"
    root.mkdir()
    other = root / "child" if nested else root
    other.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="overlap"):
        script.collect_sources([root, other])


def test_basic_source_collection_keeps_exact_bytes_and_reports_bad_json(script, tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    raw = b'{"profile_id":"lagam"}'
    (root / "manifest.json").write_bytes(raw)
    broken = root / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{")
    entries, issues = script.collect_sources([root])
    assert entries[0]["manifest_sha256"] == script.hashlib.sha256(raw).hexdigest()
    assert (root / "manifest.json").read_bytes() == raw
    assert issues[0]["reason"] == "unreadable_manifest"


@pytest.mark.parametrize("nested", [False, True])
def test_param_report_and_execution_roots_must_be_distinct_before_source_checks(
    script, tmp_path, monkeypatch, capsys, nested
):
    old = tmp_path / "old"
    old.mkdir()
    report = tmp_path / "report"
    fresh = report / "new" if nested else report
    monkeypatch.setattr(
        script, "source_identity", lambda *a: pytest.fail("checked source before output safety")
    )
    rc = script.main(
        [
            "--results-root",
            str(old),
            "--target-source",
            str(tmp_path),
            "--support-root",
            str(tmp_path),
            "--split-map",
            str(tmp_path / "map.json"),
            "--new-results",
            str(fresh),
            "--out-dir",
            str(report),
        ]
    )
    assert rc == 1 and "must be separate" in capsys.readouterr().err
    assert not report.exists()


def test_edge_dirty_target_source_is_refused(script, tmp_path, monkeypatch):
    monkeypatch.setattr(
        script.subprocess, "check_output", lambda *a, **k: b" M support_checkpoints.py"
    )
    with pytest.raises(ValueError, match="clean committed"):
        script.source_identity(tmp_path)


def test_param_wrong_split_map_never_creates_result_output(script, tmp_path):
    _, support = setup_dirs(tmp_path)
    with pytest.raises(ValueError, match="dataset directory"):
        script.pilot_commands(
            unit_plan(),
            source=tmp_path,
            support_root=support,
            split_map={"spambase": str(tmp_path / "wrong")},
            results_root=tmp_path / "new",
            report_root=tmp_path / "report",
        )
    assert not (tmp_path / "new").exists()
