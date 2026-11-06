"""Result-to-report commands preserve statistics, exclusions and optional facts."""

import importlib
import json
import sys
from pathlib import Path

import pytest
from _survey_summary_helpers import manifest, write_tree

pytest.importorskip("openpyxl")
import openpyxl  # noqa: E402

pytestmark = pytest.mark.unit
SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


@pytest.fixture
def commands():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module("summarize_survey_results"), importlib.import_module(
        "export_survey_results"
    )


def _tree(root):
    return write_tree(
        root, {f"seed_{seed}": manifest(seed=seed, accuracy=0.8 + seed * 0.01) for seed in range(5)}
    )


@pytest.mark.parametrize("mode", ["summary", "manifests"])
def test_basic_param_result_commands_export_identical_group_statistics(commands, tmp_path, mode):
    summarizer, exporter = commands
    tree = _tree(tmp_path / "runs")
    summary_dir = tmp_path / "summary"
    assert summarizer.main(["--manifests", str(tree), "--out-dir", str(summary_dir)]) == 0
    summary = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
    assert all(row["metric"]["mean"] == pytest.approx(0.82) for row in summary["rows"])
    out = tmp_path / mode
    source = (
        ["--summary", str(summary_dir / "summary.json")]
        if mode == "summary"
        else ["--manifests", str(tree)]
    )
    assert exporter.main([*source, "--out-dir", str(out)]) == 0
    workbook = openpyxl.load_workbook(out / "survey_results.xlsx")
    assert {"SCAR-PA", "SCAR-OA", "SAR-OA", "PN-oracle", "coverage", "conclusions"} <= set(
        workbook.sheetnames
    )
    cells = [cell for row in workbook["SCAR-OA"].iter_rows(values_only=True) for cell in row]
    assert any("82.00" in str(cell) and "±" in str(cell) for cell in cells)
    assert (out / "conclusions.json").is_file()


def test_basic_pending_input_remains_visible_in_workbook_and_summary(commands, tmp_path):
    _, exporter = commands
    tree = _tree(tmp_path / "runs")
    pending = tree / "seed_4/manifest.json"
    before = pending.read_bytes()
    out = tmp_path / "out"
    assert (
        exporter.main(["--manifests", str(tree), "--pending", str(pending), "--out-dir", str(out)])
        == 0
    )
    report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert report["coverage"]["pending"] == 1
    assert all(row["metric"]["missing_seeds"] == [4] for row in report["rows"])
    workbook = openpyxl.load_workbook(out / "survey_results.xlsx")
    cells = [cell for row in workbook["coverage"].iter_rows(values_only=True) for cell in row]
    assert str(pending.resolve()) in cells
    assert "pending" in cells
    assert pending.read_bytes() == before


def test_basic_optional_literature_and_native_views_reach_diagnostics(commands, tmp_path):
    _, exporter = commands
    tree = _tree(tmp_path / "runs")
    literature = tmp_path / "literature.json"
    literature.write_text(
        json.dumps(
            {
                "facts": [
                    {
                        "fact_id": "f1",
                        "method": "nnpu",
                        "statement": "Published experimental setting.",
                        "signals": ["c_sensitivity"],
                        "source": "Wang et al. 2026, Table 3",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    views = tmp_path / "views.json"
    views.write_text('{"nnpu": "os"}', encoding="utf-8")
    out = tmp_path / "out"
    assert (
        exporter.main(
            [
                "--manifests",
                str(tree),
                "--out-dir",
                str(out),
                "--literature",
                str(literature),
                "--native-views",
                str(views),
            ]
        )
        == 0
    )
    cards = json.loads((out / "diagnostic_cards.json").read_text(encoding="utf-8"))
    assert cards["nnpu"]["native_view"] == "os"
    conclusions = json.loads((out / "conclusions.json").read_text(encoding="utf-8"))
    assert any(item["kind"] == "文献事实" for item in conclusions["conclusions"])


@pytest.mark.parametrize("source", ["bad_json", "missing_file", "bad_seeds", "bad_views"])
def test_edge_param_invalid_inputs_fail_without_traceback(commands, tmp_path, capsys, source):
    _, exporter = commands
    out = tmp_path / "out"
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    args = ["--summary", str(path)]
    if source == "missing_file":
        args = ["--summary", str(tmp_path / "missing.json")]
    elif source == "bad_seeds":
        args = ["--manifests", str(_tree(tmp_path / "runs")), "--expected-seeds", "0,0"]
    elif source == "bad_views":
        path.write_text('{"nnpu": "unknown"}', encoding="utf-8")
        args = ["--manifests", str(_tree(tmp_path / "runs")), "--native-views", str(path)]
    assert exporter.main([*args, "--out-dir", str(out)]) == 1
    error = capsys.readouterr().err
    assert "error:" in error and "Traceback" not in error
    assert not out.exists()


def test_determ_repeated_summary_keeps_values_and_provenance(commands, tmp_path, monkeypatch):
    summarizer, _ = commands
    tree = _tree(tmp_path / "runs")
    monkeypatch.setattr(summarizer, "resolve_code_commit", lambda: ("f" * 40, True))
    reports = []
    for name in ("one", "two"):
        out = tmp_path / name
        assert summarizer.main(["--manifests", str(tree), "--out-dir", str(out)]) == 0
        report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        report.pop("generated_at")
        reports.append(report)
        assert {path.name for path in out.iterdir()} == {
            "summary.json",
            "summary.md",
            "summary.csv",
            "coverage.csv",
            "refusals.json",
        }
    assert reports[0] == reports[1]
    assert reports[0]["provenance"]["code_commit"] == "f" * 40
