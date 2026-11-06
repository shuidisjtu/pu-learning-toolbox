"""Finished reports export without loading the batch audit or summary builder."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("openpyxl")

from pu_toolbox.experiment.survey_summary import aggregate_metric  # noqa: E402

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def finished_report(tmp_path):
    rows = []
    for protocol, accuracy in (("PA", 0.75), ("OA", 0.85)):
        metric = aggregate_metric(
            ((seed, accuracy + seed * 0.01) for seed in range(5)),
            metric_name="accuracy",
            expected_seeds=range(5),
        )
        rows.append(
            {
                "row_key": {
                    "comparability_group": "spambase/native_2d/minibatch",
                    "run_view": "os-compatible",
                    "mechanism": "scar",
                    "method": "nnpu",
                    "selection_protocol": protocol,
                    "c_token": "0.1",
                },
                "result_identity": {
                    "dataset": "spambase",
                    "training_path": "native_2d",
                    "method": "nnpu",
                    "c_token": "0.1",
                },
                "metric": metric,
                "additional_metrics": {},
                "cost_id": "one-fit",
                "status": "partial",
                "reasons": ["formal_blockers_present"],
                "unrecorded_measurements": [],
            }
        )
    report = {
        "protocol_version": "survey-v3.0",
        "overall": "pass",
        "admission_status": "partial",
        "rows": rows,
        "run_costs": [],
        "checks": [],
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, report


def _isolated_export(summary_path, out_dir, mode):
    script = textwrap.dedent(
        """
        import importlib.abc
        import json
        import sys
        from pathlib import Path

        class NoBatchTools(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {
                    "audit_survey_batches", "summarize_survey_results",
                    "pu_toolbox.experiment.method_ledger",
                    "pu_toolbox.experiment.survey_audit",
                    "pu_toolbox.experiment.survey_protocol",
                    "pu_toolbox.experiment.survey_comparison",
                    "pu_toolbox.experiment.survey_profile_map",
                }:
                    raise AssertionError("finished reports must not import " + fullname)

        sys.meta_path.insert(0, NoBatchTools())
        sys.path.insert(0, str(Path.cwd() / "scripts"))
        import export_survey_results as exporter

        summary, out_dir, mode = sys.argv[1:]
        if mode == "cli":
            result = exporter.main(["--summary", summary, "--out-dir", out_dir])
        else:
            report = json.loads(Path(summary).read_text(encoding="utf-8"))
            exporter.export(report, out_dir)
            result = 0
        assert "audit_survey_batches" not in sys.modules
        assert "summarize_survey_results" not in sys.modules
        sys.exit(result)
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(summary_path), str(out_dir), mode],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("mode", ["cli", "function"])
def test_basic_param_determ_finished_report_exports_with_batch_modules_unavailable(
    finished_report, tmp_path, mode
):
    import openpyxl

    summary_path, report = finished_report
    out_dir = tmp_path / mode
    result = _isolated_export(summary_path, out_dir, mode)

    assert result.returncode == 0, result.stderr
    assert {path.name for path in out_dir.iterdir()} == {
        "survey_results.xlsx",
        "diagnostic_cards.md",
        "diagnostic_cards.json",
        "conclusions.md",
        "conclusions.json",
    }
    workbook = openpyxl.load_workbook(out_dir / "survey_results.xlsx")
    assert {"SCAR-PA", "SCAR-OA", "SAR-OA", "PN-oracle", "conclusions"} <= set(workbook.sheetnames)
    for sheet, expected in (("SCAR-PA", "77.00"), ("SCAR-OA", "87.00")):
        cells = [cell for row in workbook[sheet].iter_rows(values_only=True) for cell in row]
        assert "nnpu" in cells
        assert any(expected in str(cell) and "±" in str(cell) for cell in cells)
    cards = json.loads((out_dir / "diagnostic_cards.json").read_text(encoding="utf-8"))
    conclusions = json.loads((out_dir / "conclusions.json").read_text(encoding="utf-8"))
    assert cards["nnpu"]["rows_used"] == 2
    assert conclusions["conclusions"]
    assert json.loads(summary_path.read_text(encoding="utf-8")) == report
    repeated_dir = tmp_path / f"{mode}-repeated"
    repeated = _isolated_export(summary_path, repeated_dir, mode)
    assert repeated.returncode == 0, repeated.stderr
    for filename in (
        "diagnostic_cards.md",
        "diagnostic_cards.json",
        "conclusions.md",
        "conclusions.json",
    ):
        assert (repeated_dir / filename).read_bytes() == (out_dir / filename).read_bytes()


def test_edge_bad_summary_reports_error_without_importing_batch_modules(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    out_dir = tmp_path / "out"

    result = _isolated_export(path, out_dir, "cli")

    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out_dir.exists()
