"""Recorded research settings accompany result statistics and rendered reports."""

import csv
import json

import pytest
from _survey_summary_helpers import manifest, write_tree

from pu_toolbox.experiment.reporting import export, write_summary
from pu_toolbox.experiment.results import build_summary
from pu_toolbox.experiment.survey_workbook import build_workbook

pytestmark = pytest.mark.unit

LEGACY_SUMMARY = "历史记录：按原 manifest 设置读取"
RECORDED_SUMMARY = "OS | SCAR\n总体先验 0.4；OA；二维特征"


def _settings(seed):
    return {
        "schema_version": "1.0",
        "sampling": {"input_sampling": "os"},
        "labeling": {"mechanism": "scar", "n_labeled": 10 + seed},
        "risk": {"loss_unlabeled": "full_training_pool"},
        "prior": {"value": 0.4, "target": "population", "source": "known"},
        "supervision": {"selection_protocols": ["PA", "OA"]},
        "representation": {"training_path": "native_2d"},
        "summary": RECORDED_SUMMARY,
    }


def _tree(root, mode):
    payloads = {}
    for seed in range(5):
        payload = manifest(seed=seed, accuracy=0.8 + seed * 0.01)
        if mode == "new" or (mode == "mixed" and seed % 2 == 0):
            payload["research_settings"] = _settings(seed)
        payloads[f"seed_{seed}"] = payload
    return write_tree(root, payloads)


@pytest.mark.parametrize("mode", ["old", "new", "mixed"])
def test_basic_param_settings_keep_identity_statistics_costs_and_source_bytes(tmp_path, mode):
    baseline = build_summary([_tree(tmp_path / "baseline", "old")])
    tree = _tree(tmp_path / "runs", mode)
    before = {path: path.read_bytes() for path in tree.rglob("manifest.json")}

    report = build_summary([tree])

    assert report["run_costs"] == baseline["run_costs"]
    expected = {
        str(seed): _settings(seed) if mode == "new" or (mode == "mixed" and seed % 2 == 0) else None
        for seed in range(5)
    }
    summaries = sorted(
        {value["summary"] if value else LEGACY_SUMMARY for value in expected.values()}
    )
    for row, original in zip(report["rows"], baseline["rows"], strict=True):
        for field in ("result_identity", "row_key", "metric", "additional_metrics", "cost_id"):
            assert row[field] == original[field]
        assert row["research_settings_by_seed"] == expected
        assert row["research_settings_summaries"] == summaries
    for source in report["sources"]:
        assert source["research_settings"] == expected[str(source["seed"])]
    assert all(path.read_bytes() == content for path, content in before.items())


def test_basic_settings_reach_summary_workbook_and_diagnostic_exports(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    report = build_summary([_tree(tmp_path / "runs", "mixed")])
    target = tmp_path / "export"

    write_summary(report, target)
    export(report, target)

    saved = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    assert saved["rows"][0]["research_settings_by_seed"]["0"] == _settings(0)
    with (target / "summary.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    assert all(RECORDED_SUMMARY in record["research_settings"] for record in records)
    assert all(LEGACY_SUMMARY in record["research_settings"] for record in records)
    markdown = (target / "summary.md").read_text(encoding="utf-8")
    assert "OS \\| SCAR<br>总体先验" in markdown
    assert LEGACY_SUMMARY in markdown
    workbook = openpyxl.load_workbook(target / "survey_results.xlsx")
    for sheet in ("SCAR-PA", "SCAR-OA"):
        cells = [cell for row in workbook[sheet].iter_rows(values_only=True) for cell in row]
        assert "研究设置" in cells
        assert any(RECORDED_SUMMARY in str(cell) for cell in cells)
    cards = json.loads((target / "diagnostic_cards.json").read_text(encoding="utf-8"))
    assert cards["nnpu"]["research_settings_summaries"] == sorted(
        [LEGACY_SUMMARY, RECORDED_SUMMARY]
    )
    cards_md = (target / "diagnostic_cards.md").read_text(encoding="utf-8")
    assert "OS \\| SCAR<br>总体先验" in cards_md


def test_edge_incomplete_settings_remain_visible_on_coverage_sheet(tmp_path):
    pytest.importorskip("openpyxl")
    report = build_summary([_tree(tmp_path / "runs", "new")], expected_seeds=range(6))

    workbook = build_workbook(report)

    cells = [cell for row in workbook["coverage"].iter_rows(values_only=True) for cell in row]
    assert "研究设置" in cells
    assert any(RECORDED_SUMMARY in str(cell) for cell in cells)


def test_determ_repeated_reading_keeps_recorded_settings(tmp_path):
    tree = _tree(tmp_path / "runs", "new")

    first = build_summary([tree])
    second = build_summary([tree])

    assert first == second
    first["rows"][0]["research_settings_by_seed"]["0"]["prior"]["value"] = 0.9
    assert second["rows"][0]["research_settings_by_seed"]["0"]["prior"]["value"] == 0.4
