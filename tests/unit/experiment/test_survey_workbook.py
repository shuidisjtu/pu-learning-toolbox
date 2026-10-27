# tests/unit/experiment/test_survey_workbook.py

# ruff: noqa: N803, N806, S101

"""The delivery workbook: which sheet a row lands on, and what it may not do.

The report rows are built by hand rather than from a manifest tree: the exporter
reads the summary report and nothing else, so the report is its whole input.
What these pin is the protocol §5.1 split into four groups, that nothing is
ranked, and that a row the report could not complete never reaches a result sheet.
"""

import pytest

pytest.importorskip("openpyxl")

from pu_toolbox.experiment.survey_workbook import (  # noqa: E402
    DIAGNOSTIC_SHEET,
    GROUP_SHEETS,
    WorkbookError,
    build_workbook,
)

pytestmark = pytest.mark.unit

_SEEDS = [0, 1, 2, 3, 4]


def _additional(name, mean=0.8, std=0.02):
    return {
        "metric_name": name,
        "mean": mean,
        "std": std,
        "mean_percent": mean * 100,
        "std_percent": std * 100,
        "n_seeds": 5,
        "n_observed": 5,
        "seeds_observed": _SEEDS,
        "unavailable_seeds": [],
        "unavailable_reasons": {},
    }


def _row(
    *,
    dataset="mnist",
    training_path="native_cnn",
    view="os-compatible",
    mechanism="scar",
    method="upu",
    protocol="PA",
    c="0.1",
    mean=0.9,
    std=0.01,
    observed=5,
    status="formal",
    reasons=(),
):
    group = f"{dataset}/{training_path}/minibatch"
    observed_seeds = _SEEDS[:observed]
    return {
        "result_identity": {
            "method": method,
            "dataset": dataset,
            "labeling_mechanism": "c_independent" if mechanism == "pn_oracle" else mechanism,
            "c_token": c,
            "selection_protocol": protocol.lower(),
            "metric": "accuracy",
            "training_path": training_path,
        },
        "row_key": {
            "comparability_group": group,
            "run_view": view,
            "mechanism": mechanism,
            "method": method,
            "selection_protocol": protocol,
            "c_token": c,
        },
        "cost_id": "|".join((group, view, mechanism, method, c)),
        "metric": {
            "metric_name": "accuracy",
            "mean": mean,
            "std": std,
            "mean_percent": mean * 100,
            "std_percent": None if std is None else std * 100,
            "n_expected": 5,
            "n_observed": observed,
            "seeds_observed": observed_seeds,
            "missing_seeds": [seed for seed in _SEEDS if seed not in observed_seeds],
        },
        "additional_metrics": {
            name: _additional(name) for name in ("auc", "f1", "precision", "recall")
        },
        "unrecorded_measurements": [],
        "status": status,
        "reasons": list(reasons),
        "members": [],
    }


def _report(*rows, run_costs=()):
    return {
        "schema_version": "survey-summary-1",
        "generated_at": "2026-10-27T00:00:00+00:00",
        "protocol_version": "survey-v3.0",
        "protocol_sha256": "ab" * 32,
        "provenance": None,
        "metric_unit": "fraction",
        "coverage": {"rows_by_tier": {"formal": 0, "partial": 0, "diagnostic": 0}},
        "rows": list(rows),
        "run_costs": list(run_costs),
        "refusals": [],
        "checks": [],
        "overall": "pass",
    }


def _cells(workbook, sheet):
    return [list(r) for r in workbook[sheet].iter_rows(values_only=True)]


def _flat(workbook, sheet):
    return [cell for r in _cells(workbook, sheet) for cell in r if cell is not None]


def test_basic_each_row_lands_on_the_sheet_of_its_protocol_group():
    report = _report(
        _row(method="a", mechanism="scar", protocol="PA"),
        _row(method="b", mechanism="scar", protocol="OA"),
        _row(method="c", mechanism="sar_lbe_a", protocol="OA"),
        _row(method="d", mechanism="pn_oracle", protocol="OA"),
    )

    workbook = build_workbook(report)

    assert set(GROUP_SHEETS) <= set(workbook.sheetnames)
    for sheet, method in zip(GROUP_SHEETS, "abcd", strict=True):
        flat = _flat(workbook, sheet)
        assert method in flat
        assert not ({"a", "b", "c", "d"} - {method}) & set(flat)


def test_basic_clean_support_is_listed_apart_from_pure_pu_on_the_pa_sheet():
    report = _report(
        _row(method="lagam", protocol="PA+clean-support"),
        _row(method="upu", protocol="PA"),
    )

    flat = _flat(build_workbook(report), GROUP_SHEETS[0])

    assert flat.index("upu") < flat.index("lagam")
    assert any("clean support" in str(cell) for cell in flat)


def test_basic_the_two_lbe_settings_stay_in_separate_blocks_on_the_sar_sheet():
    report = _report(
        _row(method="x", mechanism="sar_lbe_b", protocol="OA"),
        _row(method="y", mechanism="sar_lbe_a", protocol="OA"),
    )

    flat = _flat(build_workbook(report), GROUP_SHEETS[2])

    assert flat.index("y") < flat.index("x")
    assert any("LBE-A" in str(cell) for cell in flat)
    assert any("LBE-B" in str(cell) for cell in flat)


def test_basic_a_cell_reads_mean_plus_minus_std_in_percent():
    workbook = build_workbook(_report(_row(mean=0.9123, std=0.0145)))

    flat = _flat(workbook, GROUP_SHEETS[0])

    assert "91.23 ± 1.45" in flat


def test_basic_cost_columns_come_from_the_run_cost_entry_and_stay_blank_when_unrecorded():
    row = _row(method="upu")
    costs = {
        "cost_id": row["cost_id"],
        "costs": {
            "peak_gpu_memory_bytes": 2 * 1024**3,
            "single_configuration_cost_seconds": {"mean": 12.5, "sum": 25.0, "n_observed": 2},
            "tuning_cost_seconds": {"sum": 300.0, "n_observed": 5},
            "seeds_observed": _SEEDS,
            "missing_seeds": [],
        },
        "unrecorded_measurements": [],
    }
    other = _row(method="nnpu")

    flat = _flat(build_workbook(_report(row, other, run_costs=[costs])), GROUP_SHEETS[0])

    assert 12.5 in flat and 300.0 in flat and 2.0 in flat
    assert 0 not in flat


def test_edge_a_row_missing_seeds_goes_to_coverage_not_to_a_result_sheet():
    short = _row(
        method="partial_run", observed=1, std=None, status="incomplete", reasons=["missing_seed"]
    )

    workbook = build_workbook(_report(short, _row(method="full_run")))

    assert "partial_run" not in _flat(workbook, GROUP_SHEETS[0])
    coverage = _flat(workbook, "coverage")
    assert "partial_run" in coverage
    assert "missing_seed" in coverage
    assert "full_run" not in coverage


def test_edge_sar_pa_oracle_pa_and_probe_rows_are_diagnostic_only():
    """A probe is a smoke run scored against the full seed grid, so it is short of seeds."""
    sar_pa = _row(method="sarpa", mechanism="sar_lbe_a", protocol="PA")
    oracle_pa = _row(method="orapa", mechanism="pn_oracle", protocol="PA", c="c_independent")
    probe = _row(
        method="probe",
        observed=1,
        std=None,
        status="technical_probe",
        reasons=["probe_only", "missing_seed"],
    )
    names = {"sarpa", "orapa", "probe"}

    workbook = build_workbook(_report(sar_pa, oracle_pa, probe))

    for sheet in GROUP_SHEETS:
        assert not names & set(_flat(workbook, sheet))
    assert names <= set(_flat(workbook, DIAGNOSTIC_SHEET))
    assert "probe" not in _flat(workbook, "coverage")


def test_edge_a_partial_row_keeps_its_numbers_but_says_it_is_not_ranked():
    workbook = build_workbook(
        _report(_row(method="p", status="partial", reasons=["extra_clean_support"]))
    )

    lines = [line for line in _cells(workbook, GROUP_SHEETS[0]) if "p" in line]

    assert len(lines) == 1
    assert "90.00 ± 1.00" in lines[0]
    assert any("不得排名" in str(cell) for cell in lines[0])


def test_param_an_unknown_mechanism_is_refused_not_dropped():
    with pytest.raises(WorkbookError, match="mechanism"):
        build_workbook(_report(_row(mechanism="mystery")))


def test_param_an_unknown_selection_protocol_is_refused_not_dropped():
    with pytest.raises(WorkbookError, match="selection protocol"):
        build_workbook(_report(_row(protocol="XX")))


def test_basic_the_readme_sheet_names_the_protocol_and_the_no_ranking_rule():
    flat = [str(cell) for cell in _flat(build_workbook(_report(_row())), "readme")]

    assert "survey-v3.0" in flat
    assert "ab" * 32 in flat
    assert any("不生成总排名" in cell for cell in flat)
    assert any("单 seed" in cell for cell in flat)


def test_determ_the_same_report_builds_the_same_cells():
    report = _report(_row(method="b"), _row(method="a", dataset="adni"))

    first = build_workbook(report)
    second = build_workbook(report)

    assert first.sheetnames == second.sheetnames
    for sheet in first.sheetnames:
        assert _cells(first, sheet) == _cells(second, sheet)


def test_edge_an_empty_report_still_builds_every_sheet():
    workbook = build_workbook(_report())

    assert {*GROUP_SHEETS, DIAGNOSTIC_SHEET, "coverage", "readme"} <= set(workbook.sheetnames)


def test_edge_the_coverage_sheet_has_one_line_per_row_with_no_ranking_notes():
    rows = [
        _row(method=f"m{i}", observed=1, std=None, status="partial", reasons=["missing_seed"])
        for i in range(3)
    ]
    cells = _cells(build_workbook(_report(*rows)), "coverage")
    assert sum(1 for line in cells if line and line[0] == "partial") == 3
    assert not any(line and "不得排名" in str(line[0]) for line in cells)
