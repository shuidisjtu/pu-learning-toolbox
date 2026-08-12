"""Tests for the benchmark audit CLI."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from pu_toolbox.cli import build_parser, main
from pu_toolbox.cli.audit_benchmark import run_audit_benchmark
from pu_toolbox.diagnostics import BenchmarkAuditReport

pytestmark = pytest.mark.unit


def _report(tmp_path, *, passed=True):
    return BenchmarkAuditReport(
        result_dir=str(tmp_path),
        protocol="clean_room",
        paper_claim=False,
        passed=passed,
        n_trials=3,
        n_methods=1,
        n_seeds=3,
        checks={"required_artifacts": passed},
        errors=() if passed else ("trials.csv contains duplicate trial rows",),
        warnings=("paper_claim=false",),
    )


def test_basic_parser_registers_required_result_directory():
    args = build_parser().parse_args(["audit-benchmark", "--result-dir", "results"])
    assert args.command == "audit-benchmark"
    assert args.result_dir == "results"
    assert args.output is None


def test_basic_pass_prints_summary_and_writes_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "pu_toolbox.cli.audit_benchmark.audit_benchmark_results",
        lambda _: _report(tmp_path),
    )
    output = tmp_path / "audit.json"
    run_audit_benchmark(Namespace(result_dir="results", output=str(output)))
    assert "Benchmark audit: PASS" in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_edge_failed_audit_maps_to_cli_exit_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "pu_toolbox.cli.audit_benchmark.audit_benchmark_results",
        lambda _: _report(tmp_path, passed=False),
    )
    with pytest.raises(SystemExit) as exc:
        main(["audit-benchmark", "--result-dir", "results"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Benchmark audit: FAIL" in captured.out
    assert "duplicate trial rows" in captured.out
    assert "error: benchmark audit failed" in captured.err


def test_param_missing_result_directory_is_argparse_error():
    with pytest.raises(SystemExit) as exc:
        main(["audit-benchmark"])
    assert exc.value.code == 2


def test_determ_repeated_success_output_is_stable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "pu_toolbox.cli.audit_benchmark.audit_benchmark_results",
        lambda _: _report(tmp_path),
    )
    args = Namespace(result_dir="results", output=None)
    run_audit_benchmark(args)
    first = capsys.readouterr().out
    run_audit_benchmark(args)
    assert capsys.readouterr().out == first
