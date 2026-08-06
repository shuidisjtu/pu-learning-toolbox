# ruff: noqa: N802, N803, N806, E501

"""Tests for the pu-workflow sensitivity step script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import make_scar_data


def _write_inputs(tmp_path, rng):
    X, y_pu, y_true = make_scar_data(rng, n=120, separation=4.0)
    # Non-numeric header row required: the CLI's CSV contract rejects
    # all-numeric first rows (indistinguishable from data) as headerless.
    pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]).to_csv(
        tmp_path / "X.csv", index=False
    )
    pd.DataFrame({"label": y_pu}).to_csv(tmp_path / "y_pu.csv", index=False)
    return tmp_path / "X.csv", tmp_path / "y_pu.csv"


def _run_script(tmp_path, *extra):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "scripts/pu_workflow/sensitivity.py", *extra, "--out-dir", str(out)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    return proc.returncode, proc.stderr, out


@pytest.mark.unit
def test_basic_sensitivity_writes_contract_json(tmp_path, rng):
    """Output sensitivity.json parses, with points and metric_ranges."""
    X, y_pu = _write_inputs(tmp_path, rng)
    code, _, out = _run_script(tmp_path, "--data", str(X), "--labels", str(y_pu))
    assert code == 0
    payload = json.loads((out / "sensitivity.json").read_text(encoding="utf-8"))
    assert payload["analysis_type"] == "fixed_output_assumption_sensitivity"
    assert len(payload["points"]) == 9  # default 0.1..0.9 grid
    assert "metric_ranges" in payload


@pytest.mark.unit
def test_param_custom_grids(tmp_path, rng):
    """Explicit --class-priors and --label-propensities override defaults."""
    X, y_pu = _write_inputs(tmp_path, rng)
    code, _, out = _run_script(
        tmp_path,
        "--data",
        str(X),
        "--labels",
        str(y_pu),
        "--class-priors",
        "0.2,0.4",
        "--label-propensities",
        "0.3,0.7",
    )
    assert code == 0
    payload = json.loads((out / "sensitivity.json").read_text(encoding="utf-8"))
    assert len(payload["points"]) == 4


@pytest.mark.unit
def test_edge_invalid_grid_reports_error(tmp_path, rng):
    """A non-numeric grid value -> exit 1 with a clear stderr message."""
    X, y_pu = _write_inputs(tmp_path, rng)
    code, stderr, out = _run_script(
        tmp_path, "--data", str(X), "--labels", str(y_pu), "--class-priors", "0.2,banana"
    )
    assert code == 1
    assert "error:" in stderr
    assert not (out / "sensitivity.json").exists()


@pytest.mark.unit
def test_deterministic_sensitivity_same_input(tmp_path, rng):
    """Same inputs -> identical sensitivity.json bytes."""
    X, y_pu = _write_inputs(tmp_path, rng)
    _, _, out1 = _run_script(tmp_path, "--data", str(X), "--labels", str(y_pu))
    _, _, out2 = _run_script(tmp_path, "--data", str(X), "--labels", str(y_pu))
    assert (out1 / "sensitivity.json").read_bytes() == (out2 / "sensitivity.json").read_bytes()
