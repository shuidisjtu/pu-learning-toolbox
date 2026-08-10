# ruff: noqa: N802, N803, N806, E501

"""Tests for the pu-workflow profile step script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers import make_scar_data


def _write_inputs(tmp_path, rng, n=60):
    """Write X.csv / y_pu.csv / y_true.csv and return their paths."""
    X, y_pu, y_true = make_scar_data(rng, n=n, separation=4.0)
    # Non-numeric header required by the CSV contract (_load_features rejects
    # numeric-first-row CSVs as headerless).
    pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]).to_csv(
        tmp_path / "X.csv", index=False
    )
    pd.DataFrame({"label": y_pu}).to_csv(tmp_path / "y_pu.csv", index=False)
    pd.DataFrame({"label": y_true}).to_csv(tmp_path / "y_true.csv", index=False)
    return tmp_path / "X.csv", tmp_path / "y_pu.csv", tmp_path / "y_true.csv"


def _run_script(tmp_path, *extra, out_dir=None):
    """Run the profile script in a subprocess; return (exit_code, stderr, out_dir)."""
    out = tmp_path / "out" if out_dir is None else Path(out_dir)
    proc = subprocess.run(
        [sys.executable, "scripts/pu_workflow/profile.py", *extra, "--out-dir", str(out)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    return proc.returncode, proc.stderr, out


@pytest.mark.e2e
def test_basic_profile_writes_contract_json(tmp_path, rng):
    """Output profile.json parses, carries the SCAR/SAR diagnostic and issues."""
    X, y_pu, y_true = _write_inputs(tmp_path, rng)
    code, _, out = _run_script(tmp_path, "--data", str(X), "--labels", str(y_pu))
    assert code == 0
    payload = json.loads((out / "profile.json").read_text(encoding="utf-8"))
    assert "selection_diagnostic" in payload
    assert "issues" in payload
    assert isinstance(payload["issues"], list)


@pytest.mark.e2e
def test_param_true_labels_optional(tmp_path, rng):
    """Without --true-labels the profile still completes (oracle hints absent)."""
    X, y_pu, _ = _write_inputs(tmp_path, rng)
    code, _, out = _run_script(tmp_path, "--data", str(X), "--labels", str(y_pu))
    assert code == 0
    assert (out / "profile.json").exists()


@pytest.mark.e2e
def test_edge_missing_file_reports_error(tmp_path):
    """Missing input file -> exit 1 with a clear stderr message."""
    code, stderr, out = _run_script(
        tmp_path, "--data", str(tmp_path / "nope.csv"), "--labels", str(tmp_path / "y.csv")
    )
    assert code == 1
    assert "error:" in stderr
    assert not (out / "profile.json").exists()


@pytest.mark.e2e
def test_edge_unwritable_out_dir_reports_error(tmp_path, rng):
    """--out-dir pointing at an existing file -> exit 1, clean error, no traceback."""
    X, y_pu, _ = _write_inputs(tmp_path, rng)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    code, stderr, _ = _run_script(
        tmp_path, "--data", str(X), "--labels", str(y_pu), out_dir=blocker
    )
    assert code == 1
    assert "error:" in stderr
    assert "Traceback" not in stderr


@pytest.mark.e2e
def test_deterministic_profile_same_seed(tmp_path, rng):
    """Same seed -> identical profile.json bytes."""
    X, y_pu, _ = _write_inputs(tmp_path, rng)
    _, _, out1 = _run_script(tmp_path, "--data", str(X), "--labels", str(y_pu), "--seed", "7")
    _, _, out2 = _run_script(tmp_path, "--data", str(X), "--labels", str(y_pu), "--seed", "7")
    assert (out1 / "profile.json").read_bytes() == (out2 / "profile.json").read_bytes()
