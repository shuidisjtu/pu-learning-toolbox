# ruff: noqa: N802, N803, N806, E501

"""Real-subprocess end-to-end journeys for demo and profile scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers import make_scar_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.e2e]


def _cli(*args: str) -> subprocess.CompletedProcess:
    """Run the real CLI in a fresh subprocess from the repo root."""
    return subprocess.run(
        [sys.executable, "-c", "from pu_toolbox.cli import main; main()", *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


def _write_inputs(tmp_path, rng):
    """Write X.csv / y_pu.csv with the CLI's non-numeric-header contract."""
    X, y_pu, _ = make_scar_data(rng, n=120, separation=4.0)
    pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]).to_csv(
        tmp_path / "X.csv", index=False
    )
    pd.DataFrame({"label": y_pu}).to_csv(tmp_path / "y_pu.csv", index=False)
    return tmp_path / "X.csv", tmp_path / "y_pu.csv"


def _run_script(*args: str) -> subprocess.CompletedProcess:
    """Run a workflow script in a fresh subprocess from the repo root."""
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


def test_basic_profile_script_writes_profile(tmp_path, rng):
    """profile.py produces a profile for the supplied data."""
    X, y_pu = _write_inputs(tmp_path, rng)
    proc = _run_script(
        "scripts/pu_workflow/profile.py",
        "--data",
        str(X),
        "--labels",
        str(y_pu),
        "--out-dir",
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    profile = tmp_path / "profile.json"
    assert profile.exists()


def test_basic_demo_to_profile(tmp_path, rng):
    """Demo data is consumable by the standalone profile script."""
    demo = tmp_path / "demo"
    proc = _cli("make-demo-data", "--out-dir", str(demo), "--n", "60", "--seed", "42")
    assert proc.returncode == 0, proc.stderr

    proc = _run_script(
        "scripts/pu_workflow/profile.py",
        "--data",
        str(demo / "X.csv"),
        "--labels",
        str(demo / "y_pu.csv"),
        "--out-dir",
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr

    assert (tmp_path / "profile.json").exists()
