# ruff: noqa: N802, N803, N806, E501

"""Real-subprocess end-to-end journeys: workflow-script chain + full chain.

Covers the pu-workflow script pipeline (profile -> recommend) and the full
demo -> profile -> recommend -> run chain, all through real subprocesses
against the real entry points with real file IO.
"""

from __future__ import annotations

import json
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


def _run_report_payload(demo: Path, out_dir: Path) -> dict:
    """Run the CLI on demo data with cheap settings; return parsed report."""
    proc = _cli(
        "run",
        "--data",
        str(demo / "X.csv"),
        "--labels",
        str(demo / "y_pu.csv"),
        "--out-dir",
        str(out_dir),
        "--classifier",
        "upu",
        "--cv",
        "3",
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads((out_dir / "report.json").read_text(encoding="utf-8"))


def test_basic_profile_recommend_chain(tmp_path, rng):
    """profile.py -> recommend.py chain produces a non-empty candidate list."""
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

    proc = _run_script(
        "scripts/pu_workflow/recommend.py",
        "--profile",
        str(profile),
        "--data",
        str(X),
        "--labels",
        str(y_pu),
        "--prior-estimator",
        "recpe",
        "--out-dir",
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    rec = json.loads((tmp_path / "recommendation.json").read_text(encoding="utf-8"))
    assert len(rec["candidates"]) > 0


def test_edge_full_chain_demo_to_report(tmp_path, rng):
    """Full journey: demo -> profile -> recommend -> run -> report."""
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

    proc = _run_script(
        "scripts/pu_workflow/recommend.py",
        "--profile",
        str(tmp_path / "profile.json"),
        "--data",
        str(demo / "X.csv"),
        "--labels",
        str(demo / "y_pu.csv"),
        "--prior-estimator",
        "recpe",
        "--out-dir",
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "recommendation.json").exists()

    payload = _run_report_payload(demo, tmp_path / "results")
    assert "cv_metrics" in payload
    assert (tmp_path / "results" / "report.md").exists()


def test_param_full_chain_bad_classifier_reports_error(tmp_path, rng):
    """Unknown classifier name fails at pipeline construction (exit 1)."""
    X, y_pu = _write_inputs(tmp_path, rng)
    proc = _cli(
        "run",
        "--data",
        str(X),
        "--labels",
        str(y_pu),
        "--out-dir",
        str(tmp_path / "out"),
        "--classifier",
        "not_a_method",
    )
    assert proc.returncode == 1
    assert "error:" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_determ_full_chain_report_reproducible(tmp_path, rng):
    """The full chain with the same seed is reproducible end to end."""
    demo = tmp_path / "demo"
    proc = _cli("make-demo-data", "--out-dir", str(demo), "--n", "60", "--seed", "42")
    assert proc.returncode == 0, proc.stderr

    payloads = [_run_report_payload(demo, tmp_path / f"r{i}") for i in (1, 2)]
    assert payloads[0]["cv_metrics"] == payloads[1]["cv_metrics"]
    assert payloads[0]["prior"] == payloads[1]["prior"]
