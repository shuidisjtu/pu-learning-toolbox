# ruff: noqa: N802, N803, N806, E501

"""Real-subprocess end-to-end journeys: quickstart CLI flow + CLI error paths.

Journeys mirror ``docs/user/quickstart.md`` exactly (demo data -> auto run ->
report), plus error journeys asserting exit code 1, ``error:`` stderr and no
traceback.  Uses a fresh subprocess for the real ``pu_toolbox.cli.main`` entry
(the package has no ``__main__.py``, so ``python -c`` drives it).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def _assert_strict_report(path: Path) -> dict:
    """report.json must be strict JSON (no NaN/Infinity literals) with schema 1.0."""

    def _reject_constant(name: str) -> None:
        raise ValueError(f"non-standard JSON literal in report: {name}")

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh, parse_constant=_reject_constant)
    assert payload["schema_version"] == "1.0"
    return payload


def test_basic_quickstart_demo_to_report(tmp_path):
    """The documented quickstart journey: make-demo-data -> auto run -> report."""
    demo = tmp_path / "demo"
    proc = _cli("make-demo-data", "--out-dir", str(demo), "--n", "200", "--seed", "42")
    assert proc.returncode == 0, proc.stderr
    for name in ("X.csv", "y_pu.csv", "y_true.csv"):
        assert (demo / name).exists(), name

    results = tmp_path / "results"
    proc = _cli(
        "run",
        "--data",
        str(demo / "X.csv"),
        "--labels",
        str(demo / "y_pu.csv"),
        "--out-dir",
        str(results),
    )
    assert proc.returncode == 0, proc.stderr
    payload = _assert_strict_report(results / "report.json")
    assert "pu_zero_one_risk" in payload["cv_metrics"]
    md = (results / "report.md").read_text(encoding="utf-8")
    assert "PU Pipeline Report" in md


def test_determ_quickstart_report_reproducible(tmp_path):
    """Same seed + same inputs must produce identical report content."""
    demo = tmp_path / "demo"
    proc = _cli("make-demo-data", "--out-dir", str(demo), "--n", "120", "--seed", "42")
    assert proc.returncode == 0, proc.stderr

    payloads = []
    for i in (1, 2):
        out = tmp_path / f"r{i}"
        proc = _cli(
            "run",
            "--data",
            str(demo / "X.csv"),
            "--labels",
            str(demo / "y_pu.csv"),
            "--out-dir",
            str(out),
            "--classifier",
            "upu",
            "--cv",
            "3",
        )
        assert proc.returncode == 0, proc.stderr
        payloads.append(_assert_strict_report(out / "report.json"))
    assert payloads[0]["cv_metrics"] == payloads[1]["cv_metrics"]
    assert payloads[0]["prior"] == payloads[1]["prior"]


def test_param_run_invalid_labels_reports_error(tmp_path):
    """A two-column labels CSV violates the single-column contract (exit 1)."""
    demo = tmp_path / "demo"
    proc = _cli("make-demo-data", "--out-dir", str(demo), "--n", "60", "--seed", "42")
    assert proc.returncode == 0, proc.stderr

    labels = pd.read_csv(demo / "y_pu.csv")
    pd.DataFrame({"label": labels["label"], "extra": labels["label"]}).to_csv(
        tmp_path / "bad_labels.csv", index=False
    )
    proc = _cli(
        "run",
        "--data",
        str(demo / "X.csv"),
        "--labels",
        str(tmp_path / "bad_labels.csv"),
        "--out-dir",
        str(tmp_path / "out"),
    )
    assert proc.returncode == 1
    assert "error:" in proc.stderr
    assert "single column" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_edge_headerless_csv_reports_error(tmp_path):
    """A numeric first row is indistinguishable from data -> header error."""
    demo = tmp_path / "demo"
    proc = _cli("make-demo-data", "--out-dir", str(demo), "--n", "60", "--seed", "42")
    assert proc.returncode == 0, proc.stderr

    X = pd.read_csv(demo / "X.csv")
    np.savetxt(tmp_path / "headerless.csv", X.to_numpy(), delimiter=",")
    proc = _cli(
        "run",
        "--data",
        str(tmp_path / "headerless.csv"),
        "--labels",
        str(demo / "y_pu.csv"),
        "--out-dir",
        str(tmp_path / "out"),
    )
    assert proc.returncode == 1
    assert "error:" in proc.stderr
    assert "header row" in proc.stderr
    assert "Traceback" not in proc.stderr
