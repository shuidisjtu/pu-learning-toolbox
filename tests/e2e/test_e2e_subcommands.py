# ruff: noqa: N802, N803, N806, E501

"""Real-subprocess e2e: profile / recommend / sensitivity CLI subcommands.

The script-level journeys stay covered by test_{profile,recommend,
sensitivity}_script.py (compatibility wrappers); this file covers the
CLI-subcommand entry points the skill now uses.
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


def test_basic_profile_recommend_chain(tmp_path, rng):
    """pu-toolbox profile -> recommend chain (skill Steps 1/2 entry points)."""
    X, y_pu = _write_inputs(tmp_path, rng)
    proc = _cli("profile", "--data", str(X), "--labels", str(y_pu), "--out-dir", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    profile = tmp_path / "profile.json"
    assert profile.exists()

    proc = _cli(
        "recommend",
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


def test_basic_sensitivity_smoke(tmp_path, rng):
    """pu-toolbox sensitivity writes a strict-JSON analysis (skill Step 4)."""
    X, y_pu = _write_inputs(tmp_path, rng)
    proc = _cli("sensitivity", "--data", str(X), "--labels", str(y_pu), "--out-dir", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    payload = json.loads((tmp_path / "sensitivity.json").read_text(encoding="utf-8"))
    assert payload["analysis_type"] == "fixed_output_assumption_sensitivity"
    assert len(payload["points"]) == 9


def test_basic_skill_install_writes_both_targets(tmp_path):
    """pu-toolbox skill install copies SKILL.md into both agent dirs."""
    proc = _cli("skill", "install", "--dest", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "installed pu-workflow skill" in proc.stdout
    for target in (".claude/skills/pu-workflow", ".agents/skills/pu-workflow"):
        assert (tmp_path / target / "SKILL.md").is_file()


def test_param_run_accepts_prior_param(tmp_path, rng):
    """--prior-param KEY=VALUE is accepted by the real CLI end-to-end."""
    X, y_pu, _ = make_scar_data(rng, n=120, separation=4.0)
    pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]).to_csv(
        tmp_path / "X.csv", index=False
    )
    pd.DataFrame({"label": y_pu}).to_csv(tmp_path / "y_pu.csv", index=False)
    proc = _cli(
        "run",
        "--data",
        str(tmp_path / "X.csv"),
        "--labels",
        str(tmp_path / "y_pu.csv"),
        "--out-dir",
        str(tmp_path / "out"),
        "--prior-estimator",
        "pen_l1",
        "--prior-param",
        "sigma=3.0",
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "out" / "report.json").is_file()


def test_param_missing_profile_reports_error(tmp_path, rng):
    """Missing profile.json -> exit 1, error: on stderr, no traceback."""
    X, y_pu = _write_inputs(tmp_path, rng)
    proc = _cli(
        "recommend",
        "--profile",
        str(tmp_path / "none.json"),
        "--out-dir",
        str(tmp_path / "out"),
    )
    assert proc.returncode == 1
    assert "error:" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_edge_headerless_csv_reports_error(tmp_path):
    """Headerless CSV -> exit 1 with a clear error, no traceback."""
    X_path = tmp_path / "noheader.csv"
    X_path.write_text("1.0,2.0\n3.0,4.0\n5.0,6.0\n", encoding="utf-8")
    y_path = tmp_path / "y.csv"
    y_path.write_text("label\n1\n0\n0\n", encoding="utf-8")
    proc = _cli(
        "profile", "--data", str(X_path), "--labels", str(y_path), "--out-dir", str(tmp_path)
    )
    assert proc.returncode == 1
    assert "error:" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_determ_profile_reproducible(tmp_path, rng):
    """Same inputs -> byte-identical profile.json across runs."""
    X, y_pu = _write_inputs(tmp_path, rng)
    outs = []
    for name in ("a", "b"):
        proc = _cli(
            "profile", "--data", str(X), "--labels", str(y_pu), "--out-dir", str(tmp_path / name)
        )
        assert proc.returncode == 0, proc.stderr
        outs.append((tmp_path / name / "profile.json").read_bytes())
    assert outs[0] == outs[1]
