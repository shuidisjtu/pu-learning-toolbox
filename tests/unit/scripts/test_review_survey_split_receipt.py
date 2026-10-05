"""Fail-closed checks for the read-only P1.4 receipt audit entry point."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/review_survey_split_receipt.py"

pytestmark = pytest.mark.unit


def run_audit(*args, optimized=False):
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_basic_determ_help_does_not_require_received_data():
    result = run_audit("--help")
    assert result.returncode == 0
    assert "--archive-dir" in result.stdout
    repeated = run_audit("--help")
    assert repeated.returncode == result.returncode
    assert repeated.stdout == result.stdout


def test_param_optimized_python_is_refused(tmp_path):
    result = run_audit("--archive-dir", str(tmp_path), optimized=True)
    assert result.returncode != 0
    assert "Assertions are required" in result.stderr


def test_edge_missing_receipt_fails_before_array_loading(tmp_path):
    result = run_audit("--archive-dir", str(tmp_path), "--splits", str(tmp_path / "missing"))
    assert result.returncode != 0
    assert "Receipt integrity check failed" in result.stderr
    assert '"ok": true' not in result.stdout
