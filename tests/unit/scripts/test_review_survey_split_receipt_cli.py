"""Offline entry-point contracts for the receipt reviewer."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


def test_basic_determ_import_has_no_cli_side_effects():
    command = [sys.executable, "-c", "import scripts.review_survey_split_receipt"]
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1")
    first = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    second = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert first.returncode == second.returncode == 0, first.stderr
    assert first.stdout == second.stdout == ""


def test_basic_cli_exposes_dataset_and_seed_selection():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/review_survey_split_receipt.py"), "--help"],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--datasets" in result.stdout
    assert "--seeds" in result.stdout
