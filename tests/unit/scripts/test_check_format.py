# ruff: noqa: N802, N803, N806, E501

"""Tests for the ruff lint+format gate (6th quality gate)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from check_format import run_ruff  # noqa: E402


@pytest.mark.unit
def test_basic_well_formatted_file_passes(tmp_path):
    """A clean file passes both the lint and the format check."""
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    ok, out = run_ruff(["check"], cwd=tmp_path, scope=("clean.py",))
    assert ok, out
    ok, out = run_ruff(["format", "--check"], cwd=tmp_path, scope=("clean.py",))
    assert ok, out


@pytest.mark.unit
def test_param_format_violation_reported(tmp_path):
    """An unformatted file fails ``format --check`` but not ``check``."""
    (tmp_path / "messy.py").write_text("x=1+1\n", encoding="utf-8")
    ok, _ = run_ruff(["check"], cwd=tmp_path, scope=("messy.py",))
    assert ok  # lint rules do not cover spacing
    ok, out = run_ruff(["format", "--check"], cwd=tmp_path, scope=("messy.py",))
    assert not ok
    assert "messy.py" in out


@pytest.mark.unit
def test_param_lint_violation_reported(tmp_path):
    """An unused import fails ``ruff check`` (F401)."""
    (tmp_path / "bad.py").write_text("import os\nx = 1\n", encoding="utf-8")
    ok, out = run_ruff(["check"], cwd=tmp_path, scope=("bad.py",))
    assert not ok
    assert "F401" in out


@pytest.mark.unit
def test_edge_missing_scope_reported(tmp_path):
    """A scope path that does not exist is an error, not a silent pass."""
    ok, out = run_ruff(["check"], cwd=tmp_path, scope=("nope.py",))
    assert not ok
    assert "nope.py" in out


@pytest.mark.unit
def test_determ_results_reproducible(tmp_path):
    """run_ruff must be deterministic: repeated runs give identical verdicts."""
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    a = run_ruff(["format", "--check"], cwd=tmp_path, scope=("clean.py",))
    b = run_ruff(["format", "--check"], cwd=tmp_path, scope=("clean.py",))
    assert a == b
