# ruff: noqa: N802, N803, N806, E501

"""Tests for the ruff lint+format gate (6th quality gate).

The last two tests guard the derived ``SCOPE``: the index query must pick
up a new top-level root, and either way of losing git must fall back to
``FALLBACK_SCOPE`` with a notice instead of scanning nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_format as c  # noqa: E402
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


@pytest.mark.unit
def test_edge_lost_git_falls_back_with_notice(monkeypatch, capsys):
    """No git at all, and a git that answers with a failure, both fall back."""

    def no_git(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(c.subprocess, "run", no_git)
    assert c._tracked_source_roots() == c.FALLBACK_SCOPE
    assert "FALLBACK_SCOPE" in capsys.readouterr().err

    class Failed:
        # A failing git can still print junk on stdout; the exit code alone
        # must trigger the fallback, so the fake stdout is not empty.
        returncode = 128
        stdout = "leftover/from_a_bad_run.py\n"

    monkeypatch.setattr(c.subprocess, "run", lambda *args, **kwargs: Failed())
    assert c._tracked_source_roots() == c.FALLBACK_SCOPE
    assert "FALLBACK_SCOPE" in capsys.readouterr().err


@pytest.mark.unit
def test_param_index_entries_map_to_sorted_roots(monkeypatch):
    """A new top-level root in the index is covered, in deterministic order."""

    class Index:
        returncode = 0
        stdout = (
            "tests/unit/t.py\n"
            "tools/new_root/n.py\n"
            "scripts/s.py\n"
            "pu_toolbox/__init__.py\n"
            "examples/e1.py\n"
            "benchmarks/b1.py\n"
        )

    monkeypatch.setattr(c.subprocess, "run", lambda *args, **kwargs: Index())
    roots = c._tracked_source_roots()
    assert roots == tuple(sorted(roots))
    assert roots == ("benchmarks", "examples", "pu_toolbox", "scripts", "tests", "tools")
