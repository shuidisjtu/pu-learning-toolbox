"""Tests for the math-rendering gate (PROJECT_ROOT anchoring).

Covers the architecture-decay fix: the gate must scan the real
method_cards dir no matter what cwd it is run from, and must refuse
to fake-green when the scan finds no files at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_math_rendering as m  # noqa: E402


@pytest.mark.unit
def test_scans_method_cards_from_any_cwd(tmp_path, monkeypatch):
    """Run from a temp cwd: still scans the real method_cards dir."""
    monkeypatch.chdir(tmp_path)
    rc = m.main([])
    assert rc == 0  # real cards were scanned, no math syntax issues


@pytest.mark.unit
def test_empty_scan_refuses_with_error(tmp_path, monkeypatch, capsys):
    """An empty scan must exit 1 with a stderr message, never fake-green."""
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    rc = m.main([])
    assert rc == 1
    assert "No method cards found" in capsys.readouterr().err


@pytest.mark.unit
def test_determ_scan_reproducible_from_any_cwd(tmp_path, monkeypatch, capsys):
    """Repeated scans from a temp cwd give identical results."""
    monkeypatch.chdir(tmp_path)
    rc1 = m.main([])
    out1 = capsys.readouterr().out
    rc2 = m.main([])
    out2 = capsys.readouterr().out
    assert (rc1, out1) == (rc2, out2)
