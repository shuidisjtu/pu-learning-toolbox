"""Tests for the API-doc coverage gate (anti-drift).

The gate must extract symbols statically (no package import, so it runs
in CI without optional deep deps), scan the real api.md from any cwd, and
fail when a public symbol is missing from the reference doc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_api_docs as m  # noqa: E402


@pytest.mark.unit
def test_extract_all_exports_parses_real_init():
    """Static parse yields the package-root exports (no import, no torch dependency)."""
    exports = m.extract_all_exports(m.PROJECT_ROOT)
    assert "PUPipeline" in exports
    assert "build_encoder" in exports
    assert "pe" not in exports  # 'pe' is a deprecated alias, not an export


@pytest.mark.unit
def test_extract_registered_names_skips_deprecated_aliases():
    """Registry names parse statically; deprecated aliases are not canonical names."""
    names = m.extract_registered_names(m.PROJECT_ROOT)
    assert "class_prior_estimation" in names
    assert "nnpu" in names
    assert "pe" not in names


@pytest.mark.unit
def test_real_api_doc_covers_every_symbol():
    """The real api.md passes the gate from any cwd."""
    rc = m.main([])
    assert rc == 0


@pytest.mark.unit
def test_deterministic_scan_repeatable(tmp_path, monkeypatch):
    """Repeated scans give identical results (static AST, no RNG)."""
    monkeypatch.chdir(tmp_path)
    rc1 = m.main([])
    rc2 = m.main([])
    assert (rc1, rc2) == (0, 0)


@pytest.mark.unit
def test_missing_symbol_fails(tmp_path, monkeypatch, capsys):
    """A doc that omits a public symbol must exit 1."""
    doc = tmp_path / "api.md"
    doc.write_text("# API 参考\n\n只有部分符号 PUPipeline。\n", encoding="utf-8")
    monkeypatch.setattr(m, "API_DOC", doc)
    rc = m.main([])
    assert rc == 1
    out = capsys.readouterr().out
    assert "Missing" in out
    assert "UPUClassifier" in out


@pytest.mark.unit
def test_empty_scan_refuses(tmp_path, monkeypatch, capsys):
    """Unparseable __all__ must exit 1 with a stderr message, never fake-green."""
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    rc = m.main([])
    assert rc == 1
    assert "refusing to pass empty scan" in capsys.readouterr().err
