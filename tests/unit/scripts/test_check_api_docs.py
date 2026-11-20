"""Tests for the API-doc coverage gate (anti-drift).

The gate must extract symbols statically (no package import, so it runs
in CI without optional deep deps), scan the API reference from any cwd, and
fail when a public symbol is missing from the contract pages.
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
    """The real API reference covers every checked symbol."""
    rc = m.main([])
    assert rc == 0


@pytest.mark.unit
def test_deterministic_scan_repeatable(tmp_path, monkeypatch):
    """Repeated scans give identical results (static AST, no RNG)."""
    monkeypatch.chdir(tmp_path)
    rc1 = m.main([])
    rc2 = m.main([])
    assert (rc1, rc2) == (0, 0)


def _write_reference(tmp_path, monkeypatch, **pages):
    monkeypatch.setattr(m, "API_DIR", tmp_path)
    for path, content in pages.items():
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


@pytest.mark.unit
def test_missing_symbol_fails_even_when_named_in_index(tmp_path, monkeypatch, capsys):
    _write_reference(
        tmp_path,
        monkeypatch,
        **{
            "index.md": "# API index: UPUClassifier",
            "classifiers.md": "# Classifiers",
            "tools.md": "# Tools: build_encoder",
            "errors.md": "# Errors",
        },
    )
    assert m.main([]) == 1
    out = capsys.readouterr().out
    assert "Missing" in out
    assert "API doc missing symbol: UPUClassifier" in out
    assert "API doc missing symbol: build_encoder" not in out


@pytest.mark.unit
@pytest.mark.parametrize("content", [None, " \n"])
def test_missing_or_empty_required_page_fails(tmp_path, monkeypatch, capsys, content):
    pages = {name: "# Reference" for name in m.REQUIRED_PAGES if name != "errors.md"}
    if content is not None:
        pages["errors.md"] = content
    _write_reference(tmp_path, monkeypatch, **pages)
    assert m.main([]) == 1
    assert "Required API reference page is missing or empty: errors.md" in capsys.readouterr().err


@pytest.mark.unit
def test_empty_reference_directory_fails(tmp_path, monkeypatch, capsys):
    _write_reference(tmp_path, monkeypatch)
    assert m.main([]) == 1
    assert "No Markdown pages found" in capsys.readouterr().err


@pytest.mark.unit
def test_recursively_reads_nested_contracts(tmp_path, monkeypatch):
    symbols = m.extract_all_exports(m.PROJECT_ROOT) + m.extract_registered_names(m.PROJECT_ROOT)
    pages = dict.fromkeys(m.REQUIRED_PAGES, "# Reference")
    pages["classifiers/details.md"] = "\n".join(symbols)
    _write_reference(tmp_path, monkeypatch, **pages)
    assert m.main([]) == 0


@pytest.mark.unit
def test_empty_scan_refuses(tmp_path, monkeypatch, capsys):
    """Unparseable __all__ must exit 1 with a stderr message, never fake-green."""
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    rc = m.main([])
    assert rc == 1
    assert "refusing to pass empty scan" in capsys.readouterr().err


@pytest.mark.unit
def test_empty_registry_refuses(tmp_path, monkeypatch, capsys):
    """A registry that yields no method name is not an empty coverage set.

    If ``builtin_methods.py`` is renamed, deleted, or stops matching the
    static parse, every registered method silently leaves the scan; the
    gate must refuse instead of passing on the reduced symbol list.
    """
    init = tmp_path / "pu_toolbox" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text('__all__ = ["build_encoder"]\n', encoding="utf-8")
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    rc = m.main([])
    assert rc == 1
    err = capsys.readouterr().err
    assert "refusing to pass empty scan" in err
    assert "builtin_methods.py" in err
