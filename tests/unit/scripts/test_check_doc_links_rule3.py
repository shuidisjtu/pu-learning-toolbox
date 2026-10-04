"""Tests for rule-3: architecture.md S8 vs the registry NATIVE methods.

Rule-3 used to degrade to a warning whenever it could not extract the
NATIVE set, and ``main`` only counts errors -- so an absent registry file
or a renamed ``_native_imports`` marker disabled the whole rule while the
gate stayed green.  These tests pin the opposite: an unextractable set
fails the gate.

Separate file because ``test_check_doc_links.py`` sits at the per-file
test budget (``scripts/check_test_quality.py``, max 15); rule-5/rule-2
tests stay there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_doc_links as d  # noqa: E402


def _rule3_only_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``main()`` at a scratch tree where rule-3 is the only live rule.

    Every other rule is stubbed to an empty issue list, so a non-zero exit
    is attributable to rule-3 alone -- which is what makes these tests
    sensitive to the withdrawal of the warning downgrade.
    """
    arch = tmp_path / "docs" / "dev" / "architecture.md"
    arch.parent.mkdir(parents=True, exist_ok=True)
    arch.write_text("## 8. 论文方法到模块的映射\n", encoding="utf-8")
    monkeypatch.setattr(d, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(d, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(d, "_find_md_files", lambda: [])
    monkeypatch.setattr(d, "check_path_references", lambda files: [])
    monkeypatch.setattr(d, "check_planned_consistency", lambda path: [])
    monkeypatch.setattr(d, "check_index_completeness", lambda *a, **k: [])
    monkeypatch.setattr(d, "check_md_links", lambda files: [])
    return arch


def _registry_under(tmp_path: Path, body: str) -> Path:
    """Write a scratch registry file under *tmp_path* and return its path."""
    registry = tmp_path / "pu_toolbox" / "registry" / "builtin_methods.py"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(body, encoding="utf-8")
    return registry


@pytest.mark.unit
def test_basic_extraction_reads_the_real_registry():
    """The real registry still yields NATIVE paths (the rule is live)."""
    paths = d._get_native_module_paths()
    assert paths
    assert "estimators/classic/elkan_noto.py" in paths


@pytest.mark.unit
def test_param_missing_registry_file_fails_the_gate(tmp_path, monkeypatch, capsys):
    """A missing registry file must fail the gate, not skip the rule."""
    _rule3_only_root(tmp_path, monkeypatch)
    monkeypatch.setattr(d, "REGISTRY_FILE", tmp_path / "pu_toolbox" / "registry" / "gone.py")
    rc = d.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "[ERROR]" in out
    assert "could not extract NATIVE paths" in out


@pytest.mark.unit
def test_param_missing_extraction_marker_fails_the_gate(tmp_path, monkeypatch, capsys):
    """A registry whose ``_native_imports`` marker is gone fails the gate."""
    _rule3_only_root(tmp_path, monkeypatch)
    monkeypatch.setattr(d, "REGISTRY_FILE", _registry_under(tmp_path, "METHODS: dict = {}\n"))
    rc = d.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "[ERROR]" in out
    assert "could not extract NATIVE paths" in out


@pytest.mark.unit
def test_edge_empty_native_import_block_fails_the_gate(tmp_path, monkeypatch):
    """An empty extraction is a broken rule, not "no native methods".

    The marker exists but binds nothing, so the set is empty for a reason
    the mapping check cannot distinguish from a real absence; it must be
    reported rather than passed.
    """
    arch = _rule3_only_root(tmp_path, monkeypatch)
    monkeypatch.setattr(
        d, "REGISTRY_FILE", _registry_under(tmp_path, "_native_imports: list = []\n")
    )
    issues = d.check_architecture_mapping(arch)
    assert [(i.rule, i.severity) for i in issues] == [("rule-3", "error")]


@pytest.mark.unit
def test_determ_repeated_rule3_checks_agree(tmp_path, monkeypatch):
    """Rule-3 is a pure function of the registry text and architecture.md."""
    monkeypatch.setattr(d, "REGISTRY_FILE", tmp_path / "absent.py")
    arch = tmp_path / "architecture.md"
    arch.write_text("## 8. 论文方法到模块的映射\n", encoding="utf-8")
    assert d.check_architecture_mapping(arch) == d.check_architecture_mapping(arch)
