# ruff: noqa: N802, N803, N806, E501

"""Tests for the project_structure.md tree generator and --check/--update CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import generate_structure as g  # noqa: E402

pytestmark = pytest.mark.unit

DOC = """\
# Project Directory Structure

## 1. 项目根目录

```text
pu-learning-toolbox/
  pyproject.toml
```

## 2. Python 包（`pu_toolbox/`）

```text
pu_toolbox/
  core/
    base.py                  (core base)
  ui/
    app.py                   (Streamlit 页面)
"""

SAMPLE_BLOCK = [
    "pu_toolbox/",
    "  core/",
    "    base.py                  (core base)",
    "  ui/",
    "    app.py                   (Streamlit 页面)",
]


def _content(root: str) -> list[str]:
    lines = DOC.splitlines()
    for start, end, r in g.find_blocks(lines):
        if r == root:
            return lines[start + 1 : end]
    raise AssertionError(f"no block for {root}")


def test_find_blocks_identifies_generatable_roots():
    lines = DOC.splitlines()
    blocks = g.find_blocks(lines)
    roots = [r for _, _, r in blocks]
    assert roots == [None, "pu_toolbox"]  # §1 根目录块 None,§2 块 pu_toolbox


def test_parse_tree_records_files_and_dir_annotations():
    tree, dir_ann = g.parse_tree(SAMPLE_BLOCK, "pu_toolbox")
    assert tree["pu_toolbox"]["core"][g.FILES_KEY]["base.py"] == "(core base)"
    assert tree["pu_toolbox"]["ui"][g.FILES_KEY]["app.py"] == "(Streamlit 页面)"
    # 目录无注释时不记录
    assert "pu_toolbox/core" not in dir_ann


def test_parse_tree_records_dir_annotation():
    block = ["pu_toolbox/", "  cli/                   (CLI 入口)"]
    tree, dir_ann = g.parse_tree(block, "pu_toolbox")
    assert dir_ann["pu_toolbox/cli"] == "(CLI 入口)"


def test_build_new_from_rel_paths():
    tree = g.build_new(["core/base.py", "ui/app.py", "core/device.py"])
    assert set(tree["core"][g.FILES_KEY]) == {"base.py", "device.py"}
    assert set(tree["ui"][g.FILES_KEY]) == {"app.py"}


def test_merge_keeps_old_order_and_annotations():
    old, dir_ann = g.parse_tree(SAMPLE_BLOCK, "pu_toolbox")
    new = {"pu_toolbox": g.build_new(["core/base.py", "core/zeta.py", "ui/app.py"])}
    out: list[str] = []
    missing: list[str] = []
    g.merge_tree(old["pu_toolbox"], new["pu_toolbox"], dir_ann, "", "pu_toolbox", 1, out, missing)
    # 旧顺序 core 在 ui 前,注释保留
    assert out[0].startswith("  core/")
    assert "(core base)" in out[1]
    # 新文件 zeta.py 字母序追加到 core 目录末尾(core 旧顺序第一,故 zeta 在 out[2])
    assert out[2].startswith("    zeta.py") and g.PLACEHOLDER in out[2]
    assert missing == ["pu_toolbox/core/zeta.py"]


def test_merge_keeps_planned_entries():
    block = ["pu_toolbox/", "  future.py                (planned)"]
    old, dir_ann = g.parse_tree(block, "pu_toolbox")
    new = {"pu_toolbox": {}}
    out: list[str] = []
    missing: list[str] = []
    g.merge_tree(old["pu_toolbox"], new["pu_toolbox"], dir_ann, "", "pu_toolbox", 1, out, missing)
    assert len(out) == 1 and "(planned)" in out[0]
    assert missing == []


def test_merge_drops_stale_entries():
    block = ["pu_toolbox/", "  gone.py"]
    old, _ = g.parse_tree(block, "pu_toolbox")
    new = {"pu_toolbox": g.build_new(["core/base.py"])}
    out: list[str] = []
    missing: list[str] = []
    g.merge_tree(old["pu_toolbox"], new["pu_toolbox"], {}, "", "pu_toolbox", 1, out, missing)
    assert "gone.py" not in "".join(out)
    assert "  core/" in "".join(out)


def test_generate_returns_missing_and_stale():
    disk = ["pu_toolbox/core/base.py", "pu_toolbox/ui/app.py", "pu_toolbox/ui/new.py"]
    new_text, missing, stale = g.generate(DOC, disk)
    assert missing == ["pu_toolbox/ui/new.py"]
    assert stale == []  # 文档条目都在磁盘上
    assert "new.py" in new_text and g.PLACEHOLDER in new_text


def test_generate_reports_stale_entry():
    disk = ["pu_toolbox/ui/app.py"]
    new_text, missing, stale = g.generate(DOC, disk)
    assert stale == ["pu_toolbox/core/base.py"]


def test_generate_excludes_planned_from_stale():
    doc = """\
```text
pu_toolbox/
  core/
    base.py                  (core base)
    future.py                (planned)
```
"""
    disk = ["pu_toolbox/core/base.py"]
    _, missing, stale = g.generate(doc, disk)
    assert missing == []
    assert stale == []  # planned 条目不算 stale


def test_generate_preserves_non_generatable_blocks():
    disk = ["pu_toolbox/ui/app.py"]
    new_text, _, _ = g.generate(DOC, disk)
    assert "pyproject.toml" in new_text  # §1 根目录块原样保留
    assert "## 1. 项目根目录" in new_text  # 非块文字保留


def test_generate_is_idempotent():
    disk = ["pu_toolbox/core/base.py", "pu_toolbox/ui/app.py", "pu_toolbox/ui/new.py"]
    once, _, _ = g.generate(DOC, disk)
    twice, _, _ = g.generate(once, disk)
    assert once == twice


def test_collect_entries_flattens_to_repo_paths():
    tree, _ = g.parse_tree(SAMPLE_BLOCK, "pu_toolbox")
    assert g.collect_entries(tree) == {
        "pu_toolbox/core/base.py",
        "pu_toolbox/ui/app.py",
    }


def test_tracked_py_files_fallback_walk(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "b.py").write_text("", encoding="utf-8")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(g, "PROJECT_ROOT", tmp_path)
    try:
        files = g.tracked_py_files()
    finally:
        monkeypatch.undo()
    assert "src/a.py" in files
    assert not any("b.py" in f for f in files)  # .venv 排除


def test_main_check_fails_on_drift(tmp_path, monkeypatch):
    md = tmp_path / "project_structure.md"
    md.write_text(DOC, encoding="utf-8")
    monkeypatch.setattr(g, "STRUCTURE_MD", md)
    monkeypatch.setattr(g, "tracked_py_files", lambda: ["pu_toolbox/ui/new.py"])
    assert g.main(["--check"]) == 1


def test_main_check_passes_when_synced(tmp_path, monkeypatch):
    disk = ["pu_toolbox/core/base.py", "pu_toolbox/ui/app.py"]
    new_text, _, _ = g.generate(DOC, disk)
    md = tmp_path / "project_structure.md"
    md.write_text(new_text, encoding="utf-8")
    monkeypatch.setattr(g, "STRUCTURE_MD", md)
    monkeypatch.setattr(g, "tracked_py_files", lambda: disk)
    assert g.main(["--check"]) == 0


def test_main_update_writes_document(tmp_path, monkeypatch):
    md = tmp_path / "project_structure.md"
    md.write_text(DOC, encoding="utf-8")
    monkeypatch.setattr(g, "STRUCTURE_MD", md)
    monkeypatch.setattr(g, "tracked_py_files", lambda: ["pu_toolbox/ui/new.py"])
    assert g.main(["--update"]) == 0
    written = md.read_text(encoding="utf-8")
    assert "new.py" in written and g.PLACEHOLDER in written
