#!/usr/bin/env python3
"""Regenerate the tree blocks of docs/dev/project_structure.md.

The structure source is the set of git-tracked ``.py`` files
(``git ls-files``), which automatically excludes ``.venv``/``__pycache__``
and caches. Hand-written annotations and directory order are preserved from
the current document; files new on disk appear with a
``<<< 新文件,补注释`` placeholder so the missing annotation stays visible.

Usage::

    uv run python scripts/generate_structure.py --check   # verify (exit 0/1)
    uv run python scripts/generate_structure.py --update  # rewrite the document

``--check`` fails when the document tree differs from disk (missing or
stale entries, or formatting drift from this generator). ``--update``
rewrites the tree blocks in place and exits 0, printing the files that
still need a hand-written annotation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURE_MD = PROJECT_ROOT / "docs" / "dev" / "project_structure.md"
GENERATABLE_ROOTS = ("pu_toolbox", "tests")
COMMENT_COL = 42
PLACEHOLDER = "<<< 新文件,补注释"
FILES_KEY = "__files__"


def find_blocks(lines: list[str]) -> list[tuple[int, int, str | None]]:
    """Locate ```text fence blocks.

    Returns ``(start, end, rootname)`` triples where *start* is the fence
    opening line, *end* the fence closing line, and *rootname* the first
    entry of the block when it is one of GENERATABLE_ROOTS (``None`` for
    any other block, e.g. the root-directory listing).
    """
    blocks: list[tuple[int, int, str | None]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            root = None
            for ln in lines[i + 1 : j]:
                s = ln.strip()
                if not s:
                    continue
                first = s.split()[0]
                if first.rstrip("/") in GENERATABLE_ROOTS:
                    root = first.rstrip("/")
                break
            blocks.append((i, j, root))
            i = j + 1
        else:
            i += 1
    return blocks


def parse_tree(content: list[str], root: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse one tree block into (nested tree, directory annotations).

    The nested tree maps directory names to nested dicts with the special
    key ``__files__`` holding ``{file_name: annotation}``. Directory
    annotations map a repo-relative directory path (e.g.
    ``pu_toolbox/estimators/risk``) to its trailing annotation text.
    """
    tree: dict[str, Any] = {}
    dir_ann: dict[str, str] = {}
    stack: list[tuple[int, Any, str]] = []  # (indent, node, full path)
    for line in content:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        parts = line.strip().split(maxsplit=1)
        name = parts[0]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if name.endswith("/"):
            d = name.rstrip("/")
            node = stack[-1][1].setdefault(d, {}) if stack else tree.setdefault(d, {})
            full = d if not stack else stack[-1][2] + "/" + d
            stack.append((indent, node, full))
            if len(parts) > 1:
                dir_ann[full] = parts[1]
        elif name.endswith(".py"):
            target = stack[-1][1] if stack else tree
            files = target.setdefault(FILES_KEY, {})
            files[name] = parts[1] if len(parts) > 1 else ""
    return tree, dir_ann


def build_new(rel_paths: list[str]) -> dict[str, Any]:
    """Build a nested tree from paths relative to one root.

    *rel_paths* are e.g. ``estimators/deep/vision.py`` for root
    ``pu_toolbox``; the returned tree contains the root key.
    """
    tree: dict[str, Any] = {}
    for p in rel_paths:
        parts = p.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault(FILES_KEY, {})[parts[-1]] = ""
    return tree


def merge_tree(
    old: dict[str, Any],
    new: dict[str, Any],
    dir_ann: dict[str, str],
    prefix: str,
    path: str,
    level: int,
    out: list[str],
    missing: list[str],
) -> None:
    """Recursively merge the old tree order/annotations into the new one.

    Entries present in both keep their old relative order and annotations;
    new directories and files are appended alphabetically. Files on disk
    but absent from the old document are appended with PLACEHOLDER and
    recorded in *missing* (repo-relative). ``(planned)`` entries that do
    not exist on disk are kept verbatim.
    """
    new_dirs = {k for k in new if k != FILES_KEY}
    new_files = set(new.get(FILES_KEY, {}))
    seen_dirs: set[str] = set()
    for k, v in old.items():
        if k == FILES_KEY or k not in new_dirs:
            continue
        line = "  " * level + k + "/"
        ann = dir_ann.get(f"{path}/{k}" if path else k, "")
        if ann.startswith(prefix):
            ann = ann[len(prefix) :]
        if ann:
            pad = max(1, COMMENT_COL - level * 2 - len(k) - 1 - len(prefix))
            line += " " * pad + prefix + ann
        out.append(line)
        merge_tree(
            v, new[k], dir_ann, prefix, f"{path}/{k}" if path else k, level + 1, out, missing
        )
        seen_dirs.add(k)
    for k in sorted(new_dirs - seen_dirs):
        out.append("  " * level + k + "/")
        merge_tree(
            {}, new[k], dir_ann, prefix, f"{path}/{k}" if path else k, level + 1, out, missing
        )
    old_files: dict[str, str] = old.get(FILES_KEY, {})
    seen_files: set[str] = set()
    for name, ann in old_files.items():
        if name not in new_files:
            if "(planned)" in ann:
                line = "  " * level + name
                if ann:
                    line += " " + ann
                out.append(line)
            continue  # stale 条目由 generate() 报告
        if ann.startswith(prefix):
            ann = ann[len(prefix) :]
        line = "  " * level + name
        if ann:
            pad = max(1, COMMENT_COL - level * 2 - len(name) - len(prefix))
            line += " " * pad + prefix + ann
        out.append(line)
        seen_files.add(name)
    for name in sorted(new_files - seen_files):
        pad = max(1, COMMENT_COL - level * 2 - len(name) - len(prefix))
        line = "  " * level + name + " " * pad + prefix + PLACEHOLDER
        out.append(line)
        missing.append(f"{path}/{name}" if path else name)


def collect_entries(trees: dict[str, Any]) -> set[str]:
    """Flatten nested trees to repo-relative paths (e.g. ``pu_toolbox/core/base.py``)."""

    def walk(node: dict[str, Any], path: str, acc: set[str]) -> None:
        for k, v in node.items():
            if k == FILES_KEY:
                for name in v:
                    acc.add(f"{path}/{name}" if path else name)
            else:
                walk(v, f"{path}/{k}" if path else k, acc)

    entries: set[str] = set()
    for root, tree in trees.items():
        walk(tree, root, entries)
    return entries


def generate(text: str, disk_files: list[str]) -> tuple[str, list[str], list[str]]:
    """Rebuild the document tree blocks.

    Returns ``(new_text, missing, stale)``: *missing* are on-disk files
    absent from the document (need annotations), *stale* are documented
    files absent from disk without a ``(planned)`` mark.
    """
    lines = text.splitlines()
    blocks = find_blocks(lines)
    disk_set = {f for f in disk_files if f.endswith(".py") and f.split("/")[0] in GENERATABLE_ROOTS}
    out_lines: list[str] = []
    missing: list[str] = []
    stale: list[str] = []
    i = 0
    while i < len(lines):
        if blocks and blocks[0][0] == i:
            start, end, root = blocks.pop(0)
            if root is None:
                out_lines.extend(lines[i : end + 1])
            else:
                content = lines[start + 1 : end]
                old_tree, dir_ann = parse_tree(content, root)
                root_disk = sorted(f[len(root) + 1 :] for f in disk_set if f.startswith(root + "/"))
                new_tree = {root: build_new(root_disk)}
                prefix = "# " if root == "tests" else ""
                block_lines = [root + "/"]
                merge_tree(
                    old_tree.get(root, {}),
                    new_tree[root],
                    dir_ann,
                    prefix,
                    root,
                    1,
                    block_lines,
                    missing,
                )
                doc_entries = collect_entries({root: old_tree.get(root, {})})
                disk_root = {f for f in disk_set if f.startswith(root + "/")}
                stale.extend(sorted(doc_entries - disk_root))
                out_lines.append("```text")
                out_lines.extend(block_lines)
                out_lines.append("```")
            i = end + 1
        else:
            out_lines.append(lines[i])
            i += 1
    return "\n".join(out_lines) + "\n", missing, stale


def tracked_py_files() -> list[str]:
    """Git-tracked ``.py`` files relative to the project root.

    Falls back to a directory walk (excluding ``.venv``/``.git``/caches)
    when ``git`` is unavailable, e.g. in scratch-dir tests. Prefer
    ``git ls-files`` in the real repository: it excludes ignored files.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        files = [ln for ln in proc.stdout.splitlines() if ln.endswith(".py")]
        if files:
            return files
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    skip = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
    return sorted(
        str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for p in PROJECT_ROOT.rglob("*.py")
        if not any(part in skip for part in p.relative_to(PROJECT_ROOT).parts)
    )


def _display_rel(p: Path) -> str:
    """Repo-relative path for messages; absolute fallback when not under root.

    ``relative_to`` raises on paths outside PROJECT_ROOT (e.g. a
    monkeypatched STRUCTURE_MD in tests), so fall back to the full path.
    """
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate/verify the project_structure.md tree blocks."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check", action="store_true", help="verify the document matches disk (default)"
    )
    group.add_argument("--update", action="store_true", help="rewrite tree blocks in the document")
    args = parser.parse_args(argv)

    if not STRUCTURE_MD.exists():
        print(f"error: {STRUCTURE_MD} not found", file=sys.stderr)
        return 1
    text = STRUCTURE_MD.read_text(encoding="utf-8")
    disk = tracked_py_files()
    new_text, missing, stale = generate(text, disk)
    changed = new_text != text

    if args.update:
        if changed:
            STRUCTURE_MD.write_text(new_text, encoding="utf-8")
            print(f"updated {_display_rel(STRUCTURE_MD)}")
        if missing:
            print("files without annotation (add one after the file name):")
            for f in missing:
                print(f"  {f}")
        if stale:
            print("documented files missing from disk (removed by this update):")
            for f in stale:
                print(f"  {f}")
        return 0

    problems = 0
    if changed:
        print(
            f"error: {_display_rel(STRUCTURE_MD)} is out of sync "
            f"-- run `uv run python scripts/generate_structure.py --update`",
            file=sys.stderr,
        )
        problems += 1
    for f in missing:
        print(f"error: {f} exists on disk but is missing from the document", file=sys.stderr)
        problems += 1
    for f in stale:
        print(
            f"error: {f} is listed but does not exist on disk -- remove it or mark `(planned)`",
            file=sys.stderr,
        )
        problems += 1
    if not problems:
        print("structure document is up to date")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
