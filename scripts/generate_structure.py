#!/usr/bin/env python3
"""Regenerate the tree blocks of docs/dev/project_structure.md.

The structure source is the tracked and non-ignored new files whose suffix
is in scope for their root (``GENERATABLE_SUFFIXES``), read with
``git ls-files --cached --others --exclude-standard``, excluding ignored
caches. Subtrees registered as a group (``GROUPED_SUBTREES``) are named by
index rather than enumerated, so their unlisted files are never reported
as missing while their listed entries are still verified. Hand-written
annotations and directory order are preserved from the current document;
files new on disk appear with a ``<<< 新文件,补注释`` placeholder so the
missing annotation stays visible.

The generator compares *file names* only. It never derives annotation text
from disk -- see the scope note in project_structure.md §5 for what that
leaves uncovered.

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

# Single source for what a block enumerates: a root is generatable when it
# is a key here, and its block covers exactly the listed suffixes.  ``docs``
# carries three because its inventory is prose (.md), figures (.png) and the
# survey data manifests (.json); suffixes outside a root's tuple are out of
# scope for that block -- neither required nor checked.
GENERATABLE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "pu_toolbox": (".py",),
    "tests": (".py",),
    "scripts": (".py",),
    "docs": (".md", ".png", ".json"),
}

# Derived, never restated: the roots ARE the mapping's keys, so the two
# cannot drift.  check_doc_links derives its prefix tuple from this name.
GENERATABLE_ROOTS: tuple[str, ...] = tuple(GENERATABLE_SUFFIXES)

# Subtrees registered as a group: the block names the subtree's own index
# instead of enumerating it, so a file below one of these that the document
# does not list is deliberate and never counts as missing.  A *listed* entry
# below one is asserted like any other -- gone from disk, it is stale.
# Repo-relative, no trailing slash.
GROUPED_SUBTREES: tuple[str, ...] = ("docs/adr", "docs/research/pu_survey")

COMMENT_COL = 42
PLACEHOLDER = "<<< 新文件,补注释"
FILES_KEY = "__files__"


def in_scope(rel_path: str) -> bool:
    """True when *rel_path* is a file some tree block enumerates.

    The top-level component must be a generatable root and the suffix must
    be one of that root's suffixes.  This is the single place for what used
    to be three separate ``.py`` hardcodes: the enumeration layer, the
    ``disk_set`` filter and ``parse_tree``.
    """
    root, sep, _ = rel_path.partition("/")
    if not sep or root not in GENERATABLE_SUFFIXES:
        return False
    return rel_path.endswith(GENERATABLE_SUFFIXES[root])


def is_grouped(rel_path: str) -> bool:
    """True when *rel_path* is inside a grouped subtree (see GROUPED_SUBTREES)."""
    return any(rel_path == sub or rel_path.startswith(sub + "/") for sub in GROUPED_SUBTREES)


def scope_suffixes() -> tuple[str, ...]:
    """Union of every root's suffixes, order-preserved and de-duplicated."""
    out: list[str] = []
    for suffixes in GENERATABLE_SUFFIXES.values():
        out.extend(s for s in suffixes if s not in out)
    return tuple(out)


def find_blocks(lines: list[str]) -> list[tuple[int, int, str | None]]:
    """Locate ```text fence blocks.

    Returns ``(start, end, rootname)`` triples where *start* is the fence
    opening line, *end* the fence closing line, and *rootname* the first
    entry of the block when it is one of GENERATABLE_ROOTS (``None`` for
    any other block, e.g. the root-directory listing or the ``examples/``
    block, which stay hand-maintained).
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
    An indent-0 bare file entry (a block-leading line that is not the
    root directory) is ignored: it is never reported as missing or stale.
    A name counts as a file entry when it ends with one of *root*'s
    suffixes (``GENERATABLE_SUFFIXES``), so the ``docs`` block's ``.md``,
    ``.png`` and ``.json`` entries are parsed, not dropped.
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
        elif name.endswith(GENERATABLE_SUFFIXES.get(root, ())):
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
    recorded in *missing* (repo-relative), unless they sit inside a grouped
    subtree (``is_grouped``), where the block registers the subtree rather
    than its files: those are neither emitted nor counted. ``(planned)``
    entries that do not exist on disk are kept verbatim. Directory-level
    ``(planned)`` markers are not preserved: only file entries are kept or
    excluded per ``"(planned)" in ann`` -- real documents do not use this
    form.
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
        if is_grouped(f"{path}/{k}" if path else k):
            continue  # 分组子树的目录不新增:其条目由该子树的索引负责
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
        rel = f"{path}/{name}" if path else name
        if is_grouped(rel):
            continue  # 分组子树下未列出的文件不计入 missing,也不产生占位符
        pad = max(1, COMMENT_COL - level * 2 - len(name) - len(prefix))
        line = "  " * level + name + " " * pad + prefix + PLACEHOLDER
        out.append(line)
        missing.append(rel)


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


def _planned_entries(tree: dict[str, Any], path: str) -> set[str]:
    """Documented file paths whose annotation contains the ``(planned)`` marker."""

    def walk(node: dict[str, Any], cur: str, acc: set[str]) -> None:
        for k, v in node.items():
            if k == FILES_KEY:
                for name, ann in v.items():
                    if "(planned)" in ann:
                        acc.add(f"{cur}/{name}" if cur else name)
            else:
                walk(v, f"{cur}/{k}" if cur else k, acc)

    entries: set[str] = set()
    walk(tree, path, entries)
    return entries


def generate(text: str, disk_files: list[str]) -> tuple[str, list[str], list[str]]:
    """Rebuild the document tree blocks.

    Returns ``(new_text, missing, stale)``: *missing* are on-disk files
    absent from the document (need annotations), *stale* are documented
    files absent from disk without a ``(planned)`` mark.
    """
    lines = text.splitlines()
    blocks = find_blocks(lines)
    disk_set = {f for f in disk_files if in_scope(f)}
    out_lines: list[str] = []
    missing: list[str] = []
    stale: list[str] = []
    # A root whose whole tree block is missing must still fail: otherwise
    # deleting an entire block would pass both gates with empty lists.
    # Grouped subtrees are exempt here too -- the block never lists them
    # file by file, so their absence from the document is not a gap.
    found_roots = {r for _, _, r in blocks if r is not None}
    for root in GENERATABLE_ROOTS:
        if root not in found_roots:
            missing.extend(
                sorted(f for f in disk_set if f.startswith(root + "/") and not is_grouped(f))
            )
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
                planned = _planned_entries(old_tree.get(root, {}), root)
                stale.extend(sorted((doc_entries - disk_root) - planned))
                out_lines.append("```text")
                out_lines.extend(block_lines)
                out_lines.append("```")
            i = end + 1
        else:
            out_lines.append(lines[i])
            i += 1
    return "\n".join(out_lines) + "\n", missing, stale


def tracked_files() -> list[str]:
    """Tracked and non-ignored new files of any in-scope suffix, repo-wide.

    The suffix union comes from ``GENERATABLE_SUFFIXES``, so adding a root
    or a suffix widens this enumeration too; callers scope the result to
    the roots they manage.  Falls back to a directory walk (excluding
    ``.venv``/``.git``/caches) when ``git`` is unavailable, e.g. in
    scratch-dir tests. Prefer ``git ls-files`` in the real repository: it
    excludes ignored files.
    """
    suffixes = scope_suffixes()
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        files = [ln for ln in proc.stdout.splitlines() if ln.endswith(suffixes)]
        if files:
            return sorted(set(files))
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    skip = {
        ".venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        "htmlcov",
    }
    return sorted(
        str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for p in PROJECT_ROOT.rglob("*")
        if p.is_file()
        and p.suffix in suffixes
        and not any(part in skip for part in p.relative_to(PROJECT_ROOT).parts)
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
    disk = tracked_files()
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
