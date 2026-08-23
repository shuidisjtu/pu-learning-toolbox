#!/usr/bin/env python3
"""Documentation-code consistency gate.

Rules:
1. **Path references** -- every ``path/file.{py,md}`` in project Markdown must exist on disk.
2. **(planned) consistency** -- ``project_structure.md`` tree must match
   actual file existence.
3. **Architecture S8 mapping** -- ``architecture.md`` S8 table must agree
   with registry NATIVE methods.
4. **Index completeness** -- ``docs/README.md`` must list all doc files
   (whole docs tree; directory entries cover their subtree);
   ``scripts/`` must be mentioned in README/CLAUDE.md.
5. **Markdown links** -- every md link target must exist on disk
   (external URLs and ``#`` anchors are skipped).

Usage::

    uv run python scripts/check_doc_links.py

Exit 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

import generate_structure as _gen  # scripts/ dir is on sys.path[0]

# ====================================================================
# Configuration
# ====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Directories whose backtick-quoted paths we check in Rule 1.
VALID_PATH_ROOTS: tuple[str, ...] = (
    "pu_toolbox",
    "tests",
    "scripts",
    "examples",
    "docs",
    "external",
)

# Regex: backtick-wrapped paths like `pu_toolbox/core/base.py` or
# `docs/dev/project_structure.md`.
_PATH_ROOT_ALT = "|".join(VALID_PATH_ROOTS)
PATH_PATTERN = re.compile(rf"`((?:{_PATH_ROOT_ALT})/[^`]+\.(?:py|md))`")

# Regex: markdown links `[label](target)`; the target may be an external
# URL, an in-file anchor, or a relative path (see _extract_md_link_targets).
MD_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

# Files in docs/ that are NOT expected to appear in docs/README.md.
DOC_INDEX_EXCLUDED: set[str] = {"README.md"}

# Docs subdirectories excluded from ALL checks. research/ (method cards)
# is in scope: it is the densest citation source and must not be
# wholesale-exempted.
_EXCLUDED_DOC_DIRS: set[str] = {"superpowers", "figures"}


# ====================================================================
# Data types
# ====================================================================


class Issue(NamedTuple):
    rule: str  # e.g. "rule-1"
    file: str  # relative path to the doc file
    line: int | None
    message: str
    severity: str  # "error" or "warning"


# ====================================================================
# Helpers
# ====================================================================


def _relative(path: Path) -> str:
    """Return *path* relative to PROJECT_ROOT, using forward slashes."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _find_md_files() -> list[Path]:
    """Return root-level and docs-tree Markdown files, sorted by path."""
    files = list(PROJECT_ROOT.glob("*.md"))
    for p in DOCS_DIR.rglob("*.md"):
        if any(p.is_relative_to(DOCS_DIR / d) for d in _EXCLUDED_DOC_DIRS):
            continue
        files.append(p)
    files.sort()
    return files


def _extract_backtick_paths(text: str) -> list[tuple[str, int]]:
    """Return (path, 1-indexed line_number) for every `root/.../file.{py,md}`.

    Tokens containing glob characters (``{``, ``*``, ``?``) are list-style
    shorthand (e.g. ``pu_toolbox/estimators/deep/{infomax_pu,vision}.py``),
    not claims about a single file, so they are skipped.
    """
    results: list[tuple[str, int]] = []
    for match in PATH_PATTERN.finditer(text):
        path = match.group(1)
        if "/" not in path:
            continue
        if any(c in path for c in ("{", "}", "*", "?")):
            continue
        line_no = text[: match.start()].count("\n") + 1
        results.append((path, line_no))
    return results


def _normalize_md_target(raw: str) -> str | None:
    """Return the file part of a markdown link target, or None if it is
    not a file reference (empty, external URL, or pure ``#`` anchor).

    The fragment of a relative target (``doc.md#sec``) is stripped.
    """
    target = raw.strip().split("#", 1)[0].strip()
    if not target or target.startswith(("http://", "https://", "mailto:")):
        return None
    return target


def _extract_md_link_targets(text: str) -> list[tuple[str, int]]:
    """Return (target, 1-indexed line_number) for every markdown link.

    External URLs (``http://``, ``https://``, ``mailto:``) and in-file
    anchors (``#section``) are not file references and are skipped.
    """
    results: list[tuple[str, int]] = []
    for match in MD_LINK_PATTERN.finditer(text):
        target = _normalize_md_target(match.group(1))
        if target is None:
            continue
        line_no = text[: match.start()].count("\n") + 1
        results.append((target, line_no))
    return results


# ====================================================================
# Rule functions -- each returns list[Issue]
# ====================================================================


def check_path_references(md_files: list[Path]) -> list[Issue]:
    """Rule 1: every `path/file.{py,md}` in docs must exist on disk."""
    issues: list[Issue] = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        for ref_path, line_no in _extract_backtick_paths(text):
            if not (PROJECT_ROOT / ref_path).exists():
                issues.append(
                    Issue(
                        "rule-1",
                        _relative(md_file),
                        line_no,
                        f"referenced file not found: `{ref_path}`",
                        "error",
                    )
                )
    return issues


def check_md_links(md_files: list[Path]) -> list[Issue]:
    """Rule 5: every markdown link target (file or dir) must exist on disk.

    Targets are resolved relative to the source document. External URLs
    (http/https/mailto) and in-file anchors (#section) are skipped by
    ``_extract_md_link_targets``.
    """
    issues: list[Issue] = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        for target, line_no in _extract_md_link_targets(text):
            if not (md_file.parent / target).exists():
                issues.append(
                    Issue(
                        "rule-5",
                        _relative(md_file),
                        line_no,
                        f"markdown link target not found: `{target}`",
                        "error",
                    )
                )
    return issues


def check_planned_consistency(structure_md: Path) -> list[Issue]:
    """Rule 2: project_structure.md tree must match git-tracked .py files.

    Bidirectional check sharing the tree logic with generate_structure.py:
    every git-tracked ``.py`` under ``pu_toolbox/``/``tests/`` must appear
    in the document, and every documented entry must exist on disk or be
    marked ``(planned)``. Entries that exist on disk while marked
    ``(planned)`` are errors too; for tree blocks the generator does not
    manage (e.g. ``examples/``), the legacy existence check still applies.
    """
    if not structure_md.exists():
        return [
            Issue(
                "rule-2", _relative(structure_md), None, "project_structure.md not found", "error"
            )
        ]

    text = structure_md.read_text(encoding="utf-8")
    lines = text.split("\n")
    issues: list[Issue] = []
    in_block = False
    path_stack: list[tuple[int, str]] = []  # (indent, dir_component)

    for i, raw in enumerate(lines):
        line_no = i + 1
        stripped = raw.strip()

        if stripped.startswith("```text"):
            in_block = True
            path_stack.clear()
            continue
        if stripped == "```":
            in_block = False
            continue
        if not in_block or not stripped:
            continue

        indent = len(raw) - len(raw.lstrip())
        parts = stripped.split()
        name = parts[0]
        annotation = " ".join(parts[1:])

        is_dir = name.endswith("/")
        if not (is_dir or name.endswith(".py")):
            continue

        # Pop stack to find parent at a shallower indent
        while path_stack and path_stack[-1][0] >= indent:
            path_stack.pop()

        clean_name = name.rstrip("/")
        if is_dir:
            path_stack.append((indent, clean_name))
            continue

        if clean_name == "__init__.py":
            continue

        prefix = "/".join(c for _, c in path_stack)
        rel_path = f"{prefix}/{clean_name}" if prefix else clean_name
        exists = (PROJECT_ROOT / rel_path).exists()
        has_planned = "(planned)" in annotation

        if exists and has_planned:
            issues.append(
                Issue(
                    "rule-2",
                    _relative(structure_md),
                    line_no,
                    f"`{rel_path}` exists on disk but marked `(planned)` -- remove the annotation",
                    "error",
                )
            )
        elif not exists and not has_planned and not rel_path.startswith(("pu_toolbox/", "tests/")):
            # Generator-managed roots are covered by the bidirectional
            # check below; keep the legacy existence check for blocks
            # that generate_structure.py does not manage (examples/, ...).
            issues.append(
                Issue(
                    "rule-2",
                    _relative(structure_md),
                    line_no,
                    f"`{rel_path}` does not exist on disk but is NOT marked "
                    f"`(planned)` -- add the annotation",
                    "error",
                )
            )

    # Bidirectional check, sharing the tree logic with generate_structure.py:
    # every git-tracked .py under pu_toolbox/tests must appear in the
    # document, and every documented entry must exist on disk or be marked
    # (planned).
    tracked = [f for f in _gen.tracked_py_files() if f.startswith(("pu_toolbox/", "tests/"))]
    _new_text, missing, stale = _gen.generate(text, tracked)
    for rel in missing:
        issues.append(
            Issue(
                "rule-2",
                _relative(structure_md),
                None,
                f"`{rel}` exists on disk but is missing from project_structure.md",
                "error",
            )
        )
    for rel in stale:
        issues.append(
            Issue(
                "rule-2",
                _relative(structure_md),
                None,
                f"`{rel}` is listed in project_structure.md but does not exist "
                f"on disk -- remove it or mark `(planned)`",
                "error",
            )
        )
    return issues


def check_architecture_mapping(arch_md: Path) -> list[Issue]:
    """Rule 3: architecture.md S8 (planned) tags vs registry NATIVE methods.

    Extracts NATIVE module paths from ``builtin_methods.py``, then checks
    that architecture.md S8 does NOT mark them as (planned).
    """
    if not arch_md.exists():
        return [Issue("rule-3", _relative(arch_md), None, "architecture.md not found", "error")]

    native_paths = _get_native_module_paths()
    if not native_paths:
        return [
            Issue(
                "rule-3",
                _relative(arch_md),
                None,
                "could not extract NATIVE paths from builtin_methods.py",
                "warning",
            )
        ]

    text = arch_md.read_text(encoding="utf-8")
    table_entries = _parse_arch_section8_table(text)
    issues: list[Issue] = []

    for native_path in native_paths:
        if native_path in table_entries and table_entries[native_path]:
            issues.append(
                Issue(
                    "rule-3",
                    _relative(arch_md),
                    table_entries[native_path],
                    f"`{native_path}` is NATIVE in registry but marked "
                    f"`(planned)` in architecture.md S8",
                    "error",
                )
            )
    return issues


def _get_native_module_paths() -> set[str]:
    """Extract NATIVE module file paths from builtin_methods.py via regex.

    Looks for the ``_native_imports`` list and converts relative import
    paths (e.g. ``..estimators.classic.elkan_noto``) to file paths
    relative to ``pu_toolbox/`` (e.g. ``estimators/classic/elkan_noto.py``).
    """
    registry_file = PROJECT_ROOT / "pu_toolbox" / "registry" / "builtin_methods.py"
    if not registry_file.exists():
        return set()

    text = registry_file.read_text(encoding="utf-8")
    start = text.find("_native_imports")
    if start == -1:
        return set()

    # Match tuples like ("name", "..estimators.classic.elkan_noto", "Class")
    block = text[start:]
    pattern = re.compile(r'\(\s*"[^"]+"\s*,\s*"([^"]+)"\s*,\s*"[^"]+"\s*\)')

    paths: set[str] = set()
    for m in pattern.finditer(block):
        mod = m.group(1)  # e.g. "..estimators.classic.elkan_noto"
        # Strip leading dots, convert dots to slashes, append .py
        paths.add(mod.lstrip(".").replace(".", "/") + ".py")
    return paths


def _parse_arch_section8_table(text: str) -> dict[str, int | None]:
    """Parse architecture.md S8 table.

    Returns ``{file_path: line_number_if_planned_or_None}``.
    Paths are relative to ``pu_toolbox/`` to match ``_get_native_module_paths``.
    """
    section_start = text.find("## 8. 论文方法到模块的映射")
    if section_start == -1:
        return {}

    next_section = text.find("\n## 9.", section_start)
    section_text = text[section_start:next_section] if next_section != -1 else text[section_start:]
    base_line = text[:section_start].count("\n") + 1

    entries: dict[str, int | None] = {}
    arch_path_pat = re.compile(r"`([\w/]+\.py)`")

    for i, line in enumerate(section_text.split("\n")):
        for m in arch_path_pat.finditer(line):
            path = m.group(1)
            has_planned = "(planned)" in line
            entries[path] = (base_line + i) if has_planned else None

    return entries


def _index_link_targets(index_text: str) -> set[str]:
    """Link targets used by an index file, e.g. ``user/quickstart.md``."""
    targets: set[str] = set()
    for match in MD_LINK_PATTERN.finditer(index_text):
        target = _normalize_md_target(match.group(1))
        if target is not None:
            targets.add(target)
    return targets


def _is_indexed(rel: str, index_targets: set[str]) -> bool:
    """True if *rel* (e.g. ``user/quickstart.md``) is listed by the index.

    An exact entry (``research/method_cards/PUSB.md``) or an ancestor
    directory entry (``research/method_cards/`` covers every card below
    it) both count.
    """
    if rel in index_targets:
        return True
    parts = rel.split("/")
    return any("/".join(parts[:i]) + "/" in index_targets for i in range(1, len(parts)))


def check_index_completeness(
    docs_readme: Path,
    root_readme: Path,
    claude_md: Path | None,
) -> list[Issue]:
    """Rule 4: docs/README.md lists all doc files; scripts mentioned somewhere.

    Two sub-checks:
    a) Every .md file under docs/ (whole tree, excluding superpowers/
       figures and README.md) should be listed in docs/README.md, either
       directly or via a listed ancestor directory.
    b) Every .py script in scripts/ should be mentioned by basename
       in README.md or CLAUDE.md.
    """
    issues: list[Issue] = []

    # -- 4a: docs/README.md lists all in-scope doc files (whole tree) --
    if not docs_readme.exists():
        issues.append(Issue("rule-4", "docs/README.md", None, "docs/README.md not found", "error"))
    else:
        index_targets = _index_link_targets(docs_readme.read_text(encoding="utf-8"))
        docs_dir = docs_readme.parent

        for p in sorted(docs_dir.rglob("*.md")):
            if any(p.is_relative_to(docs_dir / d) for d in _EXCLUDED_DOC_DIRS):
                continue
            if p.name in DOC_INDEX_EXCLUDED:
                continue
            rel = p.relative_to(docs_dir).as_posix()
            if not _is_indexed(rel, index_targets):
                issues.append(
                    Issue(
                        "rule-4",
                        _relative(docs_readme),
                        None,
                        f"`{rel}` exists under docs/ but is not listed in docs/README.md",
                        "error",
                    )
                )

    # -- 4b: scripts/ mentioned in README.md or CLAUDE.md --
    if SCRIPTS_DIR.exists():
        readme_text = root_readme.read_text(encoding="utf-8") if root_readme.exists() else ""
        claude_text = claude_md.read_text(encoding="utf-8") if claude_md else ""
        combined = readme_text + "\n" + claude_text

        for p in sorted(SCRIPTS_DIR.glob("*.py")):
            if p.stem != "__init__" and p.stem not in combined:
                issues.append(
                    Issue(
                        "rule-4",
                        "README.md / CLAUDE.md",
                        None,
                        f"script `scripts/{p.name}` is not mentioned in README.md or CLAUDE.md",
                        "warning",
                    )
                )

    return issues


# ====================================================================
# Report & main
# ====================================================================


def _print_rule_report(title: str, issues: list[Issue]) -> None:
    """Print a grouped rule report."""
    print(f"\n-- {title} --")
    if not issues:
        print("  ok")
        return
    for issue in issues:
        loc = f"{issue.file}:{issue.line}" if issue.line else issue.file
        tag = "ERROR" if issue.severity == "error" else "WARN"
        print(f"  [{tag}] {loc} -- {issue.message}")


def main() -> int:
    """Run all checks and return exit code (0 = clean, 1 = issues found)."""
    sys.stdout.reconfigure(encoding="utf-8")

    md_files = _find_md_files()
    all_issues: list[Issue] = []

    print("=" * 62)
    print(" Documentation-Code Consistency Check")
    print("=" * 62)

    issues = check_path_references(md_files)
    all_issues.extend(issues)
    _print_rule_report("Rule 1: Path references", issues)

    issues = check_planned_consistency(DOCS_DIR / "dev" / "project_structure.md")
    all_issues.extend(issues)
    _print_rule_report("Rule 2: (planned) consistency", issues)

    issues = check_architecture_mapping(DOCS_DIR / "dev" / "architecture.md")
    all_issues.extend(issues)
    _print_rule_report("Rule 3: Architecture S8 mapping", issues)

    docs_readme = DOCS_DIR / "README.md"
    root_readme = PROJECT_ROOT / "README.md"
    claude_md = PROJECT_ROOT / "CLAUDE.md"
    if not claude_md.exists():
        claude_md = None
    issues = check_index_completeness(docs_readme, root_readme, claude_md)
    all_issues.extend(issues)
    _print_rule_report("Rule 4: Index completeness", issues)

    issues = check_md_links(md_files)
    all_issues.extend(issues)
    _print_rule_report("Rule 5: Markdown link targets", issues)

    # Final verdict
    print()
    errors = [i for i in all_issues if i.severity == "error"]
    warnings = [i for i in all_issues if i.severity == "warning"]
    if not errors:
        if warnings:
            print(f"All checks passed ({len(warnings)} warning(s)).")
        else:
            print("All checks passed.")
        return 0
    else:
        print(f"{len(errors)} error(s), {len(warnings)} warning(s) found.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
