#!/usr/bin/env python3
"""Documentation-code consistency gate.

Rules:
1. **Path references** -- every ``path/file.py`` in docs must exist on disk.
2. **(planned) consistency** -- ``project_structure.md`` tree must match
   actual file existence.
3. **Architecture S8 mapping** -- ``architecture.md`` S8 table must agree
   with registry NATIVE methods.
4. **Index completeness** -- ``docs/README.md`` must list all doc files;
   ``scripts/`` must be mentioned in README/CLAUDE.md.

Usage::

    uv run python scripts/check_doc_links.py

Exit 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

# ====================================================================
# Configuration
# ====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Directories whose backtick-quoted paths we check in Rule 1.
VALID_PATH_ROOTS: tuple[str, ...] = (
    "pu_toolbox", "tests", "scripts", "examples", "docs", "external",
)

# Regex: backtick-wrapped paths like `pu_toolbox/core/base.py`
_PATH_ROOT_ALT = "|".join(VALID_PATH_ROOTS)
PATH_PATTERN = re.compile(rf"`((?:{_PATH_ROOT_ALT})/[^`]+\.py)`")

# Files in docs/ that are NOT expected to appear in docs/README.md.
DOC_INDEX_EXCLUDED: set[str] = {"README.md"}

# Docs subdirectories excluded from ALL checks.
_EXCLUDED_DOC_DIRS: set[str] = {
    "research", "superpowers", "figures", "project_management",
}

# Files in docs/project_management/ expected to be listed.
PM_FILES_EXPECTED: set[str] = {
    "decision_log.md", "process_checklist.md", "division.txt",
}


# ====================================================================
# Data types
# ====================================================================


class Issue(NamedTuple):
    rule: str       # e.g. "rule-1"
    file: str       # relative path to the doc file
    line: int | None
    message: str
    severity: str   # "error" or "warning"


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
    """Return all in-scope .md files, sorted by path."""
    files: list[Path] = []
    for name in ["README.md", "CLAUDE.md"]:
        p = PROJECT_ROOT / name
        if p.exists():
            files.append(p)
    for p in DOCS_DIR.rglob("*.md"):
        if any(p.is_relative_to(DOCS_DIR / d) for d in _EXCLUDED_DOC_DIRS):
            continue
        files.append(p)
    files.sort()
    return files


def _extract_backtick_paths(text: str) -> list[tuple[str, int]]:
    """Return (path, 1-indexed line_number) for every `root/.../file.py`."""
    results: list[tuple[str, int]] = []
    for match in PATH_PATTERN.finditer(text):
        path = match.group(1)
        if "/" not in path:
            continue
        line_no = text[: match.start()].count("\n") + 1
        results.append((path, line_no))
    return results


# ====================================================================
# Rule functions -- each returns list[Issue]
# ====================================================================


def check_path_references(md_files: list[Path]) -> list[Issue]:
    """Rule 1: every `path/file.py` in docs must exist on disk."""
    issues: list[Issue] = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        for ref_path, line_no in _extract_backtick_paths(text):
            if not (PROJECT_ROOT / ref_path).exists():
                issues.append(Issue(
                    "rule-1", _relative(md_file), line_no,
                    f"referenced file not found: `{ref_path}`", "error",
                ))
    return issues


def check_planned_consistency(structure_md: Path) -> list[Issue]:
    """Rule 2: project_structure.md (planned) tags must match filesystem.

    Parses ``text`` code blocks, tracks directory nesting via indentation,
    and checks every .py entry (except __init__.py) has ``(planned)`` iff
    the file does NOT exist on disk.
    """
    if not structure_md.exists():
        return [Issue("rule-2", _relative(structure_md), None,
                       "project_structure.md not found", "error")]

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
            issues.append(Issue(
                "rule-2", _relative(structure_md), line_no,
                f"`{rel_path}` exists on disk but marked `(planned)` "
                f"-- remove the annotation", "error",
            ))
        elif not exists and not has_planned:
            issues.append(Issue(
                "rule-2", _relative(structure_md), line_no,
                f"`{rel_path}` does not exist on disk but is NOT marked "
                f"`(planned)` -- add the annotation", "error",
            ))

    return issues


def check_architecture_mapping(arch_md: Path) -> list[Issue]:
    """Rule 3: architecture.md S8 (planned) tags vs registry NATIVE methods.

    Extracts NATIVE module paths from ``builtin_methods.py``, then checks
    that architecture.md S8 does NOT mark them as (planned).
    """
    if not arch_md.exists():
        return [Issue("rule-3", _relative(arch_md), None,
                       "architecture.md not found", "error")]

    native_paths = _get_native_module_paths()
    if not native_paths:
        return [Issue("rule-3", _relative(arch_md), None,
                       "could not extract NATIVE paths from builtin_methods.py",
                       "warning")]

    text = arch_md.read_text(encoding="utf-8")
    table_entries = _parse_arch_section8_table(text)
    issues: list[Issue] = []

    for native_path in native_paths:
        if native_path in table_entries and table_entries[native_path]:
            issues.append(Issue(
                "rule-3", _relative(arch_md), table_entries[native_path],
                f"`{native_path}` is NATIVE in registry but marked "
                f"`(planned)` in architecture.md S8", "error",
            ))
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
    section_text = (
        text[section_start:next_section] if next_section != -1
        else text[section_start:]
    )
    base_line = text[:section_start].count("\n") + 1

    entries: dict[str, int | None] = {}
    arch_path_pat = re.compile(r"`([\w/]+\.py)`")

    for i, line in enumerate(section_text.split("\n")):
        for m in arch_path_pat.finditer(line):
            path = m.group(1)
            has_planned = "(planned)" in line
            entries[path] = (base_line + i) if has_planned else None

    return entries


def check_index_completeness(
    docs_readme: Path,
    root_readme: Path,
    claude_md: Path | None,
) -> list[Issue]:
    """Rule 4: docs/README.md lists all doc files; scripts mentioned somewhere.

    Two sub-checks:
    a) Every .md file directly under docs/ (excluding subdirs) should
       appear in docs/README.md.
    b) Every .py script in scripts/ should be mentioned by basename
       in README.md or CLAUDE.md.
    """
    issues: list[Issue] = []

    # -- 4a: docs/README.md lists all top-level doc files --
    if not docs_readme.exists():
        issues.append(Issue("rule-4", "docs/README.md", None,
                            "docs/README.md not found", "error"))
    else:
        text = docs_readme.read_text(encoding="utf-8")

        for p in sorted(DOCS_DIR.iterdir()):
            if p.is_file() and p.suffix == ".md" and p.name not in DOC_INDEX_EXCLUDED:
                if p.name not in text:
                    issues.append(Issue(
                        "rule-4", _relative(docs_readme), None,
                        f"`{p.name}` exists under docs/ but is not listed "
                        f"in docs/README.md", "error",
                    ))

        # Check project_management files
        pm_dir = DOCS_DIR / "project_management"
        if pm_dir.exists():
            for p in sorted(pm_dir.iterdir()):
                if p.name in PM_FILES_EXPECTED and p.name not in text:
                    issues.append(Issue(
                        "rule-4", _relative(docs_readme), None,
                        f"`project_management/{p.name}` exists but is not "
                        f"listed in docs/README.md", "warning",
                    ))

    # -- 4b: scripts/ mentioned in README.md or CLAUDE.md --
    if SCRIPTS_DIR.exists():
        readme_text = (
            root_readme.read_text(encoding="utf-8") if root_readme.exists() else ""
        )
        claude_text = claude_md.read_text(encoding="utf-8") if claude_md else ""
        combined = readme_text + "\n" + claude_text

        for p in sorted(SCRIPTS_DIR.glob("*.py")):
            if p.stem != "__init__" and p.stem not in combined:
                issues.append(Issue(
                    "rule-4", "README.md / CLAUDE.md", None,
                    f"script `scripts/{p.name}` is not mentioned in "
                    f"README.md or CLAUDE.md", "warning",
                ))

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

    issues = check_planned_consistency(DOCS_DIR / "project_structure.md")
    all_issues.extend(issues)
    _print_rule_report("Rule 2: (planned) consistency", issues)

    issues = check_architecture_mapping(DOCS_DIR / "architecture.md")
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
