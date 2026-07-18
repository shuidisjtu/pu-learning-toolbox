#!/usr/bin/env python3
"""Documentation-code consistency gate.

Rules (aligned with ``docs/project_structure.md`` and the spec at
``docs/superpowers/specs/2026-07-18-doc-check-script-design.md``):

TIER 1 (default):
1. **Path references** — every `` `path/file.py` `` in docs must exist on disk.
2. **``(planned)`` consistency** — ``project_structure.md`` tree must match
   actual file existence.
3. **Architecture §8 mapping** — ``architecture.md`` §8 table must agree with
   registry NATIVE methods.
4. **Index completeness** — ``docs/README.md`` must list all doc files;
   ``scripts/`` must be mentioned in README/CLAUDE.md.

TIER 2 (--strict):
5. **CLAUDE.md freshness** — markers, script mentions, existence.
6. **Stale numbers** — test counts and NATIVE counts in docs vs reality.

Usage::

    uv run python scripts/check_doc_links.py [--strict]

Exit 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import NamedTuple

# ═════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
METHOD_CARDS_DIR = DOCS_DIR / "research" / "method_cards"

# Directories whose backtick-quoted paths we check in Rule 1.
VALID_PATH_ROOTS: tuple[str, ...] = (
    "pu_toolbox", "tests", "scripts", "examples", "docs", "external",
)

# Regex: backtick-wrapped paths like `pu_toolbox/core/base.py`
_PATH_ROOT_ALTERNATION = "|".join(VALID_PATH_ROOTS)
PATH_PATTERN = re.compile(
    rf"`((?:{_PATH_ROOT_ALTERNATION})/[^`]+\.py)`"
)

# Registered pytest markers from pyproject.toml (must stay in sync).
REGISTERED_MARKERS: set[str] = {
    "unit", "math", "property", "contract", "slow", "paper",
}

# Files in docs/ that are NOT expected to appear in docs/README.md §5.
DOC_INDEX_EXCLUDED: set[str] = {
    "README.md",  # the index itself
}
# Subdirectories of docs/ excluded from index completeness check.
DOC_INDEX_SKIP_DIRS: set[str] = {
    "research",       # method_cards excluded per spec
    "superpowers",    # specs/plans are internal
    "project_management",  # listed as a separate group
    "figures",        # images, not docs
}

# Files in docs/project_management/ that are expected to be listed.
PM_FILES_EXPECTED: set[str] = {
    "decision_log.md", "process_checklist.md", "division.txt",
}


# ═════════════════════════════════════════════════════════════════════
# Data types
# ═════════════════════════════════════════════════════════════════════

class Issue(NamedTuple):
    rule: str       # e.g. "rule-1", "rule-2"
    file: str       # relative path to the doc file
    line: int | None  # line number, if known
    message: str    # human-readable description
    severity: str   # "error" or "warning"


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _relative(path: Path) -> str:
    """Return *path* relative to PROJECT_ROOT, using forward slashes."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _find_md_files(exclude_method_cards: bool = True) -> list[Path]:
    """Return all .md files in scope, sorted by path."""
    files: list[Path] = []

    # Project root markdown files
    for name in ["README.md", "CLAUDE.md"]:
        p = PROJECT_ROOT / name
        if p.exists():
            files.append(p)

    # docs/ tree
    for p in DOCS_DIR.rglob("*.md"):
        if exclude_method_cards and _is_under(p, METHOD_CARDS_DIR):
            continue
        files.append(p)

    files.sort()
    return files


def _is_under(path: Path, parent: Path) -> bool:
    """Return True if *path* is under *parent* directory."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _extract_backtick_paths(text: str) -> list[tuple[str, int]]:
    """Return (path, line_number) for every `root/.../file.py` in *text*.

    Line numbers are 1-indexed.
    """
    results: list[tuple[str, int]] = []
    for match in PATH_PATTERN.finditer(text):
        quoted_path = match.group(1)
        # Skip things that look like loss names, not paths
        if quoted_path.count("/") == 0:
            continue
        line_no = text[:match.start()].count("\n") + 1
        results.append((quoted_path, line_no))
    return results


# ═════════════════════════════════════════════════════════════════════
# Rule functions — each returns list[Issue]
# ═════════════════════════════════════════════════════════════════════

def check_path_references(md_files: list[Path]) -> list[Issue]:
    """Rule 1: every `path/file.py` in docs must exist on disk.

    Scans all in-scope .md files for backtick-quoted code paths,
    verifies each referenced file exists relative to PROJECT_ROOT.
    """
    issues: list[Issue] = []

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        for ref_path, line_no in _extract_backtick_paths(text):
            full_path = PROJECT_ROOT / ref_path
            if not full_path.exists():
                issues.append(Issue(
                    rule="rule-1",
                    file=_relative(md_file),
                    line=line_no,
                    message=f"referenced file not found: `{ref_path}`",
                    severity="error",
                ))

    return issues


def check_planned_consistency(structure_md: Path) -> list[Issue]:
    """Rule 2: project_structure.md (planned) tags must match filesystem.

    Parses the ASCII directory tree in project_structure.md and verifies
    that every .py file listed has a (planned) marker iff it does NOT exist
    on disk.  ``__init__.py`` files and directory entries are exempt.
    """
    issues: list[Issue] = []
    text = structure_md.read_text(encoding="utf-8")

    # Find code-block sections (the directory trees)
    blocks = _extract_code_blocks(text, language="text")
    if not blocks:
        issues.append(Issue(
            rule="rule-2",
            file=_relative(structure_md),
            line=None,
            message="no ```text code blocks found in project_structure.md",
            severity="warning",
        ))
        return issues

    # Lines in the doc that hold the tree blocks, with global line numbers.
    # We work with the full text so line numbers are accurate.
    for block_start, block_end in blocks:
        block_issues = _check_tree_block(
            text, block_start, block_end, structure_md,
        )
        issues.extend(block_issues)

    return issues


def _extract_code_blocks(
    text: str, language: str = "text",
) -> list[tuple[int, int]]:
    """Return (start_line, end_line) 1-indexed for each ```<language> block."""
    blocks: list[tuple[int, int]] = []
    pattern = re.compile(rf"^```{language}\s*$", re.MULTILINE)
    end_pattern = re.compile(r"^```\s*$", re.MULTILINE)

    for match in pattern.finditer(text):
        start = match.end()  # position after ```text\n
        # Find closing ```
        end_match = end_pattern.search(text, start)
        if end_match:
            end = end_match.start()
            start_line = text[:match.start()].count("\n") + 2  # first content line
            end_line = text[:end].count("\n") + 1
            blocks.append((start_line, end_line))

    return blocks


def _check_tree_block(
    text: str, start_line: int, end_line: int, source_file: Path,
) -> list[Issue]:
    """Parse one ASCII tree block and check (planned) consistency."""
    issues: list[Issue] = []
    lines = text.split("\n")

    # State: stack of (indent, dir_name_or_prefix)
    # We track the "prefix" that builds up the module path.
    path_stack: list[tuple[int, str]] = []  # (indent, component)

    for i in range(start_line - 1, end_line):
        raw = lines[i]
        line_no = i + 1

        # Determine indent and extract name
        indent, name, annotation = _parse_tree_line(raw)
        if name is None:
            continue

        # Pop stack to find parent
        while path_stack and path_stack[-1][0] >= indent:
            path_stack.pop()

        is_dir = name.endswith("/")
        clean_name = name.rstrip("/")
        has_planned = "(planned)" in annotation

        if is_dir:
            path_stack.append((indent, clean_name))
        elif name.endswith(".py"):
            # Build full relative path
            prefix = "/".join(c for _, c in path_stack)
            rel_path = f"{prefix}/{clean_name}" if prefix else clean_name

            # Exempt __init__.py
            if clean_name == "__init__.py":
                continue

            exists = (PROJECT_ROOT / rel_path).exists()

            if exists and has_planned:
                issues.append(Issue(
                    rule="rule-2",
                    file=_relative(source_file),
                    line=line_no,
                    message=(
                        f"`{rel_path}` exists on disk but is marked "
                        f"`(planned)` — remove the annotation"
                    ),
                    severity="error",
                ))
            elif not exists and not has_planned:
                issues.append(Issue(
                    rule="rule-2",
                    file=_relative(source_file),
                    line=line_no,
                    message=(
                        f"`{rel_path}` does not exist on disk but is "
                        f"NOT marked `(planned)` — add the annotation"
                    ),
                    severity="error",
                ))

    return issues


def _parse_tree_line(line: str) -> tuple[int, str | None, str]:
    """Parse one line of an ASCII directory tree.

    Returns (indent_level, entry_name, annotation) where *entry_name*
    is None for non-file/dir lines.
    """
    # Strip leading tree-drawing characters to get the actual content.
    # Common patterns: "  ├── file.py", "  └── dir/", "  │   subdir/"
    # Tree chars: ├ └ ─ │ │ (box drawing)
    stripped = line.lstrip()
    indent = (len(line) - len(stripped))

    # Remove tree-drawing prefix
    content = stripped
    for prefix in ("├── ", "└── ", "├─ ", "└─ "):
        if content.startswith(prefix):
            content = content[len(prefix):]
            break
    else:
        # Might be a continuation line like "  │   subfile.py"
        if content.startswith("│   "):
            content = content[4:]
        elif content.startswith("│  "):
            content = content[3:]
        elif content.startswith("  "):
            content = content[2:]

    content = content.strip()
    if not content:
        return indent, None, ""

    # Separate annotation: everything after the filename/dirname
    # e.g., "datasets/                  (planned)" → name="datasets/", annotation="(planned)"
    # e.g., "recpe.py                  (native)" → name="recpe.py", annotation="(native)"
    parts = content.split()
    name = parts[0]
    annotation = " ".join(parts[1:]) if len(parts) > 1 else ""

    # Only interested in .py files and dirs ending with /
    if name.endswith(".py") or name.endswith("/"):
        return indent, name, annotation

    return indent, None, ""


def check_architecture_mapping(arch_md: Path) -> list[Issue]:
    """Rule 3: architecture.md §8 (planned) tags vs registry NATIVE methods."""
    return []


def check_index_completeness(
    docs_readme: Path, root_readme: Path, claude_md: Path | None
) -> list[Issue]:
    """Rule 4: docs/README.md lists all docs, scripts mentioned in README/CLAUDE."""
    return []


def check_claude_freshness(
    claude_md: Path | None, scripts_dir: Path
) -> list[Issue]:
    """Rule 5: CLAUDE.md exists, mentions scripts, markers match pyproject.toml."""
    return []


def check_stale_numbers(md_files: list[Path]) -> list[Issue]:
    """Rule 6: numeric claims in docs (test counts, NATIVE counts) vs reality."""
    return []


# ═════════════════════════════════════════════════════════════════════
# Report & main
# ═════════════════════════════════════════════════════════════════════

def main(strict: bool = False) -> int:
    """Run all checks and return exit code (0 = clean, 1 = issues found)."""
    sys.stdout.reconfigure(encoding="utf-8")

    md_files = _find_md_files()
    all_issues: list[Issue] = []

    # TIER 1 (always)
    print("=" * 62)
    print(" Documentation-Code Consistency Check")
    print("=" * 62)

    issues = check_path_references(md_files)
    all_issues.extend(issues)
    _print_rule_report("Rule 1: Path references", issues)

    structure_md = DOCS_DIR / "project_structure.md"
    issues = check_planned_consistency(structure_md)
    all_issues.extend(issues)
    _print_rule_report("Rule 2: (planned) consistency", issues)

    arch_md = DOCS_DIR / "architecture.md"
    issues = check_architecture_mapping(arch_md)
    all_issues.extend(issues)
    _print_rule_report("Rule 3: Architecture §8 mapping", issues)

    docs_readme = DOCS_DIR / "README.md"
    root_readme = PROJECT_ROOT / "README.md"
    claude_md = PROJECT_ROOT / "CLAUDE.md"
    if not claude_md.exists():
        claude_md = None
    issues = check_index_completeness(docs_readme, root_readme, claude_md)
    all_issues.extend(issues)
    _print_rule_report("Rule 4: Index completeness", issues)

    # TIER 2 (--strict)
    if strict:
        issues = check_claude_freshness(claude_md, SCRIPTS_DIR)
        all_issues.extend(issues)
        _print_rule_report("Rule 5: CLAUDE.md freshness", issues)

        issues = check_stale_numbers(md_files)
        all_issues.extend(issues)
        _print_rule_report("Rule 6: Stale numbers", issues)

    # Final verdict
    print()
    errors = [i for i in all_issues if i.severity == "error"]
    warnings = [i for i in all_issues if i.severity == "warning"]
    if not errors:
        if warnings:
            print(f"✓ All checks passed ({len(warnings)} warning(s)).")
        else:
            print("✓ All checks passed.")
        return 0
    else:
        print(
            f"✗ {len(errors)} error(s), {len(warnings)} warning(s) found."
        )
        return 1


def _print_rule_report(title: str, issues: list[Issue]) -> None:
    """Print a grouped rule report with ✓ or per-issue ✗ lines."""
    print(f"\n─ {title} ─")
    if not issues:
        print("  ✓ ok")
        return
    for issue in issues:
        loc = f"{issue.file}:{issue.line}" if issue.line else issue.file
        tag = "ERROR" if issue.severity == "error" else "WARN"
        print(f"  ✗ [{tag}] {loc} — {issue.message}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Documentation-code consistency gate for PU Learning Toolbox",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable TIER 2 heuristic checks (Rules 5-6)",
    )
    args = parser.parse_args()
    sys.exit(main(strict=args.strict))
