#!/usr/bin/env python3
"""Skill-sync consistency gate (5th quality gate).

Checks both pu-workflow resource trees, local Markdown links, and
portable frontmatter. Text resources compare by line (CRLF-tolerant);
binary resources compare by byte. Portable frontmatter fields are:

    name | description | license | compatibility | metadata

``DEFAULT_PATHS`` declares the expected entrypoint pair. Each entrypoint's
directory is a complete installable skill, including its references.

Usage::

    uv run python scripts/check_skill_sync.py

Exit 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PORTABLE_FRONTMATTER_FIELDS = frozenset(
    {"name", "description", "license", "compatibility", "metadata"}
)

DEFAULT_PATHS = (
    PROJECT_ROOT / ".claude" / "skills" / "pu-workflow" / "SKILL.md",
    PROJECT_ROOT / ".agents" / "skills" / "pu-workflow" / "SKILL.md",
)

_TEXT_SUFFIXES = frozenset(
    {".md", ".txt", ".rst", ".py", ".yaml", ".yml", ".json", ".toml", ".sh", ".ps1"}
)
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")


def _frontmatter_keys(text: str) -> dict[str, str]:
    """Return top-level ``key: value`` pairs of the YAML frontmatter block."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    block = text[3:end] if end != -1 else text[3:]
    return dict(re.findall(r"^([a-z-]+):[^\S\n]*(.*)$", block, flags=re.M))


def _check_frontmatter(path: Path) -> list[str]:
    """Validate the portable entrypoint metadata."""
    issues: list[str] = []
    fm = _frontmatter_keys(path.read_text(encoding="utf-8"))
    unknown = sorted(set(fm) - PORTABLE_FRONTMATTER_FIELDS)
    if unknown:
        issues.append(
            f"non-portable frontmatter fields: {', '.join(unknown)} "
            f"(open standard allows: {', '.join(sorted(PORTABLE_FRONTMATTER_FIELDS))})"
        )
    if fm.get("name") != path.parent.name:
        issues.append(
            f"frontmatter name {fm.get('name')!r} must match "
            f"the directory name {path.parent.name!r}"
        )
    if not fm.get("description", "").strip():
        issues.append("frontmatter description is required and must be non-empty")

    return issues


def _check_local_links(path: Path, root: Path) -> list[str]:
    """Check that local Markdown links resolve inside the installed skill."""
    issues: list[str] = []
    for match in _MARKDOWN_LINK.finditer(path.read_text(encoding="utf-8")):
        target = match.group(1).strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        resolved = (path.parent / unquote(parsed.path)).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
            issues.append(f"invalid local link in {path}: {target}")
    return issues


def check_sync(a: Path, b: Path) -> list[str]:
    """Return resource, reference, and metadata problems for the skill pair."""
    issues: list[str] = []
    for path in (a, b):
        if not path.is_file():
            issues.append(f"missing SKILL.md: {path}")
    if issues:
        return issues

    trees = [
        {path.relative_to(entry.parent): path for path in entry.parent.rglob("*") if path.is_file()}
        for entry in (a, b)
    ]
    for relative in sorted(trees[0].keys() | trees[1].keys()):
        first, second = (tree.get(relative) for tree in trees)
        if first is None or second is None:
            root = a.parent if first is None else b.parent
            issues.append(f"missing resource: {relative.as_posix()} in {root}")
            continue
        if relative.suffix.lower() in _TEXT_SUFFIXES:
            same = (
                first.read_text(encoding="utf-8").splitlines()
                == second.read_text(encoding="utf-8").splitlines()
            )
        else:
            same = first.read_bytes() == second.read_bytes()
        if not same:
            issues.append(
                f"content mismatch: {relative.as_posix()} differs between {a.parent} and {b.parent}"
            )

    for entry, tree in zip((a, b), trees, strict=True):
        issues.extend(_check_frontmatter(entry))
        for path in tree.values():
            if path.suffix.lower() == ".md":
                issues.extend(_check_local_links(path, entry.parent))
    return issues


def main(argv: list[str] | None = None) -> int:
    """Run the check on the default pair; return 0 when clean."""
    sys.stdout.reconfigure(encoding="utf-8")
    a, b = DEFAULT_PATHS
    issues = check_sync(a, b)
    if not issues:
        print(
            "Skill sync check passed: resource trees identical, "
            "local links valid, frontmatter portable."
        )
        return 0
    print(f"{len(issues)} skill sync issue(s):")
    for issue in issues:
        print(f"  - {issue}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
