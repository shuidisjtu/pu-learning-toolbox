#!/usr/bin/env python3
"""Skill-sync consistency gate (5th quality gate).

Checks that the two pu-workflow SKILL.md copies (Claude Code / Cursor
vs. other tools) are byte-consistent (line-level, CRLF-tolerant) and
that the frontmatter uses only portable open-standard fields:

    name | description | license | compatibility | metadata

Claude Code extension fields (``argument-hint``, ``user-invocable``,
``disable-model-invocation``, ``hooks``, ...) are silently ignored by
other tools, so adding them would quietly break portability.

``DEFAULT_PATHS`` is a hand-maintained pair, because the check needs a
pair to compare and an enumeration rule would silently lose exactly the
failure it exists to catch.  Pair the two trees by the skill names they
share, then delete one copy of the pair: the name now appears on one side
only, so the intersection is empty, the rule has no key left to attach a
failure to, and it reports no drift — while ``check_sync`` reports the
missing copy.  The alternative (treat every name on either side as
expected) needs a prior record of which names *should* be paired, i.e.
the hand-maintained list again.  Adding a skill that is mirrored means
adding its pair here by hand.

The two trees are not symmetric in this working tree either:
``.claude/skills/`` also holds ``dev-workflow``, a skill that is untracked
and gitignored (``.gitignore:55``), so a CI checkout contains one skill
per side.  Only the tracked pair is compared here.

Usage::

    uv run python scripts/check_skill_sync.py

Exit 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PORTABLE_FRONTMATTER_FIELDS = frozenset(
    {"name", "description", "license", "compatibility", "metadata"}
)

DEFAULT_PATHS = (
    PROJECT_ROOT / ".claude" / "skills" / "pu-workflow" / "SKILL.md",
    PROJECT_ROOT / ".agents" / "skills" / "pu-workflow" / "SKILL.md",
)


def _frontmatter_keys(text: str) -> dict[str, str]:
    """Return top-level ``key: value`` pairs of the YAML frontmatter block."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    block = text[3:end] if end != -1 else text[3:]
    return dict(re.findall(r"^([a-z-]+):[^\S\n]*(.*)$", block, flags=re.M))


def check_sync(a: Path, b: Path) -> list[str]:
    """Return problem messages; empty list means the pair is in sync."""
    issues: list[str] = []
    for path in (a, b):
        if not path.exists():
            issues.append(f"missing SKILL.md: {path}")
    if issues:
        return issues

    if a.read_text(encoding="utf-8").splitlines() != b.read_text(encoding="utf-8").splitlines():
        issues.append(
            "content mismatch: .claude/skills/pu-workflow/SKILL.md and "
            ".agents/skills/pu-workflow/SKILL.md differ (line-level)"
        )

    fm = _frontmatter_keys(a.read_text(encoding="utf-8"))
    unknown = sorted(set(fm) - PORTABLE_FRONTMATTER_FIELDS)
    if unknown:
        issues.append(
            f"non-portable frontmatter fields: {', '.join(unknown)} "
            f"(open standard allows: {', '.join(sorted(PORTABLE_FRONTMATTER_FIELDS))})"
        )
    if fm.get("name") != a.parent.name:
        issues.append(
            f"frontmatter name {fm.get('name')!r} must match the directory name {a.parent.name!r}"
        )
    if not fm.get("description", "").strip():
        issues.append("frontmatter description is required and must be non-empty")

    return issues


def main(argv: list[str] | None = None) -> int:
    """Run the check on the default pair; return 0 when clean."""
    sys.stdout.reconfigure(encoding="utf-8")
    a, b = DEFAULT_PATHS
    issues = check_sync(a, b)
    if not issues:
        print("Skill sync check passed: SKILL.md copies identical, frontmatter portable.")
        return 0
    print(f"{len(issues)} skill sync issue(s):")
    for issue in issues:
        print(f"  - {issue}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
