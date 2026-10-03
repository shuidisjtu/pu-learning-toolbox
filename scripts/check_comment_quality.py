#!/usr/bin/env python3
"""Check source comments for objective hygiene rules.

The semantic distinction between a useful intent comment and a repetition of
code requires human review.  This gate therefore checks only rules that can be
reliably enforced without guessing at intent:

* unfinished markers use ``TBD`` rather than legacy ``TODO``/``FIXME``/etc.;
* ``TBD`` must carry traceable context (``TBD:`` or ``TBD(#123)``);
* non-tooling trailing comments are reported as advisories for gradual cleanup.

Usage::

    uv run python scripts/check_comment_quality.py
    uv run python scripts/check_comment_quality.py pu_toolbox/core
    uv run python scripts/check_comment_quality.py --strict-inline pu_toolbox/core

The default scope is production package code.  ``--strict-inline`` is a
migration aid for a selected path; it turns trailing-comment advisories into
errors instead of making the whole repository pass only after a bulk rewrite.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (PROJECT_ROOT / "pu_toolbox",)
TOOLING_PREFIXES = (
    "fmt:",
    "isort:",
    "noqa",
    "pragma:",
    "ruff:",
    "type:",
)
LEGACY_MARKER_RE = re.compile(r"(?<![\w.])(?:TODO|FIXME|XXX|HACK)(?=\b|:)", re.IGNORECASE)
# Case-sensitive on purpose: the documented marker is uppercase ``TBD``, so
# lowercase prose such as "tbd" is not a marker claim.
BARE_TBD_RE = re.compile(r"(?<![\w.])TBD\b(?!\s*[:(])")

FindingKind = Literal["legacy-marker", "bare-tbd", "trailing-comment", "scan-error"]


@dataclass(frozen=True)
class CommentFinding:
    """One comment-quality finding with a stable, printable location."""

    path: Path
    line: int
    kind: FindingKind
    severity: str
    message: str


@dataclass(frozen=True)
class CommentReport:
    """Summary and findings from scanning a set of Python files."""

    files_scanned: int
    comments_scanned: int
    trailing_comments: int
    findings: tuple[CommentFinding, ...]

    @property
    def errors(self) -> tuple[CommentFinding, ...]:
        """Return findings that should make the gate fail."""
        return tuple(finding for finding in self.findings if finding.severity == "error")


def _python_files(paths: Iterable[Path]) -> list[Path]:
    """Expand files/directories into a deterministic list of Python files."""
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(
                candidate
                for candidate in path.rglob("*.py")
                if "__pycache__" not in candidate.parts
            )
        elif path.suffix == ".py":
            files.add(path)
    return sorted(files)


def _is_tooling_comment(body: str) -> bool:
    """Return whether *body* is formatter, linter, typing, or coverage metadata."""
    normalized = body.strip().lower()
    return normalized.startswith(TOOLING_PREFIXES)


def _relative(path: Path, root: Path) -> str:
    """Format a path relative to *root* when possible."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def scan_files(paths: Iterable[Path], *, strict_inline: bool = False) -> CommentReport:
    """Scan *paths* and return comment findings without importing project code."""
    findings: list[CommentFinding] = []
    files = _python_files(paths)
    comment_count = 0
    trailing_count = 0

    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            for token in tokens:
                if token.type != tokenize.COMMENT:
                    continue
                comment_count += 1
                body = token.string[1:].strip()
                if LEGACY_MARKER_RE.search(body):
                    findings.append(
                        CommentFinding(
                            path=path,
                            line=token.start[0],
                            kind="legacy-marker",
                            severity="error",
                            message="use TBD for unfinished work instead of TODO/FIXME/XXX/HACK",
                        )
                    )

                if BARE_TBD_RE.search(body):
                    findings.append(
                        CommentFinding(
                            path=path,
                            line=token.start[0],
                            kind="bare-tbd",
                            severity="error",
                            message=(
                                "TBD must carry traceable context, e.g. "
                                "'TBD: ...' or 'TBD(#123) ...'"
                            ),
                        )
                    )

                prefix = lines[token.start[0] - 1][: token.start[1]].strip()
                if prefix and not _is_tooling_comment(body):
                    trailing_count += 1
                    findings.append(
                        CommentFinding(
                            path=path,
                            line=token.start[0],
                            kind="trailing-comment",
                            severity="error" if strict_inline else "warning",
                            message=(
                                "trailing comments are advisory; prefer a preceding "
                                "intent comment or clearer code"
                            ),
                        )
                    )
        except (OSError, SyntaxError, tokenize.TokenError) as exc:
            findings.append(
                CommentFinding(
                    path=path,
                    line=1,
                    kind="scan-error",
                    severity="error",
                    message=f"could not scan Python comments: {exc}",
                )
            )

    return CommentReport(
        files_scanned=len(files),
        comments_scanned=comment_count,
        trailing_comments=trailing_count,
        findings=tuple(
            sorted(
                findings,
                key=lambda finding: (finding.path.as_posix(), finding.line, finding.kind),
            )
        ),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Python files or directories to scan (default: pu_toolbox/).",
    )
    parser.add_argument(
        "--strict-inline",
        action="store_true",
        help="treat non-tooling trailing comments as errors for the selected paths",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comment-quality gate and return a process status."""
    args = _parse_args(argv)
    paths = [
        path if path.is_absolute() else PROJECT_ROOT / path
        for path in (args.paths or DEFAULT_ROOTS)
    ]
    report = scan_files(paths, strict_inline=args.strict_inline)

    for finding in report.findings:
        if finding.severity == "error":
            print(
                f"{_relative(finding.path, PROJECT_ROOT)}:{finding.line}: "
                f"{finding.kind}: {finding.message}"
            )

    advisory_count = sum(finding.severity == "warning" for finding in report.findings)
    if report.errors:
        print(
            f"Comment quality gate failed: {len(report.errors)} error(s) in "
            f"{report.files_scanned} file(s)."
        )
        return 1

    print(
        f"Comment quality gate passed: scanned {report.files_scanned} file(s) and "
        f"{report.comments_scanned} comment(s); {advisory_count} trailing-comment advisory(ies)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
