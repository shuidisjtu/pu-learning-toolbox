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

The default scope is production package code, scanned with a partition
policy declared in this module: files inside ``STRICT_INLINE_PARTITIONS``
see non-tooling trailing comments as errors, files inside
``PENDING_INLINE_PARTITIONS`` keep them as advisories until their backlog
is cleared.  The pass message reports the pending partitions, so the
remaining work stays visible without pinning a shrinking count into docs.

``--strict-inline`` is a manual migration switch with different semantics:
it turns *every* scanned path strict, so one directory can be cleared
without rewriting the whole repository first.  Passing explicit paths
without the flag scans them with advisories only, as before.

Known boundaries (deliberate, not defects):

* ``tracked_top_level_entries`` reads the git index, so a brand-new
  top-level directory that has not been ``git add``-ed yet is not required
  to be classified; staging it makes the coverage assertion demand a
  partition.
* Passing explicit paths turns the partition policy off for that run: those
  paths are scanned with advisories only, or all-strict under
  ``--strict-inline``.  The strict/pending split therefore only binds the
  default run.
* Inside a strict partition the trailing-comment exemption looks only at
  whether the comment body starts with a tooling directive.  A genuine
  directive followed by prose (``type: ...``) is indistinguishable from the
  real thing by keyword matching.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import tokenize
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (PROJECT_ROOT / "pu_toolbox",)
TRACKED_PACKAGE_ROOT = "pu_toolbox"
# Inline-trailing-comment partitions, relative to PROJECT_ROOT, in sorted
# order.  Directory partitions match by path prefix, file partitions by
# exact relative path.  The two sets must partition every tracked
# top-level entry of ``pu_toolbox/`` -- see ``partition_problems``.
STRICT_INLINE_PARTITIONS: tuple[str, ...] = (
    "pu_toolbox/__init__.py",
    "pu_toolbox/cli",
    "pu_toolbox/diagnostics",
    "pu_toolbox/metrics",
    "pu_toolbox/model_selection",
    "pu_toolbox/preprocessing",
    "pu_toolbox/progress.py",
    "pu_toolbox/run_config.py",
    "pu_toolbox/ui",
)
# ``core`` and ``losses`` stay pending: their trailing comments are the
# ones a prior ruling kept on purpose, so a whole-directory strict flip
# would need that ruling revisited first.
PENDING_INLINE_PARTITIONS: tuple[str, ...] = (
    "pu_toolbox/advisor",
    "pu_toolbox/core",
    "pu_toolbox/estimators",
    "pu_toolbox/experiment",
    "pu_toolbox/losses",
    "pu_toolbox/prior",
    "pu_toolbox/registry",
    "pu_toolbox/utils",
    "pu_toolbox/workflows",
)
TOOLING_PREFIXES = (
    "fmt",
    "isort",
    "noqa",
    "pragma",
    "ruff",
    "type",
)
# A tooling keyword is exempt only as a whole directive: the keyword must be
# followed by end of comment, ``:``, or whitespace.  Prose that merely starts
# with a keyword -- ``noqaXYZ`` instead of a real directive -- is not exempt.
TOOLING_DIRECTIVE_RE = re.compile(
    r"^(?:" + "|".join(re.escape(prefix) for prefix in TOOLING_PREFIXES) + r")(?=$|:|\s)"
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
    """Return whether *body* is formatter, linter, typing, or coverage metadata.

    The keyword must form a whole directive, so ``noqaXYZ`` prose is not
    exempt.  A genuine directive followed by prose (``type: ...``) cannot
    be told apart from a real one by keyword matching alone; see the module
    docstring's known-boundary note.
    """
    return TOOLING_DIRECTIVE_RE.match(body.strip().lower()) is not None


def _relative(path: Path, root: Path) -> str:
    """Format a path relative to *root* when possible."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def is_strict_path(
    path: Path,
    strict_partitions: Sequence[str] = STRICT_INLINE_PARTITIONS,
    *,
    root: Path = PROJECT_ROOT,
) -> bool:
    """Return whether *path* lies inside a strict inline partition.

    A partition ending in ``.py`` is a single file and matches by exact
    relative path; any other partition is a directory and matches by path
    prefix, so ``pu_toolbox/cli`` covers ``pu_toolbox/cli/run.py``.
    """
    relative = _relative(path, root)
    for partition in strict_partitions:
        # Known limit: a directory literally named ``x.py`` would be read
        # as a single-file partition.  No declared partition looks like
        # that today, so the suffix test stands without extra code.
        if partition.endswith(".py"):
            if relative == partition:
                return True
        elif relative == partition or relative.startswith(f"{partition}/"):
            return True
    return False


def partition_problems(
    strict: Iterable[str],
    pending: Iterable[str],
    tracked: Iterable[str],
) -> list[str]:
    """Return how *strict* and *pending* fail to cover *tracked* exactly.

    The declarations must classify every tracked top-level entry of
    ``pu_toolbox/`` exactly once: their union equals *tracked* and the two
    sets are disjoint.  Each returned message names the offending entries;
    an empty list means the declarations are complete and mutually
    exclusive.
    """
    strict_set = set(strict)
    pending_set = set(pending)
    tracked_set = set(tracked)
    problems: list[str] = []
    overlap = sorted(strict_set & pending_set)
    if overlap:
        problems.append(f"declared in both strict and pending: {', '.join(overlap)}")
    unclassified = sorted(tracked_set - strict_set - pending_set)
    if unclassified:
        problems.append(f"tracked entries without a partition: {', '.join(unclassified)}")
    stale = sorted((strict_set | pending_set) - tracked_set)
    if stale:
        problems.append(f"partitions without a tracked entry: {', '.join(stale)}")
    return problems


def tracked_top_level_entries() -> tuple[list[str], bool]:
    """Return the sorted tracked top-level entries of ``pu_toolbox/``.

    ``git ls-files`` is preferred so untracked scratch files do not demand
    a partition classification.  The second element is True when git was
    unavailable and a directory walk was used instead, so the caller can
    say so in its output.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", TRACKED_PACKAGE_ROOT],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        entries = set()
        for line in proc.stdout.splitlines():
            parts = line.split("/")
            if len(parts) >= 2 and parts[0] == TRACKED_PACKAGE_ROOT:
                entries.add(f"{TRACKED_PACKAGE_ROOT}/{parts[1]}")
        if entries:
            return sorted(entries), False
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    root = PROJECT_ROOT / TRACKED_PACKAGE_ROOT
    entries = set()
    if root.is_dir():
        for candidate in root.rglob("*"):
            if "__pycache__" in candidate.parts:
                continue
            entries.add(f"{TRACKED_PACKAGE_ROOT}/{candidate.relative_to(root).parts[0]}")
    return sorted(entries), True


def partition_advisory_counts(
    findings: Iterable[CommentFinding],
    partitions: Sequence[str] = PENDING_INLINE_PARTITIONS,
    *,
    root: Path = PROJECT_ROOT,
) -> list[tuple[str, int]]:
    """Count trailing-comment advisories per partition, in *partitions* order."""
    counts = dict.fromkeys(partitions, 0)
    for finding in findings:
        if finding.severity != "warning":
            continue
        relative = _relative(finding.path, root)
        for partition in partitions:
            if relative == partition or relative.startswith(f"{partition}/"):
                counts[partition] += 1
                break
    return [(partition, counts[partition]) for partition in partitions]


def scan_files(
    paths: Iterable[Path],
    *,
    strict_inline: bool = False,
    strict_partitions: Sequence[str] = (),
    partitions_root: Path = PROJECT_ROOT,
) -> CommentReport:
    """Scan *paths* and return comment findings without importing project code.

    ``strict_inline`` promotes every non-tooling trailing comment to an
    error.  Otherwise, when *strict_partitions* is non-empty, only files
    inside those partitions (resolved against *partitions_root*) are
    promoted; everything else keeps advisories.  Legacy markers, bare
    ``TBD``, and scan errors are always errors.
    """
    findings: list[CommentFinding] = []
    files = _python_files(paths)
    comment_count = 0

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
                    strict_here = strict_inline or is_strict_path(
                        path, strict_partitions, root=partitions_root
                    )
                    findings.append(
                        CommentFinding(
                            path=path,
                            line=token.start[0],
                            kind="trailing-comment",
                            severity="error" if strict_here else "warning",
                            message=(
                                "trailing comments are advisory; prefer a preceding "
                                "intent comment or clearer code"
                            ),
                        )
                    )
        except (OSError, SyntaxError, UnicodeDecodeError, tokenize.TokenError) as exc:
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

    missing = sorted(path for path in paths if not path.exists())
    if missing:
        for path in missing:
            print(
                f"Path does not exist: {_relative(path, PROJECT_ROOT)}; "
                "refusing to pass empty scan.",
                file=sys.stderr,
            )
        return 1

    tracked, used_fallback = tracked_top_level_entries()
    problems = partition_problems(STRICT_INLINE_PARTITIONS, PENDING_INLINE_PARTITIONS, tracked)
    if problems:
        for problem in problems:
            print(f"comment-quality partition coverage: {problem}", file=sys.stderr)
        return 1
    if used_fallback:
        print(
            "git unavailable; partition coverage computed from a directory walk "
            "(untracked files included).",
            file=sys.stderr,
        )

    default_scope = not args.paths and not args.strict_inline
    if default_scope:
        report = scan_files(paths, strict_partitions=STRICT_INLINE_PARTITIONS)
    else:
        report = scan_files(paths, strict_inline=args.strict_inline)

    if report.files_scanned == 0:
        print("No Python files found; refusing to pass empty scan.", file=sys.stderr)
        return 1

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
    if default_scope:
        summary = " ".join(
            f"{partition}={count}"
            for partition, count in partition_advisory_counts(report.findings)
        )
        print(f"Pending inline partitions (advisory only): {summary}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
