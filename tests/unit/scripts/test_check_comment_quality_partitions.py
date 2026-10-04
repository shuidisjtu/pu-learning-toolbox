"""Tests for the inline-comment partition mechanism of the comment gate.

The default gate run scans ``pu_toolbox/`` and promotes trailing-comment
advisories to errors only inside the declared strict partitions.  These
tests pin the declaration, the coverage assertion, the empty-scan refusal,
and the per-file partition behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_comment_quality as module  # noqa: E402
from check_comment_quality import (  # noqa: E402
    PENDING_INLINE_PARTITIONS,
    PROJECT_ROOT,
    STRICT_INLINE_PARTITIONS,
    is_strict_path,
    main,
    partition_advisory_counts,
    partition_problems,
    scan_files,
    tracked_top_level_entries,
)

# Ratchet baseline: every partition already declared strict must stay strict.
# Adding a partition is a normal change -- append it here in the same commit.
# Removing one is a relaxation, not a refactor: delete it from this baseline
# explicitly, so the loosening shows up in review instead of slipping through
# as an unremarkable edit to the declaration alone.
BASELINE_STRICT_PARTITIONS = (
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


@pytest.mark.unit
def test_basic_partition_constants_cover_tracked_top_level_entries():
    """Strict and pending together classify every tracked package entry."""
    tracked, _ = tracked_top_level_entries()

    assert "pu_toolbox/cli" in tracked
    assert partition_problems(STRICT_INLINE_PARTITIONS, PENDING_INLINE_PARTITIONS, tracked) == []


@pytest.mark.unit
def test_basic_default_run_locks_strict_partitions_without_errors():
    """The declared strict partitions hold no trailing comments to report."""
    package = PROJECT_ROOT / "pu_toolbox"
    strict_files = [
        path
        for path in package.rglob("*.py")
        if "__pycache__" not in path.parts and is_strict_path(path)
    ]

    report = scan_files([package], strict_partitions=STRICT_INLINE_PARTITIONS)

    assert strict_files
    assert report.errors == ()
    assert report.files_scanned > len(strict_files)


@pytest.mark.unit
def test_basic_strict_partitions_never_shrink_below_the_ratchet():
    """A declared strict partition cannot silently return to pending."""
    relaxed = sorted(set(BASELINE_STRICT_PARTITIONS) - set(STRICT_INLINE_PARTITIONS))

    assert relaxed == []


@pytest.mark.unit
def test_edge_core_and_losses_stay_pending_not_strict():
    """A prior ruling keeps core and losses pending, never strict."""
    for partition in ("pu_toolbox/core", "pu_toolbox/losses"):
        assert partition in PENDING_INLINE_PARTITIONS
        assert partition not in STRICT_INLINE_PARTITIONS


@pytest.mark.unit
def test_edge_empty_directory_is_refused(tmp_path, capsys):
    """An existing directory with no Python files never passes silently."""
    empty = tmp_path / "empty"
    empty.mkdir()

    assert main([str(empty)]) == 1
    assert "refusing to pass empty scan" in capsys.readouterr().err


@pytest.mark.unit
def test_edge_missing_positional_path_is_refused(tmp_path, capsys):
    """A positional path that does not exist is refused, not skipped."""
    assert main([str(tmp_path / "absent")]) == 1
    assert "refusing to pass empty scan" in capsys.readouterr().err


@pytest.mark.unit
def test_edge_missing_default_root_is_refused(tmp_path, monkeypatch, capsys):
    """A missing default root is refused exactly like an explicit path."""
    monkeypatch.setattr(module, "DEFAULT_ROOTS", (tmp_path / "pu_toolbox",))

    assert main([]) == 1
    assert "refusing to pass empty scan" in capsys.readouterr().err


@pytest.mark.unit
def test_edge_strict_partition_trailing_comment_is_error(tmp_path):
    """A trailing comment inside a strict partition fails the scan."""
    package = tmp_path / "pu_toolbox"
    (package / "cli").mkdir(parents=True)
    (package / "cli" / "sample.py").write_text("value = 1  # explain this\n", encoding="utf-8")

    report = scan_files(
        [package],
        strict_partitions=STRICT_INLINE_PARTITIONS,
        partitions_root=tmp_path,
    )

    assert len(report.errors) == 1
    assert report.errors[0].kind == "trailing-comment"


@pytest.mark.unit
def test_edge_pending_partition_trailing_comment_stays_advisory(tmp_path):
    """The same comment stays an advisory inside a pending partition."""
    package = tmp_path / "pu_toolbox"
    (package / "core").mkdir(parents=True)
    (package / "core" / "sample.py").write_text("value = 1  # explain this\n", encoding="utf-8")

    report = scan_files(
        [package],
        strict_partitions=STRICT_INLINE_PARTITIONS,
        partitions_root=tmp_path,
    )

    assert report.errors == ()
    assert [finding.severity for finding in report.findings] == ["warning"]


@pytest.mark.unit
def test_edge_tooling_exemption_needs_a_whole_keyword(tmp_path):
    """Keyword-prefixed prose is flagged; real directives stay exempt."""
    package = tmp_path / "pu_toolbox"
    (package / "cli").mkdir(parents=True)
    (package / "cli" / "prose.py").write_text(
        "value = 1  # noqaXYZ a paragraph, not a directive\n", encoding="utf-8"
    )
    (package / "cli" / "directives.py").write_text(
        "a = 1  # noqa\nb = 2  # noqa: E501\nc = 3  # type: ignore[assignment]\n",
        encoding="utf-8",
    )

    report = scan_files(
        [package],
        strict_partitions=STRICT_INLINE_PARTITIONS,
        partitions_root=tmp_path,
    )

    assert len(report.errors) == 1
    assert report.errors[0].path.name == "prose.py"


@pytest.mark.unit
def test_edge_non_utf8_file_reports_scan_error(tmp_path):
    """A file that is not valid UTF-8 yields a finding, never a crash."""
    package = tmp_path / "pu_toolbox"
    (package / "cli").mkdir(parents=True)
    (package / "cli" / "latin1.py").write_bytes(b"value = 1  # \xff\xfe not utf-8\n")

    report = scan_files(
        [package],
        strict_partitions=STRICT_INLINE_PARTITIONS,
        partitions_root=tmp_path,
    )

    assert len(report.errors) == 1
    assert report.errors[0].kind == "scan-error"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("strict", "pending", "tracked", "expected"),
    [
        (
            {"pu_toolbox/a"},
            {"pu_toolbox/b"},
            {"pu_toolbox/a", "pu_toolbox/c"},
            "without a partition",
        ),
        ({"pu_toolbox/a", "pu_toolbox/gone"}, set(), {"pu_toolbox/a"}, "without a tracked entry"),
        ({"pu_toolbox/a"}, {"pu_toolbox/a"}, {"pu_toolbox/a"}, "both strict and pending"),
    ],
)
def test_param_partition_violation_is_reported(strict, pending, tracked, expected):
    """Missing, extra, and duplicated classifications are each reported."""
    problems = partition_problems(strict, pending, tracked)

    assert problems
    assert any(expected in problem for problem in problems)


@pytest.mark.unit
def test_param_broken_declaration_fails_the_gate(monkeypatch, capsys):
    """A declaration set that cannot cover the package fails the default run."""
    monkeypatch.setattr(module, "PENDING_INLINE_PARTITIONS", ("pu_toolbox/cli",))

    assert main([]) == 1
    assert "partition coverage" in capsys.readouterr().err


@pytest.mark.unit
def test_determ_partition_advisory_counts_are_reproducible(tmp_path):
    """Advisory counts per partition repeat identically across runs."""
    package = tmp_path / "pu_toolbox"
    (package / "registry").mkdir(parents=True)
    (package / "registry" / "sample.py").write_text(
        "first = 1  # one\nsecond = 2  # two\n", encoding="utf-8"
    )

    def counts() -> list[tuple[str, int]]:
        report = scan_files(
            [package],
            strict_partitions=STRICT_INLINE_PARTITIONS,
            partitions_root=tmp_path,
        )
        return partition_advisory_counts(report.findings, root=tmp_path)

    assert counts() == counts()
    assert dict(counts())["pu_toolbox/registry"] == 2


@pytest.mark.unit
def test_determ_scan_reports_are_reproducible(tmp_path):
    """Repeated scans of one package yield field-identical reports."""
    package = tmp_path / "pu_toolbox"
    (package / "utils").mkdir(parents=True)
    (package / "utils" / "sample.py").write_text("value = 1  # stable intent\n", encoding="utf-8")

    first = scan_files(
        [package], strict_partitions=STRICT_INLINE_PARTITIONS, partitions_root=tmp_path
    )
    second = scan_files(
        [package], strict_partitions=STRICT_INLINE_PARTITIONS, partitions_root=tmp_path
    )

    # Discriminating assertions: a scan that degenerates to an empty report
    # would compare equal to itself and must not pass this test.
    assert first.files_scanned > 0
    assert first.findings
    assert first == second
