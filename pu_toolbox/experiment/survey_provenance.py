"""P2.2 report provenance: which code and which inputs produced this report.

The P2.2 implementation plan requires every report to record its input result
roots, the code commit, the protocol digest, the comparison-matrix digest and a
generation time.  Three entry points produce those reports, so the block is
assembled here once instead of three times.

**This is not the per-manifest identity check.**  Protocol §5 clause 6 asks for
reproducibility of a *result*, which the survey runner satisfies through the
batch plan and the evidence packet: that identity is recorded once per batch
rather than once per manifest, which is what ``survey_audit.BATCH_IDENTITY_FIELDS``
names.  The block here answers a different question, asked of a *report*: which
checkout and which input tree was read when this report was written?  A reader who
conflates the two will look for a commit in every manifest and find none.

**Why the git call is not in the builders.**  ``build_audit`` and
``build_summary`` are pure functions of what they are handed, and there is a
determinism check that compares two builds over one tree.  Reading a commit is
process I/O, so it happens only in each entry point's ``main``, through
:func:`resolve_code_commit`.  Everything else here is pure: no clock, no
filesystem.

**A digest that was not read is not a digest.**  ``code_commit_state`` is
``"recorded"`` or ``"unavailable"``, never absent, so a report produced outside
a git work tree says so instead of leaving a reader to guess whether the field
was forgotten.  The same reasoning drives rendering the commit rather than
omitting it.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from typing import Any

#: The two states the ``code_commit_state`` field may take.  ``unavailable`` is a first-class
#: value: a report whose commit could not be read is not a report whose commit
#: matched something, and collapsing the two is how an unreadable identity turns
#: into a silent claim.
CODE_COMMIT_RECORDED = "recorded"
CODE_COMMIT_UNAVAILABLE = "unavailable"
CODE_COMMIT_STATES: tuple[str, ...] = (CODE_COMMIT_RECORDED, CODE_COMMIT_UNAVAILABLE)

_GIT_COMMIT: tuple[str, ...] = ("git", "rev-parse", "HEAD")
_GIT_STATUS: tuple[str, ...] = ("git", "status", "--porcelain")


def _git_output(argv: tuple[str, ...]) -> str | None:
    """The stripped stdout of *argv*, or ``None`` when git cannot be read.

    Not fatal, and deliberately loud about it: a report is still worth writing
    without a commit, but silence would read as "there was nothing to record".
    """
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        print(
            f"warning: {' '.join(argv)} failed; the report records no code identity",
            file=sys.stderr,
        )
        return None
    return result.stdout.strip()


def input_result_roots(config: Mapping[str, Any]) -> dict[str, str]:
    """``{batch name: result root}`` from a batch-root whitelist config.

    Pure: reads the mapping it is handed, never the filesystem.  A batch without
    a usable ``root`` is omitted rather than invented -- the whitelist validator
    already refuses such a config, so an omission here means a caller built the
    mapping by hand.
    """
    roots: dict[str, str] = {}
    for batch in config.get("batches") or ():
        if not isinstance(batch, Mapping):
            continue
        name, root = batch.get("name"), batch.get("root")
        if isinstance(name, str) and name and isinstance(root, str) and root:
            roots[name] = root
    return dict(sorted(roots.items()))


def recorded_source_roots(config: Mapping[str, Any]) -> dict[str, str]:
    """``{batch name: upstream source root}`` from a whitelist config, or ``{}``.

    ``source_roots`` is not part of the whitelist schema -- it is an optional
    block recording where the manifests were produced.  Recorded when present
    because a report whose checkpoint paths point at another machine should say
    so itself, rather than leaving the reader to find the note elsewhere.
    """
    source = config.get("source_roots")
    if not isinstance(source, Mapping):
        return {}
    return dict(
        sorted((str(name), root) for name, root in source.items() if isinstance(root, str) and root)
    )


def build_provenance(
    *,
    protocol_version: str | None,
    protocol_sha256: str | None,
    comparison_version: str | None = None,
    comparison_sha256: str | None = None,
    result_roots: Mapping[str, str] | None = None,
    source_roots: Mapping[str, str] | None = None,
    code_commit: str | None = None,
    code_commit_dirty: bool | None = None,
) -> dict[str, Any]:
    """The identity block, with every key present on every path.

    Pure: no clock and no I/O, so two builds over one tree are equal -- which is
    what lets a determinism check compare them.  The caller passes what it
    actually read; a field it could not read stays ``None`` rather than being
    filled with a placeholder that looks like a reading.
    """
    return {
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "comparison_version": comparison_version,
        "comparison_sha256": comparison_sha256,
        "input_result_roots": dict(sorted((result_roots or {}).items())),
        "source_roots": dict(sorted((source_roots or {}).items())),
        "code_commit": code_commit,
        "code_commit_dirty": code_commit_dirty,
        "code_commit_state": (CODE_COMMIT_RECORDED if code_commit else CODE_COMMIT_UNAVAILABLE),
    }


def with_code_commit(
    block: Mapping[str, Any] | None, code_commit: str | None, *, dirty: bool | None = None
) -> dict[str, Any]:
    """A copy of *block* carrying the commit and its state.

    Split out of :func:`build_provenance` so the pure builders never see a
    commit: only an entry point's ``main`` calls this, after
    :func:`resolve_code_commit`.  Accepts ``None`` so a hand-built report still
    stamps rather than raising on the way out.
    """
    if block is None:
        return build_provenance(
            protocol_version=None,
            protocol_sha256=None,
            code_commit=code_commit,
            code_commit_dirty=dirty,
        )
    stamped = dict(block)
    stamped["code_commit"] = code_commit
    stamped["code_commit_dirty"] = dirty
    stamped["code_commit_state"] = CODE_COMMIT_RECORDED if code_commit else CODE_COMMIT_UNAVAILABLE
    return stamped


def resolve_code_commit() -> tuple[str | None, bool | None]:
    """``(commit, worktree_dirty)`` for the current checkout, or ``(None, None)``.

    The one function here that touches the process table.  ``dirty`` is reported
    because it changes what the commit means: a report produced from a modified
    work tree names a commit that does not contain the code that produced it, and
    a reader who is not told will take it as a complete identity.  ``None`` means
    the question could not be asked, which is not the same answer as ``False``.
    """
    commit = _git_output(_GIT_COMMIT)
    if commit is None:
        return None, None
    status = _git_output(_GIT_STATUS)
    return commit, None if status is None else bool(status)


def render_provenance_lines(block: Mapping[str, Any] | None) -> list[str]:
    """Markdown bullets for *block*, or ``[]`` when there is no block.

    **Stamped-free on purpose.**  A summary's Markdown and CSV are pinned to be
    byte-identical across two runs of the same input, so nothing here may read a
    clock, and every collection is sorted by batch name rather than left in
    mapping order.

    The protocol and comparison digests are *not* re-rendered: each report
    already prints them above, and a second copy is a second chance to disagree
    with the first.
    """
    if not block:
        return []
    lines: list[str] = []
    commit = block.get("code_commit")
    if commit:
        # Three states, not two: a work tree whose cleanliness could not be read
        # is not a clean one, and printing it bare would read as clean.  Same
        # doctrine as ``unavailable`` above -- an absence is stated, not implied.
        dirty = block.get("code_commit_dirty")
        if dirty is True:
            changed = "（工作区有未提交改动）"
        elif dirty is None:
            changed = "（工作区状态未能读取）"
        else:
            changed = ""
        lines.append(f"- 代码 commit：`{commit}`{changed}")
    else:
        state = block.get("code_commit_state") or CODE_COMMIT_UNAVAILABLE
        lines.append(f"- 代码 commit：不可用（`{state}`，不在 git 工作树内）")
    for name, root in sorted((block.get("input_result_roots") or {}).items()):
        lines.append(f"- 输入结果根 `{name}`：`{root}`")
    for name, root in sorted((block.get("source_roots") or {}).items()):
        lines.append(f"- 上游源根 `{name}`：`{root}`（仅溯源）")
    return lines
