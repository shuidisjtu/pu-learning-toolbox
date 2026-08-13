"""Backward-compatible benchmark helper imports and shared git provenance primitives."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pu_toolbox.utils.serialization import canonical_hash as canonical_hash

__all__ = ["canonical_hash", "git_worktree_dirty"]


def git_worktree_dirty(project_root: Path, exclude: Path | None = None) -> bool | None:
    """Return True when the repository has uncommitted changes, None when git fails.

    ``exclude``, when located under ``project_root``, is omitted from the status
    via pathspecs so a runner's own output directory does not count as dirty.
    """
    command = ["git", "status", "--porcelain", "--untracked-files=all"]
    if exclude is not None:
        try:
            relative = exclude.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            pass
        else:
            command.extend(["--", ".", f":(exclude){relative}", f":(exclude){relative}/**"])
    try:
        status = subprocess.run(
            command, cwd=project_root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(status.stdout.strip())
