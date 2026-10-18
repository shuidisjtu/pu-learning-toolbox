"""Read-only provenance for exploratory scripts, independent of caller cwd."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def source_provenance(script, implementations):
    """Bind script-checkout Git separately from the actually loaded source files.

    Missing Git identity is explicit, never borrowed from a parent/caller repo.
    This records bytes on disk, not proof against concurrent source mutation.
    It does not authorize formal runs or certify a frozen runtime environment.
    """
    script = Path(script).resolve()
    root = script.parents[1]
    if not isinstance(implementations, dict) or not implementations:
        raise ValueError("loaded implementation paths must be a nonempty mapping")
    commit, dirty, state = None, None, "unavailable"

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout

    try:
        top = Path(git("rev-parse", "--show-toplevel").strip()).resolve()
        if top == root:
            commit = git("rev-parse", "HEAD").strip()
            if len(commit) not in {40, 64} or any(c not in "0123456789abcdef" for c in commit):
                raise ValueError("unverified Git commit identity")
            dirty = bool(git("status", "--porcelain"))
            state = "script_checkout"
        else:
            state = "not_script_checkout"
    except (OSError, ValueError, subprocess.SubprocessError):
        commit, dirty = None, None
    loaded = {}
    for method, reference in implementations.items():
        path = Path(reference).resolve()
        loaded[method] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "inside_script_checkout": path.is_relative_to(root),
        }
    return {
        "script_checkout_root": str(root),
        "script_checkout_commit": commit,
        "script_checkout_dirty": dirty,
        "git_identity_status": state,
        "git_commit_scope": "script_checkout_only_not_loaded_distribution",
        "loaded_implementations": loaded,
        "all_loaded_implementations_inside_script_checkout": all(
            row["inside_script_checkout"] for row in loaded.values()
        ),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "provenance_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "formal_admission": False,
    }
