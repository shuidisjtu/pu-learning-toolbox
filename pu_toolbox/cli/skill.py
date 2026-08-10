# ruff: noqa: N802, N803, N806, E501

"""The ``skill`` subcommand: install the bundled pu-workflow skill.

The Agent-Skills ``SKILL.md`` files ship inside the wheel, so pip users
can enable the pu-workflow workflow without cloning the repo.  The
command copies the assets into ``~/.claude/skills/pu-workflow`` and
``~/.agents/skills/pu-workflow`` (or ``--dest``), skipping existing
installations unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .. import __version__
from ..core.exceptions import PULearningError

__all__ = ["build_skill_parser", "run_skill"]

_ASSET_DIR = ".claude/skills/pu-workflow"
_TARGETS = (".claude/skills/pu-workflow", ".agents/skills/pu-workflow")


def build_skill_parser(sub: argparse._SubParsersAction) -> None:
    """Attach the ``skill`` subcommand to *sub* (side-effect only)."""
    parser = sub.add_parser(
        "skill",
        help="manage the bundled pu-workflow skill (Agent Skills)",
    )
    actions = parser.add_subparsers(dest="skill_action", required=True, metavar="ACTION")
    install = actions.add_parser(
        "install", help="install the skill into the user's agent directories"
    )
    install.add_argument(
        "--force", action="store_true", help="overwrite an existing skill installation"
    )
    install.add_argument(
        "--dest",
        default=None,
        help="installation root directory (default: the user home directory)",
    )
    install.set_defaults(func=run_skill)


def _skill_source() -> Path:
    """Locate the bundled skill assets (repo, editable, or wheel layout).

    ``parents[2]`` is the project root when running from source and the
    ``site-packages`` root when installed from the wheel (the wheel
    bundles ``.claude/skills/pu-workflow`` beside ``pu_toolbox``).
    """
    return Path(__file__).resolve().parents[2] / _ASSET_DIR


def run_skill(args: argparse.Namespace) -> None:
    """Copy the bundled skill into the user-level agent directories."""
    source = _skill_source()
    if not source.is_dir():
        raise PULearningError(
            f"skill assets not found at {source}; reinstall pu-toolbox {__version__}"
        )
    dest_root = Path(args.dest) if args.dest else Path.home()
    for target in _TARGETS:
        dest = dest_root / target
        if dest.exists() and not args.force:
            print(f"info: {dest} already installed; use --force to overwrite")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dest, dirs_exist_ok=True)
        print(f"installed pu-workflow skill (from pu-toolbox {__version__}) to {dest}")
