"""Pack the survey split artifacts for transfer, and verify them where they land.

``data/`` is never distributed, so the P1.4 products travel out of band: this
tool is both ends of that trip.  ``pack`` writes one deterministic tar per
dataset plus an index describing every file in them; ``verify`` runs on the
receiving machine, against the unpacked tree, and reports every disagreement
between that tree and the index -- including the one nothing else in the
repository checks, that the ``.npz`` indices still hash to the value the
manifest records for them.

The index's archive digests are meant to be recorded in the repository (the
issue or a delivery note) before the archives are sent.  A digest that
travelled with the bytes it describes proves the trip was faithful, not that
what was sent was right; the repository copy is what the receiver compares
against.

Usage::

    uv run python scripts/survey_splits_archive.py pack \\
        --root data/splits --out-dir dist/p1.4-splits

    uv run python scripts/survey_splits_archive.py verify \\
        --root unpacked/ --index dist/p1.4-splits/split_artifacts_index.json \\
        --archive-dir dist/p1.4-splits
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pu_toolbox.experiment.split_archive import dump_index, load_index, pack, verify
from pu_toolbox.experiment.survey_protocol import load_protocol

INDEX_NAME = "split_artifacts_index.json"


def _producing_commit() -> str | None:
    """The commit the artifacts were produced from, or ``None`` outside a repo.

    Not fatal: the index is still worth writing without it, and the caller's
    own ``git log`` can answer the question -- but a delivery whose commit is
    unknown cannot be reproduced, so say so out loud rather than in silence.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        print("warning: not inside a git work tree; the index records no commit", file=sys.stderr)
        return None
    return result.stdout.strip()


def _protocol_version() -> str:
    return str(load_protocol()["protocol_version"])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pack_parser = sub.add_parser("pack", help="build the index and the per-dataset archives")
    pack_parser.add_argument("--root", default="data/splits", help="split products root")
    pack_parser.add_argument("--out-dir", required=True, help="where archives and index go")
    pack_parser.add_argument(
        "--datasets", default=None, help="comma-separated subset; default is every dataset found"
    )

    verify_parser = sub.add_parser("verify", help="check a tree against an index")
    verify_parser.add_argument("--root", required=True, help="unpacked split products root")
    verify_parser.add_argument("--index", required=True, help=f"path to {INDEX_NAME}")
    verify_parser.add_argument(
        "--archive-dir",
        default=None,
        help="directory holding the archives, when their digests are to be checked too",
    )
    return parser.parse_args(argv)


def _print_problems(problems: list[dict[str, Any]]) -> None:
    for problem in problems:
        extra = " ".join(
            f"{key}={value}" for key, value in problem.items() if key not in {"kind", "path"}
        )
        print(
            f"  {problem['kind']}: {problem['path']}{' ' + extra if extra else ''}", file=sys.stderr
        )


def _run_pack(args: argparse.Namespace) -> int:
    datasets = (
        [name.strip() for name in args.datasets.split(",") if name.strip()]
        if args.datasets
        else None
    )
    try:
        index = pack(
            Path(args.root),
            Path(args.out_dir),
            datasets=datasets,
            commit=_producing_commit(),
            protocol_version=_protocol_version(),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    index_path = Path(args.out_dir) / INDEX_NAME
    dump_index(index, index_path)
    for dataset, entry in sorted(index["datasets"].items()):
        archive = entry["archive"]
        splits = [name for name in entry if name != "archive"]
        print(
            f"{dataset}: {len(splits)} split(s) -> {archive['file']} "
            f"({archive['bytes']} bytes, sha256={archive['sha256']})"
        )
    print(f"index: {index_path}")
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    try:
        index = load_index(args.index)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report = verify(Path(args.root), index, archive_dir=args.archive_dir)
    checked = report["checked"]
    if report["ok"]:
        print(f"ok: {checked['splits']} split(s), {checked['files']} file(s) match {args.index}")
        return 0
    print(
        f"FAILED: {len(report['problems'])} problem(s) across "
        f"{checked['splits']} split(s), {checked['files']} file(s)",
        file=sys.stderr,
    )
    _print_problems(report["problems"])
    return 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return _run_pack(args) if args.command == "pack" else _run_verify(args)


if __name__ == "__main__":
    sys.exit(main())
