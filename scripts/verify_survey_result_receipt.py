#!/usr/bin/env python3
"""Verify received P2.2 bytes before replay; never train, extract or modify inputs.

Roots are supplied explicitly as ``--batch-root B1=/absolute/path`` (repeatable).
Partial delivery exits 2, identity errors exit 1, complete input identity exits
0. None of these outcomes is a scientific verdict or a human signature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_INDEX = (
    Path(__file__).resolve().parents[1]
    / "docs/research/pu_survey/data/p2_2_artifacts_index_20261009.json"
)


def _file_identity(path: Path) -> dict[str, Any]:
    """Hash regular, stable files only; an in-flight upload is not a receipt."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    current = path.lstat()
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    identities = [
        tuple(getattr(item, key) for key in keys) for item in (before, opened, after, current)
    ]
    if any(value != identities[0] for value in identities[1:]):
        raise ValueError(f"file changed during verification: {path}")
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


def _validate_identity(value: Any, *, tree: bool) -> None:
    if not isinstance(value, dict):
        raise ValueError("index identity must be an object")
    for key in ("bytes", "manifests") if tree else ("bytes",):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError(f"index {key} must be a non-negative integer")
    key = "tree_sha256" if tree else "sha256"
    if not isinstance(value.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
        raise ValueError(f"index {key} must be a lowercase SHA256")


def _load_index(path: Path) -> dict[str, Any]:
    def unique_keys(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate index key: {key}")
            result[key] = item
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite index constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_keys,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict) or value.get("schema_version") != "survey-p22-artifacts-index-1":
        raise ValueError("unsupported P2.2 index")
    inputs = value.get("inputs", {})
    batches = inputs.get("batches") if isinstance(inputs, dict) else None
    if not isinstance(batches, dict) or not batches:
        raise ValueError("index needs a non-empty inputs.batches object")
    for name, identity in batches.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("unsafe batch name in index")
        _validate_identity(identity, tree=True)
    _validate_identity(inputs.get("total"), tree=True)
    for key in ("manifests", "bytes"):
        if sum(identity[key] for identity in batches.values()) != inputs["total"][key]:
            raise ValueError(f"index batch {key} do not sum to total")
    for section in ("artifacts", "evidence"):
        entries = value.get(section, {})
        if not isinstance(entries, dict):
            raise ValueError(f"index {section} must be an object")
        for relative, identity in entries.items():
            if not isinstance(relative, str) or "\\" in relative:
                raise ValueError("unsafe artifact path in index")
            parts = PurePosixPath(relative)
            if parts.is_absolute() or ".." in parts.parts or not parts.parts or ":" in relative:
                raise ValueError("unsafe artifact path in index")
            _validate_identity(identity, tree=False)
    return value


def _tree(root: Path) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    records = []

    def raise_walk_error(error):
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=raise_walk_error):
        for name in dirs:
            child = Path(directory) / name
            if child.is_symlink():
                raise ValueError(f"symlink directory in input tree: {child}")
            if name == "B3ab_merged":
                raise ValueError("derived B3ab_merged working copy is not an original batch")
        if "manifest.json" in files:
            path = Path(directory) / "manifest.json"
            identity = _file_identity(path)
            records.append((path.relative_to(root).as_posix(), identity))
    records.sort(key=lambda item: item[0])
    lines = [(relative, identity["sha256"]) for relative, identity in records]
    payload = "".join(f"{sha}  {relative}\n" for relative, sha in lines)
    return {
        "manifests": len(records),
        "bytes": sum(identity["bytes"] for _, identity in records),
        "tree_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }, lines


def review_receipt(
    index_path: Path, roots: dict[str, Path], *, reports: Path | None = None
) -> dict[str, Any]:
    """Compare exact bytes with one selected historical index, batch by batch."""
    index_before = _file_identity(index_path)
    index = _load_index(index_path)
    expected_batches = index["inputs"]["batches"]
    if set(roots) - set(expected_batches):
        raise ValueError("unknown batch roots; do not merge or rename historical batches")
    resolved = {name: root.resolve() for name, root in roots.items()}
    for name, root in resolved.items():
        if roots[name].is_symlink() or root.name == "B3ab_merged":
            raise ValueError("batch root must not be a symlink or derived working copy")
        for other, other_root in resolved.items():
            if name != other and (root == other_root or root in other_root.parents):
                raise ValueError("batch roots must be distinct and non-overlapping")
    report: dict[str, Any] = {
        "schema_version": "survey-p22-receipt-review-1",
        "index": {"path": str(index_path), **index_before},
        "produced_by_commit": index.get("produced_by_commit"),
        "identity": index.get("identity"),
        "scope": "exact input bytes only; no replay, weight restoration or human approval",
        "formal_acceptance": False,
        "batches": {},
        "problems": [],
        "missing": [],
        "reports": {"status": "not_requested", "files": {}},
    }
    total_lines = []
    for name, expected in expected_batches.items():
        root = resolved.get(name)
        if root is None or not root.is_dir():
            report["missing"].append(f"{name}: original batch root unavailable")
            report["batches"][name] = {"status": "missing", "expected": expected}
            continue
        observed, lines = _tree(root)
        status = "verified"
        if observed["manifests"] < expected["manifests"]:
            status = "incomplete"
            report["missing"].append(f"{name}: fewer manifests than indexed")
        elif observed != expected:
            status = "mismatch"
            report["problems"].append(f"{name}: manifest count, bytes or tree digest differs")
        report["batches"][name] = {"status": status, "expected": expected, "observed": observed}
        total_lines.extend(f"{sha}  {name}/{relative}\n" for relative, sha in lines)
    if all(batch["status"] == "verified" for batch in report["batches"].values()):
        observed_total = {
            "manifests": sum(
                batch["observed"]["manifests"] for batch in report["batches"].values()
            ),
            "bytes": sum(batch["observed"]["bytes"] for batch in report["batches"].values()),
            "tree_sha256": hashlib.sha256("".join(total_lines).encode("utf-8")).hexdigest(),
        }
        report["total"] = {"expected": index["inputs"]["total"], "observed": observed_total}
        if observed_total != index["inputs"]["total"]:
            report["problems"].append("overall manifest tree identity differs")
    if reports is not None:
        reports_root = reports.resolve()
        for section in ("artifacts", "evidence"):
            for relative, expected in index.get(section, {}).items():
                path = reports_root / relative
                if not path.resolve().is_relative_to(reports_root) or any(
                    parent.is_symlink()
                    for parent in (path, *path.parents)
                    if parent != reports_root
                ):
                    raise ValueError("report path escapes receipt root or contains a symlink")
                if not path.exists():
                    report["missing"].append(f"{section}/{relative}: unavailable")
                    report["reports"]["files"][relative] = {"status": "missing"}
                    continue
                observed = _file_identity(path)
                status = "verified" if observed == expected else "mismatch"
                report["reports"]["files"][relative] = {"status": status, "observed": observed}
                if status == "mismatch":
                    report["problems"].append(f"{section}/{relative}: file identity differs")
        statuses = {item["status"] for item in report["reports"]["files"].values()}
        report["reports"]["status"] = (
            "failed"
            if "mismatch" in statuses
            else "incomplete"
            if "missing" in statuses
            else "verified"
        )
    if _file_identity(index_path) != index_before:
        raise ValueError("index changed during verification")
    report["status"] = (
        "failed" if report["problems"] else "incomplete" if report["missing"] else "verified"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--batch-root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--reports", type=Path, help="root containing original indexed report/evidence directories"
    )
    args = parser.parse_args(argv)
    try:
        roots = {}
        for item in args.batch_root:
            name, separator, path = item.partition("=")
            if not separator or not path or name in roots:
                raise ValueError("batch roots require unique NAME=PATH entries")
            roots[name] = Path(path)
        report = review_receipt(args.index, roots, reports=args.reports)
    except (OSError, ValueError, TypeError, KeyError) as error:
        report = {"status": "failed", "formal_acceptance": False, "problems": [str(error)]}
    print(json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2))
    return {"verified": 0, "failed": 1, "incomplete": 2}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
