"""Read-only rebuild diagnostics; never accept a changed original receipt.

A ZIP creator-system byte can differ between Windows and Unix even when all
compressed payload bytes match. Diagnose ONLY that difference, retaining the
original verifier's failure and every missing/unexpected file. No files are
rewritten. Original archive receipt/signature verification is not performed.
Exit 2 when the unpacked reference index does not match, even if diagnosed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import zipfile
import zlib
from collections import Counter
from pathlib import Path

from pu_toolbox.experiment.split_archive import (
    MANIFEST_NAME,
    SPLIT_ROLE_FILES,
    load_index,
    read_indices,
    verify,
)
from pu_toolbox.utils.serialization import canonical_hash, file_hash

MAX_UNCOMPRESSED_BYTES = 512 * 1024**2


def _validate_reference(root, index):
    if not root.is_dir() or not isinstance(index["datasets"], dict) or not index["datasets"]:
        raise ValueError("existing rebuild root and nonempty reference datasets required")
    for dataset, entries in index["datasets"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", dataset):
            raise ValueError("unsafe reference dataset name")
        if not isinstance(entries, dict) or not set(entries) - {"archive"}:
            raise ValueError("reference dataset must describe at least one split")
        for split, entry in entries.items():
            if split == "archive":
                continue
            if not re.fullmatch(r"split_[0-9]+", split):
                raise ValueError("unsafe reference split name")
            expected_names = {*SPLIT_ROLE_FILES, MANIFEST_NAME}
            if set(entry["files"]) != expected_names:
                raise ValueError("reference must describe all four roles and manifest")
            for component in (root / dataset, root / dataset / split):
                if component.is_symlink():
                    raise ValueError("rebuild dataset/split symlinks are not supported")
            for name, record in entry["files"].items():
                if type(record["bytes"]) is not int or record["bytes"] < 0:
                    raise ValueError("invalid reference file size")
                if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
                    raise ValueError("invalid reference file digest")
                if (root / dataset / split / name).is_symlink():
                    raise ValueError("rebuild file symlinks are not supported")


def _creator_hashes(path):
    """Counterfactual hashes changing only central-directory creator bytes.

    Read and CRC-check bounded ordinary NPZ members, then stream hashing with
    sparse byte substitutions. No decompression/recompression or array edits.
    The result cannot hide payload, timestamp, permissions or comment changes.
    """
    offsets = []
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if sorted(m.filename for m in members) != ["X.npy", "indices.npy", "y.npy"]:
            raise ValueError("not an ordinary three-member survey NPZ")
        if sum(m.file_size for m in members) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("NPZ exceeds bounded diagnostic uncompressed size")
        if any(m.flag_bits & 1 or m.create_system not in (0, 3) for m in members):
            raise ValueError("encrypted or unsupported creator-system NPZ")
        if archive.testzip() is not None:
            raise ValueError("NPZ CRC mismatch")
        with path.open("rb") as source:
            source.seek(archive.start_dir)
            for _ in members:
                position = source.tell()
                header = source.read(46)
                if len(header) != 46 or header[:4] != b"PK\x01\x02":
                    raise ValueError("invalid ZIP central directory")
                offsets.append((position + 5, header[5]))
                source.seek(sum(struct.unpack_from("<HHH", header, 28)), 1)
    digesters = {system: hashlib.sha256() for system in (0, 3)}
    with path.open("rb") as source:
        position = 0
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            for system, digest in digesters.items():
                patched = bytearray(chunk)
                for offset, _ in offsets:
                    if position <= offset < position + len(chunk):
                        patched[offset - position] = system
                digest.update(patched)
            position += len(chunk)
    return {
        system: {
            "sha256": digest.hexdigest(),
            "changed_creator_bytes": sum(original != system for _, original in offsets),
        }
        for system, digest in digesters.items()
    }


def _file_diagnostic(path, expected, *, npz):
    result = {"expected_sha256": expected["sha256"], "expected_bytes": expected["bytes"]}
    if not path.is_file():
        return {**result, "status": "missing"}
    try:
        result.update(actual_sha256=file_hash(path), actual_bytes=path.stat().st_size)
    except OSError as exc:
        return {**result, "status": "unreadable", "error": str(exc)}
    if (
        result["actual_sha256"] == expected["sha256"]
        and result["actual_bytes"] == expected["bytes"]
    ):
        return {**result, "status": "byte_identical"}
    if npz and result["actual_bytes"] == expected["bytes"]:
        try:
            candidates = _creator_hashes(path)
        except (
            OSError,
            ValueError,
            zipfile.BadZipFile,
            RuntimeError,
            EOFError,
            zlib.error,
            struct.error,
        ) as exc:
            return {**result, "status": "different", "zip_diagnostic_error": str(exc)}
        for system, candidate in candidates.items():
            if candidate["sha256"] == expected["sha256"]:
                return {
                    **result,
                    "status": "zip_creator_system_only",
                    "creator_system": system,
                    "counterfactual": candidate,
                }
    return {**result, "status": "different"}


def diagnose(root: Path, index_path: Path):
    """Keep strict verification intact; add narrow cross-platform evidence."""
    root, index_path = Path(root), Path(index_path)
    index = load_index(index_path)
    _validate_reference(root, index)
    strict = verify(root, index)
    splits, role_counts, manifest_counts, index_counts = [], Counter(), Counter(), Counter()
    for dataset, entries in sorted(index["datasets"].items()):
        for split, entry in sorted(entries.items()):
            if split == "archive":
                continue
            directory = root / dataset / split
            files = {}
            for name in (*SPLIT_ROLE_FILES, MANIFEST_NAME):
                record = _file_diagnostic(
                    directory / name, entry["files"][name], npz=name.endswith(".npz")
                )
                files[name] = record
                (manifest_counts if name == MANIFEST_NAME else role_counts)[record["status"]] += 1
            try:
                actual = canonical_hash(
                    {
                        name.removesuffix(".npz"): read_indices(directory / name)
                        for name in SPLIT_ROLE_FILES
                    }
                )
                indices = dict(
                    status="identical" if actual == entry["indices_sha256"] else "different",
                    expected=entry["indices_sha256"],
                    actual=actual,
                )
            except ValueError as exc:
                indices = dict(status="not_verified", error=str(exc))
            index_counts[indices["status"]] += 1
            splits.append(dict(dataset=dataset, split=split, files=files, role_indices=indices))
    return {
        "schema_version": "survey-split-rebuild-diagnostic-1",
        "reference_index_file_sha256": file_hash(index_path),
        "strict_unpacked_reference_index": strict,
        "original_archive_receipt_verified": False,
        "formal_acceptance": False,
        "files_rewritten": False,
        "scope": "rebuild diagnostic only; creator-byte match is not receipt acceptance",
        "summary": dict(
            expected_splits=len(splits),
            role_files=dict(role_counts),
            manifests=dict(manifest_counts),
            role_indices=dict(index_counts),
        ),
        "splits": splits,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = diagnose(args.root, args.index)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Rebuild diagnostic failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["strict_unpacked_reference_index"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
