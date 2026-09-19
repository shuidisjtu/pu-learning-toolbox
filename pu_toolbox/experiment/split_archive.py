"""Pack the survey split artifacts for transfer, and verify them on arrival.

``data/`` is never distributed -- the split products are machine-local by
design -- so what P1.4 prepares has to reach another machine by hand.  The
consuming side checks the artifacts' *shape*: the runner calls
``validate_bundle`` and refuses a manifest naming another dataset or seed.
Nothing checks their *integrity*.  No digest of the ``.npz`` files is compared
to anything: the runner copies a manifest's ``indices_sha256`` into each run
manifest without ever recomputing it, so a set that was truncated in transit,
or that arrived beside the wrong manifest, would be accepted silently and
surface later as an unexplained result.

This module is the missing half.  :func:`build_index` records what every file
is and what each split's indices hash to; :func:`verify` recomputes both from
the tree on disk and reports *every* disagreement rather than the first, in
both directions -- what the index describes but the tree lacks, and what the
tree holds but the index does not.  :func:`pack` writes a deterministic
archive per dataset so that two machines packing the same tree produce the
same bytes.

A digest that arrives alongside the bytes it describes proves only that the
transfer was faithful, not that the sender was right.  The archive digests in
the index are therefore meant to be recorded in the repository, which is what
gives the receiver something to compare against that did not come from the
sender.

Deviations worth stating, because they bound what a clean report means.  The
index pins ``X`` only by file digest, so corruption that predates packing is
endorsed by it rather than caught; what the re-derivation covers is the
indices a run consumes.  The positive rates and the preprocessing block are
the manifest's own claims about itself and are pinned by nothing but that
manifest's file digest.  And this is a tool the operator runs -- it is not
wired into the run path, so a delivery nobody verifies is a delivery nobody
has verified.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from pu_toolbox.utils.serialization import canonical_hash

from .survey_protocol import ROLES

#: The files every split directory must hold, beyond ``split_manifest.json``.
SPLIT_ROLE_FILES: tuple[str, ...] = tuple(f"{role}.npz" for role in ROLES)

MANIFEST_NAME = "split_manifest.json"

#: Bumped when the index stops meaning what this reader thinks it means.  A
#: reader that guessed at a newer index would report a delivery as verified
#: against a document it did not understand.
INDEX_SCHEMA_VERSION = "1.0"

#: 1 MiB read chunks: large enough that hashing a 800 MB split is I/O bound
#: rather than syscall bound, small enough not to hold a partition in memory.
_CHUNK_BYTES = 1024 * 1024


def file_sha256(path: str | Path) -> str:
    """Stream a file through SHA-256 without loading it whole."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_indices(path: Path) -> list[int]:
    """Read one role's indices, raising ``ValueError`` for an unreadable product.

    The ``except`` is deliberately broad.  Which exception a damaged ``.npz``
    raises depends entirely on which byte moved: ``BadZipFile`` from the
    central directory, ``zlib.error`` from a deflate stream, ``EOFError`` from
    a truncated member, ``KeyError`` or ``ValueError`` from numpy.  A verifier
    that dies on one of them dies on exactly the input it exists to catch, so
    the shape of the failure is not something to enumerate -- only the fact
    that this file could not be read.
    """
    try:
        with np.load(path, allow_pickle=False) as payload:
            return payload["indices"].tolist()
    except Exception as exc:  # noqa: BLE001 - any unreadable product is one finding
        raise ValueError(f"{path} cannot be read as a split product: {exc}") from exc


def _split_dirs(root: Path, datasets: Iterable[str] | None) -> list[tuple[str, Path]]:
    """Every ``(dataset, split_dir)`` to cover, in a stable order."""
    names = sorted(datasets) if datasets is not None else sorted(p.name for p in root.iterdir())
    found: list[tuple[str, Path]] = []
    for dataset in names:
        dataset_dir = root / dataset
        if not dataset_dir.is_dir():
            raise ValueError(f"no such dataset directory: {dataset_dir}")
        split_dirs = sorted(
            path for path in dataset_dir.iterdir() if path.name.startswith("split_")
        )
        if not split_dirs:
            raise ValueError(f"{dataset_dir} holds no split_<seed> directory")
        found.extend((dataset, path) for path in split_dirs)
    if not found:
        raise ValueError(f"{root} holds no dataset directories")
    return found


def _split_entry(split_dir: Path) -> dict[str, Any]:
    """Describe one split: its files, and what its indices must hash to.

    The manifest is read rather than trusted: the digest recorded here is
    recomputed from the ``.npz`` files, so a manifest that disagrees with the
    data beside it fails at pack time instead of at the receiver.
    """
    manifest_path = split_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"{split_dir} has no {MANIFEST_NAME}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: dict[str, Any] = {
        MANIFEST_NAME: {
            "sha256": file_sha256(manifest_path),
            "bytes": manifest_path.stat().st_size,
        }
    }
    role_indices: dict[str, list[int]] = {}
    for name in SPLIT_ROLE_FILES:
        path = split_dir / name
        if not path.is_file():
            raise ValueError(f"{split_dir} is missing {name}")
        files[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        role_indices[name.removesuffix(".npz")] = read_indices(path)
    recomputed = canonical_hash(role_indices)
    if recomputed != manifest.get("indices_sha256"):
        raise ValueError(
            f"{split_dir}: the .npz indices hash to {recomputed} but the manifest "
            f"records {manifest.get('indices_sha256')}; one of the two is not the "
            "artifact that was produced"
        )
    return {
        "manifest_sha256": files[MANIFEST_NAME]["sha256"],
        "indices_sha256": recomputed,
        "role_sizes": {role: len(indices) for role, indices in role_indices.items()},
        "files": files,
    }


def build_index(
    root: str | Path,
    *,
    datasets: Iterable[str] | None = None,
    commit: str | None = None,
    protocol_version: str | None = None,
) -> dict[str, Any]:
    """Describe every split under ``root`` well enough to detect any change.

    Fails rather than recording an incomplete set: an index built over a tree
    that is already wrong would certify the wrong thing.
    """
    root = Path(root)
    catalog: dict[str, dict[str, Any]] = {}
    for dataset, split_dir in _split_dirs(root, datasets):
        catalog.setdefault(dataset, {})[split_dir.name] = _split_entry(split_dir)
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "generator": "pu_toolbox.experiment.split_archive",
        "protocol_version": protocol_version,
        "produced_by_commit": commit,
        "datasets": catalog,
    }


def _archive_member(name: str, size: int) -> tarfile.TarInfo:
    """A tar header with everything host-specific zeroed.

    ``mtime``, ownership and permissions otherwise vary per machine, which
    would make two packers' archives differ in bytes while describing the same
    tree -- and the archive digest is published, so it has to be reproducible.
    """
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.type = tarfile.REGTYPE
    return info


def write_archive(root: str | Path, dataset: str, archive_path: str | Path) -> dict[str, Any]:
    """Write one dataset's splits to a deterministic tar, and describe it."""
    root = Path(root)
    archive_path = Path(archive_path)
    members = sorted(_split_dirs(root, [dataset]), key=lambda item: item[1].name)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w", format=tarfile.GNU_FORMAT) as archive:
        for _, split_dir in members:
            for name in (*SPLIT_ROLE_FILES, MANIFEST_NAME):
                path = split_dir / name
                with open(path, "rb") as handle:
                    archive.addfile(
                        _archive_member(f"{dataset}/{split_dir.name}/{name}", path.stat().st_size),
                        handle,
                    )
    return {
        "file": archive_path.name,
        "sha256": file_sha256(archive_path),
        "bytes": archive_path.stat().st_size,
    }


def pack(
    root: str | Path,
    out_dir: str | Path,
    *,
    datasets: Iterable[str] | None = None,
    commit: str | None = None,
    protocol_version: str | None = None,
) -> dict[str, Any]:
    """Build the index for ``root`` and write one archive per dataset."""
    root = Path(root)
    out_dir = Path(out_dir)
    index = build_index(root, datasets=datasets, commit=commit, protocol_version=protocol_version)
    for dataset in index["datasets"]:
        index["datasets"][dataset]["archive"] = write_archive(
            root, dataset, out_dir / f"{dataset}-splits.tar"
        )
    return index


def dump_index(index: dict[str, Any], path: str | Path) -> None:
    """Write the index as sorted, human-diffable JSON."""
    Path(path).write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_index(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "datasets" not in payload:
        raise ValueError(f"{path} is not a split artifact index")
    if payload.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError(
            f"{path} is index schema {payload.get('schema_version')!r}; this tool reads "
            f"{INDEX_SCHEMA_VERSION!r}"
        )
    return payload


def _expected_files(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return entry["files"]


def _verify_split(split_dir: Path, entry: dict[str, Any], problems: list[dict[str, Any]]) -> int:
    """Check one split against its index entry; return the files it expected.

    Every file is attempted even after an earlier one fails.  An operator
    repairing a transfer needs the whole list of what is wrong, and stopping
    at the first bad file would mean re-running the check once per damaged
    product to discover the rest of them.
    """
    expected = _expected_files(entry)
    for name, recorded in sorted(expected.items()):
        path = split_dir / name
        if not path.is_file():
            problems.append({"kind": "missing_file", "path": str(path)})
            continue
        try:
            size = path.stat().st_size
            found = None if size != recorded["bytes"] else file_sha256(path)
        except OSError as exc:
            # Held open by a sync client, or denied by permissions: a product
            # that cannot be read is not one that passed.
            problems.append({"kind": "unreadable_file", "path": str(path), "detail": str(exc)})
            continue
        if found is None:
            problems.append(
                {
                    "kind": "size_mismatch",
                    "path": str(path),
                    "expected": recorded["bytes"],
                    "found": size,
                }
            )
            continue
        if found != recorded["sha256"]:
            problems.append(
                {
                    "kind": "digest_mismatch",
                    "path": str(path),
                    "expected": recorded["sha256"],
                    "found": found,
                }
            )
    extras = sorted(
        path.name for path in split_dir.iterdir() if path.is_file() and path.name not in expected
    )
    for name in extras:
        problems.append({"kind": "unexpected_file", "path": str(split_dir / name)})

    # Reading a damaged product must report it, not raise: a truncated ``.npz``
    # is the likeliest thing to arrive from a partial transfer, and a verifier
    # that dies on exactly the input it exists to catch is worse than useless.
    manifest_path = split_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return len(expected)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
        problems.append(
            {"kind": "unreadable_manifest", "path": str(manifest_path), "detail": str(exc)}
        )
        return len(expected)
    role_indices: dict[str, list[int]] = {}
    for name in SPLIT_ROLE_FILES:
        path = split_dir / name
        if not path.is_file():
            continue
        try:
            role_indices[name.removesuffix(".npz")] = read_indices(path)
        except ValueError as exc:
            # Recorded and the loop continues, so a split with two damaged
            # products names both of them rather than only the first.
            problems.append({"kind": "unreadable_indices", "path": str(path), "detail": str(exc)})
    if len(role_indices) == len(SPLIT_ROLE_FILES):
        recomputed = canonical_hash(role_indices)
        if recomputed != manifest.get("indices_sha256"):
            problems.append(
                {
                    "kind": "manifest_disagrees_with_data",
                    "path": str(manifest_path),
                    "expected": manifest.get("indices_sha256"),
                    "found": recomputed,
                }
            )
        if recomputed != entry.get("indices_sha256"):
            problems.append(
                {
                    "kind": "index_disagrees_with_data",
                    "path": str(manifest_path),
                    "expected": entry.get("indices_sha256"),
                    "found": recomputed,
                }
            )
        sizes = {role: len(indices) for role, indices in role_indices.items()}
        if sizes != manifest.get("role_sizes"):
            problems.append(
                {
                    "kind": "role_sizes_mismatch",
                    "path": str(manifest_path),
                    "expected": manifest.get("role_sizes"),
                    "found": sizes,
                }
            )
    return len(expected)


def _unexpected_splits(root: Path, indexed: set[tuple[str, str]]) -> list[dict[str, Any]]:
    """Split directories on disk that the index does not describe.

    The forward pass only walks the index, so a delivery carrying an extra
    split -- or an entire extra dataset -- would otherwise be reported clean.
    What the receiver ends up training on has to be what was indexed, and a
    copy that was renamed or duplicated in transit is not.
    """
    return [
        {"kind": "unexpected_split", "path": str(path)}
        for path in sorted(root.glob("*/*"))
        if path.is_dir()
        and path.name.startswith("split_")
        and (path.parent.name, path.name) not in indexed
    ]


def _verify_archive(
    dataset: str, entries: dict[str, Any], archive_dir: str | Path, problems: list[dict[str, Any]]
) -> None:
    """Check the dataset's archive, or say why it could not be checked.

    Asking for the archives to be verified and then silently skipping a
    dataset whose index carries no archive block would report success for the
    one artifact the procedure tells the receiver to trust.
    """
    archive_entry = entries.get("archive")
    if not isinstance(archive_entry, dict) or "file" not in archive_entry:
        expected_name = Path(archive_dir) / f"{dataset}-splits.tar"
        problems.append({"kind": "index_has_no_archive", "path": str(expected_name)})
        return
    archive_path = Path(archive_dir) / archive_entry["file"]
    if not archive_path.is_file():
        problems.append({"kind": "missing_archive", "path": str(archive_path)})
        return
    try:
        found = file_sha256(archive_path)
    except OSError as exc:
        problems.append({"kind": "unreadable_file", "path": str(archive_path), "detail": str(exc)})
        return
    if found != archive_entry["sha256"]:
        problems.append(
            {
                "kind": "archive_digest_mismatch",
                "path": str(archive_path),
                "expected": archive_entry["sha256"],
                "found": found,
            }
        )


def verify(
    root: str | Path,
    index: dict[str, Any],
    *,
    archive_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Check a tree against an index, collecting every problem it finds.

    Returns a report rather than raising: a receiver needs the full list of
    what is wrong with a delivery, not the first thing that went wrong with
    it.  Schema-level misuse (an index that is not an index, or one this tool
    does not read) still raises.
    """
    if not isinstance(index, dict) or not isinstance(index.get("datasets"), dict):
        raise ValueError("not a split artifact index")
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError(
            f"index schema_version is {index.get('schema_version')!r}; this tool reads "
            f"{INDEX_SCHEMA_VERSION!r}"
        )
    root = Path(root)
    problems: list[dict[str, Any]] = []
    splits = files = indexed = 0
    for dataset, entries in sorted(index["datasets"].items()):
        if not isinstance(entries, dict):
            raise ValueError(f"index entry for {dataset!r} is not a mapping")
        for split_name, entry in sorted(entries.items()):
            if split_name == "archive":
                continue
            if not isinstance(entry, dict) or "files" not in entry:
                raise ValueError(f"index entry for {dataset}/{split_name} is not a split")
            indexed += 1
            split_dir = root / dataset / split_name
            if not split_dir.is_dir():
                problems.append({"kind": "missing_split", "path": str(split_dir)})
                continue
            splits += 1
            files += _verify_split(split_dir, entry, problems)
        if archive_dir is not None:
            _verify_archive(dataset, entries, archive_dir, problems)
    # An index that describes nothing would otherwise verify a delivery of
    # nothing, which is the one report a receiver must never be handed.
    if not indexed:
        problems.append({"kind": "index_has_no_splits", "path": str(root)})
    problems.extend(_unexpected_splits(root, _indexed_splits(index)))
    return {
        "ok": not problems,
        "problems": problems,
        "checked": {"splits": splits, "files": files},
    }


def _indexed_splits(index: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (dataset, split_name)
        for dataset, entries in index["datasets"].items()
        if isinstance(entries, dict)
        for split_name in entries
        if split_name != "archive"
    }
