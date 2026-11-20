"""Shared local directory and archive readers for raw text documents."""

from __future__ import annotations

import tarfile
from pathlib import Path, PurePosixPath

import numpy as np


def _read_text_documents(root: Path, part, options) -> tuple[np.ndarray, np.ndarray]:
    from .dataset_sources import _check_keys, _local_path, _require_file

    _check_keys(options, {"classes", "encoding"}, {"decode_errors", "suffix"}, "text options")
    classes = options["classes"]
    if (
        not isinstance(classes, list)
        or not classes
        or any(
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or "/" in name
            or "\\" in name
            for name in classes
        )
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("text classes must be a nonempty list of distinct directory names")
    encoding = options["encoding"]
    errors = options.get("decode_errors", "strict")
    suffix = options.get("suffix", "")
    if not isinstance(encoding, str) or not encoding:
        raise ValueError("text encoding must be a nonempty string")
    if not isinstance(errors, str) or not errors:
        raise ValueError("text decode_errors must be a nonempty string")
    if not isinstance(suffix, str):
        raise ValueError("text suffix must be a string")

    texts, labels = [], []
    if "directory" in part:
        _check_keys(part, {"directory"}, set(), "text directory partition")
        directory = _local_path(root, part["directory"])
        entries = list(directory.iterdir())
        if {entry.name for entry in entries} != set(classes) or any(
            not entry.is_dir() for entry in entries
        ):
            raise ValueError(f"{directory}: expected configured categories")
        for category in classes:
            category_dir = _local_path(directory, category)
            members = sorted(category_dir.iterdir(), key=lambda path: path.name)
            if not members:
                raise ValueError(f"{directory}: empty category {category}")
            for member in members:
                if not member.is_file() or member.is_symlink():
                    raise ValueError(f"{directory}: not a document file: {member}")
                if member.name.endswith(suffix):
                    texts.append(member.read_bytes().decode(encoding, errors=errors))
                    labels.append(category)
    else:
        _check_keys(part, {"archive", "member_root"}, set(), "text archive partition")
        path = _local_path(root, part["archive"])
        _require_file(path)
        member_root = part["member_root"]
        if (
            not isinstance(member_root, str)
            or not member_root
            or "\\" in member_root
            or PurePosixPath(member_root).is_absolute()
            or any(piece in (".", "..") for piece in member_root.split("/"))
        ):
            raise ValueError("invalid archive member_root")
        with tarfile.open(path, "r:*") as archive:
            members = archive.getmembers()
            for category in classes:
                prefix = f"{member_root.rstrip('/')}/{category}/"
                for member in members:
                    if (
                        member.isfile()
                        and member.name.startswith(prefix)
                        and member.name.endswith(suffix)
                    ):
                        stream = archive.extractfile(member)
                        if stream is None:
                            raise ValueError(f"cannot read archive member: {member.name}")
                        with stream:
                            texts.append(stream.read().decode(encoding, errors=errors))
                        labels.append(category)
    return np.asarray(texts, dtype=object), np.asarray(labels, dtype=object)
