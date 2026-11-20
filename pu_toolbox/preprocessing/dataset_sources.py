"""Configuration-driven reading of local raw dataset partitions."""

from __future__ import annotations

import gzip
import json
import re
import struct
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

from ..utils.serialization import json_scalars
from .pu_construction import SourcePartition

__all__ = ["read_dataset"]

_SourceConfig = Mapping[str, Any] | str | Path
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MAX_IDX_BYTES = 256 * 1024 * 1024


def _mapping(value, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _check_keys(fields: Mapping[str, Any], required: set[str], optional: set[str], name: str):
    missing = required - fields.keys()
    unknown = fields.keys() - required - optional
    if missing or unknown:
        raise ValueError(
            f"invalid {name} fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _token(value, name: str) -> None:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise ValueError(f"{name} must use letters, digits, dots, dashes or underscores")


def _load_config(config: _SourceConfig) -> Mapping[str, Any]:
    if isinstance(config, (str, Path)):
        with Path(config).open(encoding="utf-8") as handle:
            config = json.load(handle)
    config = _mapping(config, "dataset config")
    _check_keys(
        config,
        {"source_id", "version", "format", "partitions"},
        {"options", "artifacts"},
        "dataset config",
    )
    for name in ("source_id", "version"):
        _token(config[name], name)
    codec = config["format"]
    if not isinstance(codec, str) or codec not in _READERS:
        raise ValueError(f"unknown dataset format {codec!r}; expected one of {sorted(_READERS)}")
    parts = _mapping(config["partitions"], "partitions")
    if not parts:
        raise ValueError("partitions must declare at least one source partition")
    for name, part in parts.items():
        _token(name, "partition name")
        _mapping(part, f"partition {name!r}")
    _mapping(config.get("options", {}), "format options")
    _mapping(config.get("artifacts", {}), "artifacts")
    return config


def _local_path(root: Path, filename: str) -> Path:
    if not isinstance(filename, str) or not filename or any(c in filename for c in "\\:\x00"):
        raise ValueError(f"invalid relative dataset path: {filename!r}")
    relative = PurePosixPath(filename)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(
            part == ".." or PureWindowsPath(part).is_reserved() or part.endswith((" ", "."))
            for part in relative.parts
        )
    ):
        raise ValueError(f"invalid relative dataset path: {filename!r}")
    base = root.resolve()
    path = base.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(base):
        raise ValueError(f"dataset path escapes its root: {filename!r}")
    return path


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing local dataset file: {path}. Supply it locally or call fetch_dataset "
            "explicitly; reading does not download files."
        )


def _image_shape(options: Mapping[str, Any]) -> tuple[int, int] | None:
    if "image_shape" not in options:
        return None
    shape = options["image_shape"]
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or not all(type(value) is int and value > 0 for value in shape)
    ):
        raise ValueError("image_shape must contain positive height and width")
    return shape[0], shape[1]


def _read_idx(path: Path, *, shape: tuple[int, ...], max_rows: int) -> np.ndarray:
    is_image = bool(shape)
    header_format = ">4I" if is_image else ">2I"
    header_size = struct.calcsize(header_format)
    open_stream = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with open_stream(path, "rb") as stream:
            header = stream.read(header_size)
            if len(header) != header_size:
                raise ValueError(f"{path.name}: incomplete IDX header")
            magic, count, *dimensions = struct.unpack(header_format, header)
            if magic != (2051 if is_image else 2049):
                raise ValueError(f"{path.name}: unexpected IDX magic number {magic}")
            if tuple(dimensions) != shape:
                raise ValueError(
                    f"{path.name}: expected IDX shape {shape}, got {tuple(dimensions)}"
                )
            if not 1 <= count <= max_rows:
                raise ValueError(f"{path.name}: IDX row count must be between 1 and {max_rows}")
            size = count * (shape[0] * shape[1] if is_image else 1)
            if size > _MAX_IDX_BYTES:
                raise ValueError(f"{path.name}: IDX payload exceeds the 256 MiB limit")
            payload = stream.read(size + 1)
    except (gzip.BadGzipFile, EOFError) as error:
        raise ValueError(f"{path.name}: invalid gzip IDX payload") from error
    if len(payload) != size:
        raise ValueError(f"{path.name}: IDX payload length does not match its header")
    return np.frombuffer(payload, dtype=np.uint8).reshape(count, *shape)


def _read_idx_images(root: Path, part, options) -> tuple[np.ndarray, np.ndarray]:
    _check_keys(part, {"features", "labels", "max_rows"}, set(), "IDX partition")
    _check_keys(options, {"image_shape"}, {"label_values"}, "IDX options")
    shape = _image_shape(options)
    max_rows = part["max_rows"]
    if type(max_rows) is not int or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    images = _local_path(root, part["features"])
    labels = _local_path(root, part["labels"])
    _require_file(images)
    _require_file(labels)
    return _read_idx(images, shape=shape, max_rows=max_rows), _read_idx(
        labels, shape=(), max_rows=max_rows
    )


def _read_npz_arrays(root: Path, part, options) -> tuple[np.ndarray, np.ndarray]:
    _check_keys(part, {"file", "features_key", "labels_key"}, set(), "NPZ partition")
    _check_keys(options, set(), {"image_shape", "archive_keys", "label_values"}, "NPZ options")
    path = _local_path(root, part["file"])
    keys = (part["features_key"], part["labels_key"])
    if not all(isinstance(key, str) and key for key in keys) or keys[0] == keys[1]:
        raise ValueError("features_key and labels_key must be distinct nonempty strings")
    expected = options.get("archive_keys")
    if "archive_keys" in options and (
        not isinstance(expected, (list, tuple))
        or not all(isinstance(key, str) and key for key in expected)
        or len(expected) != len(set(expected))
        or not set(keys) <= set(expected)
    ):
        raise ValueError("archive_keys must be distinct strings including the requested keys")
    _require_file(path)
    archive = np.load(path, allow_pickle=False)
    if not hasattr(archive, "files"):
        raise ValueError(f"{path.name}: expected an NPZ array archive")
    with archive:
        if expected is not None and (
            len(archive.files) != len(expected) or set(archive.files) != set(expected)
        ):
            raise ValueError(f"{path.name}: expected exactly the keys {expected}")
        if not set(keys) <= set(archive.files):
            raise ValueError(f"{path.name}: missing requested keys {keys}")
        return archive[keys[0]], archive[keys[1]]


def _validate_arrays(features, labels, options, context: str) -> None:
    shape = _image_shape(options)
    if shape is not None:
        if features.dtype != np.uint8:
            raise ValueError(f"{context}: images must be uint8")
        if features.ndim != 3 or features.shape[1:] != shape:
            raise ValueError(f"{context}: images must have shape (N,{shape[0]},{shape[1]})")
    if features.ndim < 2 or not all(features.shape) or features.dtype.kind not in "biuf":
        raise ValueError(f"{context}: features must be a non-empty real numeric sample array")
    if features.dtype.kind == "f" and not np.isfinite(features).all():
        raise ValueError(f"{context}: features must be finite")
    if labels.ndim != 1:
        raise ValueError(f"{context}: labels must be one-dimensional")
    if shape is not None and not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{context}: labels must have integer dtype")
    if len(labels) != len(features):
        raise ValueError(f"{context}: feature/label length mismatch")
    json_scalars(labels, name=f"{context}: labels")
    if "label_values" in options:
        values = options["label_values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("label_values must list the permitted raw classes")
        allowed = json_scalars(np.asarray(values, dtype=object), name="label_values")
        if not np.isin(labels, np.asarray(allowed, dtype=object)).all():
            raise ValueError(f"{context}: labels must be in configured label_values {allowed}")


_READERS = {"idx_images": _read_idx_images, "npz_arrays": _read_npz_arrays}


def read_dataset(config: _SourceConfig, root: str | Path, *, partition: str) -> SourcePartition:
    """Read only the configured source partition; never acquire, map or split it.

    ``config`` is a mapping or a JSON file. Paths in it are relative to the
    explicit dataset directory ``root``. ``source_id@version:partition:row``
    records the declared source's row positions; local reading does not
    independently verify that two copies are identical.
    """
    spec = _load_config(config)
    _token(partition, "partition")
    if partition not in spec["partitions"]:
        raise ValueError(f"source partition {partition!r} is not declared in the config")
    options = spec.get("options", {})
    features, labels = _READERS[spec["format"]](Path(root), spec["partitions"][partition], options)
    source_id = f"{spec['source_id']}@{spec['version']}"
    _validate_arrays(features, labels, options, f"{source_id} {partition}")
    if "image_shape" in options:
        features = features[:, None]
    sample_ids = np.array([f"{source_id}:{partition}:{row}" for row in range(len(labels))])
    return SourcePartition(features, labels, source_id, partition, sample_ids)
