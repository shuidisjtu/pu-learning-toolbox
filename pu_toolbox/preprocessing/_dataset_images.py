"""Read local image datasets without changing their raw labels or partitions."""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _read_cifar_batches(
    root: Path, part: Mapping, options: Mapping
) -> tuple[np.ndarray, np.ndarray]:
    from .dataset_sources import _check_keys

    _check_keys(part, {"train"}, set(), "CIFAR partition")
    _check_keys(options, {"variant"}, set(), "CIFAR options")
    if type(part["train"]) is not bool:
        raise ValueError("CIFAR train must be a boolean")
    if options["variant"] != "cifar10":
        raise ValueError("unsupported CIFAR variant; expected 'cifar10'")
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=str(root), train=part["train"], download=False)
    return np.asarray(dataset.data).transpose(0, 3, 1, 2), np.asarray(dataset.targets)


def _read_archive_images(
    root: Path, part: Mapping, options: Mapping
) -> tuple[np.ndarray, np.ndarray]:
    from .dataset_sources import _check_keys, _local_path, _require_file

    _check_keys(part, {"archives", "expected_counts"}, set(), "image archive partition")
    _check_keys(options, {"classes", "suffixes"}, set(), "image archive options")
    classes = options["classes"]
    suffixes = options["suffixes"]
    if (
        not isinstance(classes, (list, tuple))
        or not classes
        or any(
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or name.startswith(".")
            for name in classes
        )
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("classes must contain distinct nonempty directory names")
    if (
        not isinstance(suffixes, (list, tuple))
        or not suffixes
        or any(
            not isinstance(suffix, str) or not suffix.startswith(".") or len(suffix) < 2
            for suffix in suffixes
        )
    ):
        raise ValueError("suffixes must contain nonempty filename extensions")
    archives = part["archives"]
    counts = part["expected_counts"]
    if not isinstance(archives, Mapping) or not isinstance(counts, Mapping):
        raise ValueError("archives and expected_counts must be objects keyed by class")
    if set(archives) != set(classes) or set(counts) != set(classes):
        raise ValueError("archives and expected_counts must match the declared classes")
    if any(type(counts[name]) is not int or counts[name] < 0 for name in classes):
        raise ValueError("expected_counts must contain nonnegative integers")
    suffix_tuple = tuple(suffix.lower() for suffix in suffixes)
    images = []
    labels = []
    for name in classes:
        path = _local_path(root, archives[name])
        _require_file(path)
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                info.filename
                for info in archive.infolist()
                if not info.is_dir()
                and info.filename.startswith(f"{name}/")
                and not info.filename.rsplit("/", 1)[-1].startswith(".")
                and info.filename.lower().endswith(suffix_tuple)
            )
            if len(members) != counts[name]:
                raise ValueError(f"{name}: found {len(members)} images, expected {counts[name]}")
            images.extend(archive.read(member) for member in members)
            labels.extend([name] * len(members))
    result = np.empty(len(images), dtype=object)
    result[:] = images
    return result, np.asarray(labels, dtype=object)
