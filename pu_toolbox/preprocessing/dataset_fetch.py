"""Explicit, checksum-verified acquisition of configured dataset artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from .dataset_sources import _load_config, _local_path, _SourceConfig


def checksum(path, algorithm):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _collect(
    dataset,
    directory,
    entries,
    opener,
    *,
    checksum_source="publisher",
    mirror_url=None,
    source_version=None,
):
    directory = Path(directory)
    receipt_path = _local_path(directory, "provenance.json")
    targets = [_local_path(directory, name) for name, *_ in entries]
    resolved = [receipt_path.resolve(), *(target.resolve() for target in targets)]
    if len(resolved) != len(set(resolved)):
        raise ValueError("artifact paths must be distinct and not collide with the receipt")
    directory.mkdir(parents=True, exist_ok=True)
    previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    previous_files = {entry["filename"]: entry for entry in previous.get("files", [])}
    records = []
    for (name, url, algorithm, expected), target in zip(entries, targets, strict=True):
        publisher_expected = expected
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if expected is None:
                expected = previous_files.get(name, {}).get("sha256")
                if expected is None:
                    raise ValueError(f"unverified existing file: {target}")
            if checksum(target, algorithm) != expected:
                raise ValueError(f"checksum mismatch, existing file preserved: {target}")
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=".download-", dir=directory)
            staging = Path(temporary)
            try:
                with os.fdopen(descriptor, "wb") as output, opener(url, timeout=45) as response:
                    total = 0
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > 256 * 1024 * 1024:
                            raise ValueError("public archive exceeded the 256 MiB safety limit")
                        output.write(block)
                if expected is not None and checksum(staging, algorithm) != expected:
                    label = "publisher " if checksum_source == "publisher" else ""
                    raise ValueError(f"{label}checksum mismatch: {name}")
                # Exclusive destination creation prevents overwriting concurrent work.
                os.link(staging, target)
            finally:
                staging.unlink(missing_ok=True)
        pinned = checksum_source == "project_pin"
        publisher = publisher_expected if checksum_source == "publisher" else None
        record = {
            "filename": name,
            "source_url": url,
            "bytes": target.stat().st_size,
            "sha256": checksum(target, "sha256"),
            "publisher_checksum_algorithm": algorithm if publisher else None,
            "publisher_checksum": publisher,
            "publisher_checksum_verified": publisher is not None,
        }
        if pinned:
            record["checksum_source"] = "project_pin"
            record["pinned_sha256"] = publisher_expected
        elif checksum_source == "config":
            record["checksum_source"] = "config"
            record["config_checksum_algorithm"] = algorithm
            record["config_checksum"] = publisher_expected
        records.append(record)
        print(f"verified {dataset}/{name} ({target.stat().st_size} bytes)", flush=True)
    payload = {
        "dataset": dataset,
        # Kept from the first collection: a verified reuse changes nothing, and a
        # receipt that moved with every call could not be compared with itself.
        "collected_at": previous.get("collected_at") or datetime.now(timezone.utc).isoformat(),
        "files": records,
        "status": "raw_collected_not_formal_splits",
        "extracted": False,
        "license_review": "see docs/research/pu_survey/data/README.md",
    }
    if checksum_source == "config":
        payload.pop("license_review")
    if source_version is not None:
        payload["source_version"] = source_version
    if mirror_url is not None:
        payload["mirror_url"] = mirror_url
    descriptor, temporary = tempfile.mkstemp(prefix=".receipt-", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(payload, output, indent=2)
            output.write("\n")
        os.replace(temporary, receipt_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return payload


def fetch_dataset(config: _SourceConfig, root: str | Path) -> dict:
    """Explicitly acquire configured artifacts into ``root`` without overwriting data.

    Check the configured digest before publishing each file. The provenance
    receipt records the source version and actual hashes, without treating a
    caller-supplied checksum as proof of publisher authentication.
    """
    spec = _load_config(config)
    artifacts = spec.get("artifacts", {})
    if not artifacts:
        raise ValueError("artifacts must be a nonempty mapping")
    entries = []
    for filename, item in artifacts.items():
        if not isinstance(item, Mapping):
            raise ValueError("invalid artifact entry")
        url = item.get("url")
        if not isinstance(url, str):
            raise ValueError("artifact URL must be HTTPS")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("artifact URL must be HTTPS")
        expected = item.get("checksum")
        if not isinstance(expected, Mapping):
            raise ValueError("artifact checksum is required")
        algorithm, digest = expected.get("algorithm"), expected.get("digest")
        length = {"md5": 32, "sha256": 64}.get(algorithm) if isinstance(algorithm, str) else None
        if (
            length is None
            or not isinstance(digest, str)
            or not re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", digest)
        ):
            raise ValueError("artifact checksum must be a valid MD5 or SHA256 digest")
        entries.append((filename, url, algorithm, digest.lower()))
    return _collect(
        spec["source_id"],
        Path(root),
        entries,
        urlopen,
        checksum_source="config",
        source_version=spec["version"],
    )
