"""Collect missing public raw data without extracting archives.

ADNI itself needs approved access and is never fetched here.  The ``adni`` slot is
filled by a pinned mirror of the Kaggle 4-class Alzheimer MRI set instead (see
``ADNI_ARCHIVES``), which is not ADNI data.

Existing files are verified and never overwritten. Runtime receipts are local
data artifacts, not amendments to the frozen pilot protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

#: Mirror of the Kaggle "Alzheimer's Dataset (4 class of Images)" that stands in for
#: ADNI.  It is pinned to one commit and to per-archive digests because the mirror is
#: a personal repository that states no licence and could be changed or removed; the
#: digests are this project's pin, not a checksum its publisher states.
ADNI_MIRROR_REPO = "Jahnabi19/AD-Dataset"
ADNI_MIRROR_COMMIT = "d90729118954bbedfe481dea8a53b4f3ee5a21b2"
ADNI_ARCHIVES = (
    ("NonDemented.zip", "989b9fb78a05c8951744bed3528b38251cb2317c71795764ff08060ffc7ae605"),
    ("VeryMildDemented.zip", "963270e02aeb6f5ceeb91d0ab13282d9c1b199e0301f8277c515750ab1076d65"),
    ("ModerateDemented.zip", "5f8eb289e2ee8d362df07e312f127b9cefeaf7791536bf0ae018fde5a94ff2b4"),
    ("MildDemented.zip", "1dc707628874ebb7e25abfa81cf655f5dbeb845a43ab1dbbd955fa1bebdf36c6"),
)
#: Datasets whose expected digest is this project's pin rather than a publisher's.
PROJECT_PINNED = frozenset({"adni"})


def resources(dataset):
    if dataset in {"mnist", "fashion_mnist"}:
        from torchvision.datasets import MNIST, FashionMNIST

        cls = MNIST if dataset == "mnist" else FashionMNIST
        base = (
            "https://ossci-datasets.s3.amazonaws.com/mnist/"
            if dataset == "mnist"
            else "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
        )
        return [(name, base + name, "md5", checksum) for name, checksum in cls.resources]
    if dataset == "twenty_newsgroups":
        from sklearn.datasets._twenty_newsgroups import ARCHIVE

        return [(ARCHIVE.filename, ARCHIVE.url, "sha256", ARCHIVE.checksum)]
    if dataset == "connect_4":
        return [
            (
                "connect+4.zip",
                "https://archive.ics.uci.edu/static/public/26/connect+4.zip",
                "sha256",
                None,
            )
        ]
    if dataset == "adni":
        base = f"https://raw.githubusercontent.com/{ADNI_MIRROR_REPO}/{ADNI_MIRROR_COMMIT}/"
        return [(name, base + name, "sha256", digest) for name, digest in ADNI_ARCHIVES]
    raise ValueError(f"unsupported public dataset {dataset!r}")


def checksum(path, algorithm):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def collect(dataset, raw_dir):
    entries = resources(dataset)
    directory = raw_dir / dataset
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "provenance.json"
    previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    previous_files = {entry["filename"]: entry for entry in previous.get("files", [])}
    records = []
    for name, url, algorithm, expected in entries:
        publisher_expected = expected
        target = directory / name
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
                with os.fdopen(descriptor, "wb") as output, urlopen(url, timeout=45) as response:
                    total = 0
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > 256 * 1024 * 1024:
                            raise ValueError("public archive exceeded the 256 MiB safety limit")
                        output.write(block)
                if expected is not None and checksum(staging, algorithm) != expected:
                    raise ValueError(f"publisher checksum mismatch: {name}")
                # Exclusive destination creation prevents overwriting concurrent work.
                os.link(staging, target)
            finally:
                staging.unlink(missing_ok=True)
        pinned = dataset in PROJECT_PINNED
        publisher = None if pinned else publisher_expected
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
        "license_review": "see collaborator_followup_20261004.md",
    }
    if dataset in PROJECT_PINNED:
        payload["mirror_url"] = f"https://github.com/{ADNI_MIRROR_REPO}/tree/{ADNI_MIRROR_COMMIT}"
    descriptor, temporary = tempfile.mkstemp(prefix=".receipt-", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(payload, output, indent=2)
            output.write("\n")
        os.replace(temporary, receipt_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mnist", "fashion_mnist", "twenty_newsgroups", "connect_4"],
        choices=["mnist", "fashion_mnist", "twenty_newsgroups", "connect_4", "adni"],
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    for dataset in args.datasets:
        collect(dataset, args.raw_dir)


if __name__ == "__main__":
    main()
