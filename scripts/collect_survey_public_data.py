"""Collect missing public raw data without extracting archives or accepting ADNI terms.

Existing files are verified and never overwritten. Runtime receipts are local
data artifacts, not amendments to the frozen pilot protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen


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
    raise ValueError(f"unsupported public dataset {dataset!r}; ADNI requires approved access")


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
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
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
        records.append(
            {
                "filename": name,
                "source_url": url,
                "bytes": target.stat().st_size,
                "sha256": checksum(target, "sha256"),
                "publisher_checksum_algorithm": algorithm if publisher_expected else None,
                "publisher_checksum": publisher_expected,
                "publisher_checksum_verified": publisher_expected is not None,
            }
        )
        print(f"verified {dataset}/{name} ({target.stat().st_size} bytes)", flush=True)
    payload = {
        "dataset": dataset,
        "files": records,
        "status": "raw_collected_not_formal_splits",
        "extracted": False,
        "license_review": "see collaborator_followup_20261004.md",
    }
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
        choices=["mnist", "fashion_mnist", "twenty_newsgroups", "connect_4"],
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    for dataset in args.datasets:
        collect(dataset, args.raw_dir)


if __name__ == "__main__":
    main()
