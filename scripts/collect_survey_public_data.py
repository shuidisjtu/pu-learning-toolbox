"""Collect missing public raw data without extracting archives.

ADNI itself needs approved access and is never fetched here.  The ``adni`` slot is
filled by a pinned mirror of the Kaggle 4-class Alzheimer MRI set instead (see
``ADNI_ARCHIVES``), which is not ADNI data.

Existing files are verified and never overwritten. Runtime receipts are local
data artifacts, not amendments to the frozen pilot protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from pu_toolbox.preprocessing.dataset_fetch import _collect

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
        path = (
            Path(__file__).resolve().parents[1]
            / "pu_toolbox/preprocessing/source_configs"
            / f"{dataset}.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        return [
            (
                name,
                artifact["url"],
                artifact["checksum"]["algorithm"],
                artifact["checksum"]["digest"],
            )
            for name, artifact in config["artifacts"].items()
        ]
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


def collect(dataset, raw_dir):
    entries = resources(dataset)
    pinned = dataset in PROJECT_PINNED
    mirror_url = (
        f"https://github.com/{ADNI_MIRROR_REPO}/tree/{ADNI_MIRROR_COMMIT}" if pinned else None
    )
    return _collect(
        dataset,
        raw_dir / dataset,
        entries,
        urlopen,
        checksum_source="project_pin" if pinned else "publisher",
        mirror_url=mirror_url,
    )


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
