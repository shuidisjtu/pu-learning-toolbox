#!/usr/bin/env python3
"""Prepare independent-support products from reviewed raw readers, not old splits.

Support size and seed are explicit proposals, never formal label-budget approval.
Default is dry-run: --write is required to materialize a new directory. No fit,
download, GPU allocation, or mutation of existing source/split products occurs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prepare_survey_splits import _load_dataset, load_source_records

from pu_toolbox.experiment.support_data import (
    SUPPORT_DATASETS,
    prepare_support_dataset,
    save_support_dataset,
)
from pu_toolbox.utils.serialization import file_hash, strict_canonical_hash


def reader_identity(raw_dir, dataset):
    root = Path(raw_dir) / dataset
    if dataset in ("mnist", "fashion_mnist"):
        names = ["raw_arrays.npz"]
    elif dataset == "spambase":
        names = ["spambase.data"]
    elif dataset == "connect_4":
        names = ["connect-4.data"]
    else:
        names = [f"cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
        names.append("cifar-10-batches-py/test_batch")
    return {name: file_hash(root / name) for name in names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUPPORT_DATASETS)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--support-size", type=int, required=True)
    parser.add_argument("--support-seed", type=int, required=True)
    parser.add_argument("--group-policy", required=True, choices=["not_applicable"])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.write and args.out_dir is None:
            raise ValueError("--write requires a new --out-dir")
        if args.out_dir is not None:
            output = args.out_dir.resolve()
            raw = args.raw_dir.resolve()
            if output == raw or raw in output.parents or output in raw.parents:
                raise ValueError("output must not overlap raw source directory")
            if args.write and output.exists():
                raise ValueError("refusing to overwrite an existing output directory")
        before = reader_identity(args.raw_dir, args.dataset)
        loaded = _load_dataset(args.dataset, args.raw_dir)
        if reader_identity(args.raw_dir, args.dataset) != before:
            raise ValueError("raw source changed while loading")
        download, _ = load_source_records(args.raw_dir, args.dataset)
        bundle, support, manifest = prepare_support_dataset(
            *loaded[:2],
            dataset=args.dataset,
            seed=args.seed,
            support_size=args.support_size,
            support_seed=args.support_seed,
            source=f"reviewed_raw_reader:{args.dataset}:{strict_canonical_hash(before)}",
            group_policy=args.group_policy,
            download=download,
            X_test=loaded[2] if len(loaded) == 4 else None,
            y_test=loaded[3] if len(loaded) == 4 else None,
        )
        manifest["reader_files_sha256"] = before
        if args.write:
            manifest = save_support_dataset(bundle, support, manifest, args.out_dir)
        print(
            json.dumps(
                {
                    "mode": "written" if args.write else "dry_run",
                    "dataset": args.dataset,
                    "role_sizes": manifest["role_sizes"],
                    "population_prior": manifest["class_prior"]["population"],
                    "support_positive_labels": int(support.labels.sum()),
                    "support_negative_labels": int(len(support.labels) - support.labels.sum()),
                    "formal_admission": False,
                    "budget_approval": "not_granted_by_preparer",
                    "receipt_sha256": manifest.get("receipt_sha256"),
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
