#!/usr/bin/env python3
"""Prepare independent-support products from a reviewed raw reader or a frozen split.

Support size and seed are explicit proposals, never formal label-budget approval.
The survey policy is fixed: support_seed equals the split seed and is declared
before any label is inspected.  A mismatching support seed is refused rather
than silently recorded.  Default is dry-run: --write is required to materialize
a new directory.  No fit, download, GPU allocation, or mutation of existing
source/split products occurs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prepare_survey_splits import _load_dataset, load_source_records

from pu_toolbox.core.training_views import ROLES
from pu_toolbox.experiment.support_data import (
    RAW_SUPPORT_DATASETS,
    SPLIT_SUPPORT_DATASETS,
    SUPPORT_DATASETS,
    SUPPORT_SEED_POLICY,
    prepare_support_dataset,
    prepare_support_from_split,
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
    elif dataset == "cifar10":
        names = [f"cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
        names.append("cifar-10-batches-py/test_batch")
    else:
        raise ValueError("this dataset has no reviewed raw reader; use --split-dir")
    return {name: file_hash(root / name) for name in names}


def split_identity(split_dir):
    root = Path(split_dir)
    files = {f"{role}.npz": file_hash(root / f"{role}.npz") for role in ROLES}
    files["split_manifest.json"] = file_hash(root / "split_manifest.json")
    return files


def _check_output(output, source_root, write):
    output = output.resolve()
    source_root = source_root.resolve()
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError("output must not overlap source directory")
    if write and output.exists():
        raise ValueError("refusing to overwrite an existing output directory")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUPPORT_DATASETS)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-dir", type=Path)
    source.add_argument(
        "--split-dir",
        type=Path,
        help=(
            "frozen four-way split directory containing the four role NPZs and split_manifest.json"
        ),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--support-size", type=int, required=True)
    parser.add_argument(
        "--support-seed",
        type=int,
        default=None,
        help="must equal --seed; omitted means --seed",
    )
    parser.add_argument(
        "--group-policy",
        required=True,
        choices=["not_applicable", "unverifiable"],
    )
    parser.add_argument(
        "--owner-approval",
        default=None,
        help="required for ADNI's unverifiable subject-isolation policy",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.write and args.out_dir is None:
            raise ValueError("--write requires a new --out-dir")
        support_seed = args.seed if args.support_seed is None else args.support_seed
        if support_seed != args.seed:
            raise ValueError(
                f"{SUPPORT_SEED_POLICY}; got --seed {args.seed} and --support-seed {support_seed}"
            )
        if args.dataset == "adni":
            if args.group_policy != "unverifiable":
                raise ValueError(
                    "ADNI requires --group-policy unverifiable: the reviewed substitute "
                    "source has no subject IDs"
                )
            if not args.owner_approval:
                raise ValueError("ADNI unverifiable support requires --owner-approval")
        else:
            if args.group_policy != "not_applicable":
                raise ValueError("non-ADNI support requires --group-policy not_applicable")
            if args.owner_approval is not None:
                raise ValueError("--owner-approval is only recorded for the ADNI subject gap")

        if args.raw_dir is not None:
            if args.dataset not in RAW_SUPPORT_DATASETS:
                raise ValueError("this dataset has no reviewed raw reader; use --split-dir")
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
                support_seed=support_seed,
                source=(f"reviewed_raw_reader:{args.dataset}:{strict_canonical_hash(before)}"),
                group_policy=args.group_policy,
                download=download,
                X_test=loaded[2] if len(loaded) == 4 else None,
                y_test=loaded[3] if len(loaded) == 4 else None,
            )
            manifest["reader_files_sha256"] = before
            source_root = args.raw_dir
        else:
            if args.dataset not in SPLIT_SUPPORT_DATASETS:
                raise ValueError("this dataset must use its reviewed raw reader: --raw-dir")
            before = split_identity(args.split_dir)
            bundle, support, manifest = prepare_support_from_split(
                args.split_dir,
                dataset=args.dataset,
                seed=args.seed,
                support_size=args.support_size,
                support_seed=support_seed,
                source=(f"reviewed_split_product:{args.dataset}:{strict_canonical_hash(before)}"),
                group_policy=args.group_policy,
                owner_approval=args.owner_approval,
            )
            if split_identity(args.split_dir) != before:
                raise ValueError("frozen split changed while loading")
            manifest["split_files_sha256"] = before
            source_root = args.split_dir

        if args.out_dir is not None:
            _check_output(args.out_dir, source_root, args.write)
        if args.write:
            manifest = save_support_dataset(bundle, support, manifest, args.out_dir)
        print(
            json.dumps(
                {
                    "mode": "written" if args.write else "dry_run",
                    "dataset": args.dataset,
                    "source_mode": manifest["source_mode"],
                    "role_sizes": manifest["role_sizes"],
                    "population_prior": manifest["class_prior"]["population"],
                    "support_size": args.support_size,
                    "support_seed": support_seed,
                    "support_seed_policy": manifest["support_seed_policy"],
                    "support_positive_labels": int(support.labels.sum()),
                    "support_negative_labels": int(len(support.labels) - support.labels.sum()),
                    "group_policy": manifest["support_adapter_options"]["group_policy"],
                    "owner_approval": manifest["support_adapter_options"].get("owner_approval"),
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
