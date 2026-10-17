# ruff: noqa: N803
"""Independent clean-support products; never overwrite frozen four-way splits.

The array API expects original, unprocessed source features. The repository
CLI uses the reviewed raw readers rather than accepting processed split NPZs.
Subject/group-sensitive and text inputs are deliberately not admitted here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.training_views import ROLES
from ..utils.serialization import array_hash, file_hash, strict_canonical_hash
from .bundle import DatasetBundle, DatasetPart, validate_bundle
from .datasets import prepare_survey_dataset
from .image import fit_survey_image_preprocessing

SUPPORT_DATASETS = ("spambase", "connect_4", "mnist", "fashion_mnist", "cifar10")
_PARTS = (*ROLES, "support")
_SCHEMA = "independent-clean-support-v1"
_RULE = "uniform_without_replacement_from_base_train_no_label_screening_no_retry"


def _part_identity(part):
    return {
        "rows": len(part.X),
        "indices": part.indices.tolist(),
        "X_sha256": array_hash(part.X),
        "y_sha256": array_hash(part.labels),
        "indices_sha256": array_hash(part.indices),
    }


def prepare_support_dataset(
    features,
    labels,
    *,
    dataset,
    seed,
    support_size,
    support_seed,
    source,
    group_policy,
    X_test=None,
    y_test=None,
    download=None,
):
    """Derive a five-role exploratory product from unprocessed source arrays.

    ``support_size`` and ``support_seed`` are mandatory planning decisions,
    not method defaults or formal budget approvals. Uniform sampling inspects
    no support labels; a single-class draw fails without resampling. Existing
    validation/test membership is retained; preprocessing is refitted only on
    the remaining train. Population pi remains the original generation prior.

    Returns ``(bundle, support, manifest)``. Only the five listed independent-
    sample datasets are supported, with explicit ``group_policy='not_applicable'``.
    ADNI requires separate subject/group evidence and is rejected.
    """
    if dataset not in SUPPORT_DATASETS:
        raise ValueError("dataset needs an independently reviewed support/group/text policy")
    if group_policy != "not_applicable":
        raise ValueError("this preparer supports explicit not_applicable group policy only")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("raw source identity must be recorded")
    for name, value in (("seed", seed), ("support_seed", support_seed)):
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError(f"{name} must be an unsigned 32-bit integer")
    if type(support_size) is not int or support_size < 2:
        raise ValueError("support_size must be an explicit integer of at least two")
    features = np.asarray(features)
    image = dataset not in ("spambase", "connect_4")
    for values in (features, X_test):
        if values is None:
            continue
        values = np.asarray(values)
        if image:
            if values.ndim != 4 or values.dtype != np.uint8:
                raise ValueError("image source must be original uint8 NCHW, not processed splits")
        elif values.ndim != 2 or values.shape[1] == 0 or values.dtype.kind not in "fiu":
            raise ValueError("tabular source must be a real numeric feature matrix")
        if not np.isfinite(values).all():
            raise ValueError("raw source features must be finite")
    original, manifest = prepare_survey_dataset(
        features,
        labels,
        dataset=dataset,
        seed=seed,
        X_test=X_test,
        y_test=y_test,
        download=download,
    )
    train = original.train
    if support_size > len(train.X) - 2:
        raise ValueError("support_size must leave at least two training rows")
    positions = np.sort(
        np.random.RandomState(support_seed).choice(len(train.X), support_size, replace=False)
    )
    keep = np.ones(len(train.X), dtype=bool)
    keep[positions] = False
    if any(set(np.unique(train.labels[mask])) != {0, 1} for mask in (positions, keep)):
        raise ValueError("single-class support or remaining train draw; no label-based retry")
    remaining = DatasetPart(train.X[keep], train.labels[keep], "clean", train.indices[keep])
    support = DatasetPart(
        train.X[positions], train.labels[positions], "clean", train.indices[positions], False
    )
    bundle = DatasetBundle(remaining, original.pu_val, original.clean_val, original.test)
    parent_digest = strict_canonical_hash(manifest)
    if image:
        _, spec = fit_survey_image_preprocessing(
            remaining.X, dataset=dataset, train_augmentation="none"
        )
        preprocessing = spec.to_manifest()
        preprocessing["stored_features"] = "raw_uint8_normalization_not_applied"
    else:
        mean = remaining.X.mean(axis=0, dtype=np.float64)
        std = remaining.X.std(axis=0, dtype=np.float64)
        std = np.where(std < 1e-12, 1.0, std)
        preprocessing = {
            "kind": "z-score",
            "fitted": "remaining_train_only",
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "raw_train_sha256": array_hash(remaining.X),
        }

        def transform(part):
            return DatasetPart(
                ((part.X - mean) / std).astype(np.float32),
                part.labels,
                "clean",
                part.indices,
                part.for_selection,
            )

        bundle = DatasetBundle(*(transform(getattr(bundle, name)) for name in ROLES))
        support = transform(support)
    parts = {name: getattr(bundle, name) for name in ROLES} | {"support": support}
    manifest.update(
        schema_version=_SCHEMA,
        formal_admission=False,
        comparison_group="extra_clean_support",
        budget_approval="not_granted_by_preparer",
        split_policy="base_four_way_then_uniform_support_from_train",
        source=source,
        preprocessing=preprocessing,
        parts={name: _part_identity(part) for name, part in parts.items()},
        indices={name: part.indices.tolist() for name, part in parts.items()},
        role_sizes={name: len(part.X) for name, part in parts.items()},
        role_positive_rates={name: float(part.labels.mean()) for name, part in parts.items()},
        lineage={
            "base_split_manifest_sha256": parent_digest,
            "original_train_indices": train.indices.tolist(),
            "original_train_count": len(train.X),
            "raw_source_X_sha256": array_hash(features),
            "raw_source_y_sha256": array_hash(np.asarray(labels)),
        },
        support_source_audit={
            "support_origin": "carved_from_original_pool",
            "original_role_counts": {
                **{name: len(getattr(original, name).X) for name in ROLES},
                "prior_estimation": 0,
            },
            "support_source_verified": False,
            "support_source_receipt": source,
        },
        support_adapter_options={
            "dataset": dataset,
            "source": source,
            "prior_estimation_indices": [],
            "original_train_count": len(train.X),
            "support_removed_from_train": True,
            "group_policy": group_policy,
            "preparation_seed": support_seed,
            "sampling_rule": _RULE,
        },
    )
    manifest["indices_sha256"] = strict_canonical_hash(manifest["indices"])
    manifest["class_prior"]["train"] = float(bundle.train.labels.mean())
    _validate_product(bundle, support, manifest)
    return bundle, support, manifest


def _validate_product(bundle, support, manifest):
    if manifest.get("schema_version") != _SCHEMA or manifest.get("formal_admission") is not False:
        raise ValueError("not an exploratory independent-support product")
    if manifest.get("dataset") not in SUPPORT_DATASETS:
        raise ValueError("unsupported support dataset")
    if (
        manifest.get("comparison_group") != "extra_clean_support"
        or manifest.get("budget_approval") != "not_granted_by_preparer"
        or any(set(manifest[key]) != set(_PARTS) for key in ("parts", "indices", "role_sizes"))
    ):
        raise ValueError("support product comparison/budget/role declaration is inconsistent")
    validate_bundle(bundle)
    parts = {name: getattr(bundle, name) for name in ROLES} | {"support": support}
    seen = set()
    for name, part in parts.items():
        if (
            part.X.ndim not in (2, 4)
            or part.X.dtype.kind not in "fiu"
            or not np.isfinite(part.X).all()
            or part.labels.shape != (len(part.X),)
            or part.indices.shape != (len(part.X),)
            or part.indices.dtype.kind not in "iu"
            or set(np.unique(part.labels)) != {0, 1}
            or part.view != "clean"
            or (name in ("support", "test") and part.for_selection)
        ):
            raise ValueError(f"{name} violates support product array/label/role contract")
        if part.X.shape[1:] != bundle.train.X.shape[1:]:
            raise ValueError("support product feature shapes differ")
        ids = part.indices.tolist()
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("support product role identities overlap")
        seen.update(ids)
        if (
            _part_identity(part) != manifest["parts"][name]
            or ids != manifest["indices"][name]
            or len(ids) != manifest["role_sizes"][name]
        ):
            raise ValueError(f"{name} arrays disagree with support receipt")
    original = manifest["lineage"]["original_train_indices"]
    if set(original) != set(bundle.train.indices) | set(support.indices):
        raise ValueError("remaining train and support do not partition the original train")
    options = manifest["support_adapter_options"]
    if (
        options["dataset"] != manifest["dataset"]
        or options["source"] != manifest["source"]
        or options["support_removed_from_train"] is not True
        or options["group_policy"] != "not_applicable"
        or options["prior_estimation_indices"] != []
        or options["original_train_count"] != len(original)
        or options["sampling_rule"] != _RULE
        or manifest["lineage"]["original_train_count"] != len(original)
        or len(set(original)) != len(original)
        or manifest["indices_sha256"] != strict_canonical_hash(manifest["indices"])
    ):
        raise ValueError("support adapter options disagree with lineage")
    source_audit = manifest["support_source_audit"]
    expected_counts = {
        **{name: len(getattr(bundle, name).X) for name in ROLES},
        "train": len(original),
        "prior_estimation": 0,
    }
    if (
        source_audit["support_origin"] != "carved_from_original_pool"
        or source_audit["original_role_counts"] != expected_counts
        or source_audit["support_source_verified"] is not False
        or source_audit["support_source_receipt"] != manifest["source"]
    ):
        raise ValueError("support source audit disagrees with raw/role lineage")
    preparation_seed = options["preparation_seed"]
    if type(preparation_seed) is not int or not 0 <= preparation_seed < 2**32:
        raise ValueError("support preparation seed must be an unsigned 32-bit integer")
    positions = np.sort(
        np.random.RandomState(preparation_seed).choice(len(original), len(support.X), replace=False)
    )
    keep = np.ones(len(original), dtype=bool)
    keep[positions] = False
    if (
        np.asarray(original)[positions].tolist() != support.indices.tolist()
        or np.asarray(original)[keep].tolist() != bundle.train.indices.tolist()
    ):
        raise ValueError("support identities do not reproduce the declared label-blind sampling")
    prior = manifest["class_prior"]["population"]
    if isinstance(prior, bool) or not isinstance(prior, int | float) or not 0 < prior < 1:
        raise ValueError("support receipt requires original population prior, not test inference")


def save_support_dataset(bundle, support, manifest, directory):
    """Write five NPZs and a new receipt exclusively; refuse existing output."""
    _validate_product(bundle, support, manifest)
    payload = json.loads(json.dumps(manifest, allow_nan=False))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in _PARTS:
        part = support if name == "support" else getattr(bundle, name)
        path = directory / f"{name}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, X=part.X, y=part.labels, indices=part.indices)
        files[path.name] = file_hash(path)
    payload["files_sha256"] = files
    payload["receipt_sha256"] = strict_canonical_hash(payload)
    with (directory / "support_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return payload


def load_support_dataset(directory):
    """Verify receipt, files and all five roles before returning arrays.

    Checksums detect corruption, not authorisation or a signed label budget.
    Pickles are never loaded. A partial write has no usable final receipt.
    """
    directory = Path(directory)
    payload = json.loads((directory / "support_manifest.json").read_text(encoding="utf-8"))
    identity = payload.pop("receipt_sha256")
    if strict_canonical_hash(payload) != identity:
        raise ValueError("support receipt digest mismatch")
    if set(payload["files_sha256"]) != {f"{name}.npz" for name in _PARTS}:
        raise ValueError("support receipt must bind exactly five fixed role filenames")
    parts = {}
    for name in _PARTS:
        path = directory / f"{name}.npz"
        if file_hash(path) != payload["files_sha256"][path.name]:
            raise ValueError(f"{name} support product checksum mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"X", "y", "indices"}:
                raise ValueError("support NPZ must contain exactly X/y/indices")
            parts[name] = DatasetPart(
                arrays["X"].copy(),
                arrays["y"].copy(),
                "clean",
                arrays["indices"].copy(),
                name not in ("support", "test"),
            )
    bundle = DatasetBundle(*(parts[name] for name in ROLES))
    support = parts["support"]
    _validate_product(bundle, support, payload)
    payload["receipt_sha256"] = identity
    return bundle, support, payload
