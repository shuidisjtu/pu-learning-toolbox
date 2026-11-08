# ruff: noqa: N803
"""Independent clean-support products; never overwrite frozen four-way splits.

Two source modes exist for the same five-role product contract:

* the raw-reader path, which rebuilds the four-way split from reviewed source
  arrays (tabular and image datasets);
* the frozen-split path, which carves support directly out of an existing
  four-way split.  Text datasets must use it because their split products hold
  the pinned SBERT representation; ADNI uses it so the substitute images are
  not re-read and re-scaled by a second path.

The support draw is always label-blind, uniform without replacement, and
predeclared: ``support_seed`` equals the split seed and is fixed before any
label is inspected.  ADNI has no subject IDs in the reviewed substitute source,
so its only admitted group policy is ``unverifiable``: the support manifest
records that subject isolation cannot be checked, plus the owner approval for
that known risk.  It is never reported as verified isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.training_views import ROLES
from ..utils.serialization import array_hash, file_hash, strict_canonical_hash
from .bundle import DatasetBundle, DatasetPart, split_indices_digest, validate_bundle
from .datasets import prepare_survey_dataset
from .image import fit_survey_image_preprocessing
from .text import SBERT_EMBEDDING_DIMENSION

#: Every dataset the five-role support contract covers, and its input modality.
SUPPORT_MODALITY = {
    "spambase": "tabular",
    "connect_4": "tabular",
    "mnist": "image",
    "fashion_mnist": "image",
    "cifar10": "image",
    "adni": "image",
    "imdb": "text",
    "twenty_newsgroups": "text",
}
SUPPORT_DATASETS = tuple(SUPPORT_MODALITY)
#: Datasets whose reviewed raw readers may rebuild the split from source arrays.
RAW_SUPPORT_DATASETS = ("spambase", "connect_4", "mnist", "fashion_mnist", "cifar10")
#: Datasets whose support must be carved from an existing frozen four-way split.
SPLIT_SUPPORT_DATASETS = ("imdb", "twenty_newsgroups", "adni")
SUPPORT_SEED_POLICY = "support_seed equals split_seed, predeclared before label inspection"
_EXPLICIT_SEED_POLICY = "explicit support_seed differs from split_seed"
_ADNI_GROUP_POLICY = "unverifiable"
_ADNI_GROUP_REASON = "source_has_no_subject_ids"
_ADNI_KNOWN_RISKS = ("same-subject slices may cross roles",)
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

    Returns ``(bundle, support, manifest)``. Only the reviewed raw-reader
    datasets are admitted here, with explicit ``group_policy='not_applicable'``.
    Text datasets must use :func:`prepare_support_from_split`; ADNI must use it
    too, because its only admitted policy is the explicit ``unverifiable``
    subject-isolation gap.
    """
    if dataset not in RAW_SUPPORT_DATASETS:
        raise ValueError(
            "dataset needs an independently reviewed support/group/text policy; "
            "text and ADNI products must be carved from a frozen split"
        )
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
    image = SUPPORT_MODALITY[dataset] == "image"
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
        source_mode="raw_arrays",
        modality=SUPPORT_MODALITY[dataset],
        support_seed=support_seed,
        support_seed_policy=(
            SUPPORT_SEED_POLICY if support_seed == seed else _EXPLICIT_SEED_POLICY
        ),
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


def prepare_support_from_split(
    split_dir,
    *,
    dataset,
    support_size,
    support_seed=None,
    seed=None,
    source,
    group_policy="not_applicable",
    owner_approval=None,
):
    """Carve a five-role product directly out of a frozen four-way split.

    Text datasets use this path because their four-way products already hold the
    pinned SBERT representation; re-encoding the raw reviews would be a second
    representation and could move the support rows away from the rows they train
    beside.  ADNI uses it so the substitute images are not re-read and re-scaled
    through a second code path.

    The sampling rule is the protocol rule: positions are drawn uniformly
    without replacement from the split's train role, the seed is fixed before
    any label is inspected, and both classes must survive in support and in the
    remaining train.  A single-class draw is refused, never retried.

    support_seed is part of the product identity: it must equal the split seed
    ('support_seed equals split_seed, predeclared before label inspection').
    ADNI has no subject IDs in the reviewed substitute source, so its only
    admitted group_policy is 'unverifiable'; the returned manifest records that
    subject isolation cannot be checked and the owner approval for that known
    risk.  It never reports verified subject isolation.
    """
    if dataset not in SPLIT_SUPPORT_DATASETS:
        raise ValueError(
            "split-derived support is defined for text and ADNI datasets only; "
            "use the reviewed raw-reader path for tabular/image datasets"
        )
    if not isinstance(source, str) or not source.strip():
        raise ValueError("split source identity must be recorded")
    if type(support_size) is not int or support_size < 2:
        raise ValueError("support_size must be an explicit integer of at least two")

    directory = Path(split_dir)
    manifest_path = directory / "split_manifest.json"
    try:
        split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read frozen split manifest {manifest_path}: {exc}") from exc
    if not isinstance(split_manifest, dict):
        raise ValueError("frozen split manifest must be a JSON object")
    if split_manifest.get("dataset") != dataset:
        raise ValueError(
            f"frozen split declares dataset {split_manifest.get('dataset')!r}, not {dataset!r}"
        )
    split_seed = split_manifest.get("seed")
    if type(split_seed) is not int or not 0 <= split_seed < 2**32:
        raise ValueError("frozen split manifest seed must be an unsigned 32-bit integer")
    if seed is not None and seed != split_seed:
        raise ValueError("requested seed disagrees with the frozen split manifest")
    if support_seed is None:
        support_seed = split_seed
    if type(support_seed) is not int or not 0 <= support_seed < 2**32:
        raise ValueError("support_seed must be an unsigned 32-bit integer")
    if support_seed != split_seed:
        raise ValueError(
            f"{SUPPORT_SEED_POLICY}; got split seed {split_seed} and support_seed {support_seed}"
        )

    modality = SUPPORT_MODALITY[dataset]
    if split_manifest.get("modality") != modality:
        raise ValueError(
            f"frozen split modality {split_manifest.get('modality')!r} disagrees with "
            f"{dataset!r} ({modality!r})"
        )
    declared_digest = split_manifest.get("indices_sha256")
    if not isinstance(declared_digest, str) or not declared_digest:
        raise ValueError("frozen split manifest must record indices_sha256")
    split_indices_digest(directory, declared=declared_digest)

    parts = {}
    for name in ROLES:
        path = directory / f"{name}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"X", "y", "indices"}:
                    raise ValueError(f"{name}.npz must contain exactly X/y/indices")
                arrays = {key: archive[key].copy() for key in ("X", "y", "indices")}
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read frozen split {path}: {exc}") from exc
        parts[name] = DatasetPart(
            arrays["X"], arrays["y"], "clean", arrays["indices"], name != "test"
        )
    bundle = DatasetBundle(*(parts[name] for name in ROLES))
    validate_bundle(bundle)

    for name, part in parts.items():
        if part.X.ndim not in (2, 4) or part.X.dtype.kind not in "fiu":
            raise ValueError(f"{name} must hold a real numeric feature array")
        if not np.isfinite(part.X).all():
            raise ValueError(f"{name} features must be finite")
        if part.labels.shape != (len(part.X),) or part.indices.shape != (len(part.X),):
            raise ValueError(f"{name} labels/indices must align with features")
    if modality == "text":
        preprocessing = split_manifest.get("preprocessing")
        if not isinstance(preprocessing, dict) or preprocessing.get("kind") != "sbert":
            raise ValueError("text split must record its pinned SBERT preprocessing")
        if preprocessing.get("embedding_dimension") != SBERT_EMBEDDING_DIMENSION:
            raise ValueError("text split SBERT dimension disagrees with the pinned representation")
        for name, part in parts.items():
            if part.X.ndim != 2 or part.X.shape[1] != SBERT_EMBEDDING_DIMENSION:
                raise ValueError(
                    f"{name} must hold the frozen {SBERT_EMBEDDING_DIMENSION}-d SBERT features"
                )
    elif modality == "image":
        for name, part in parts.items():
            if part.X.ndim != 4 or part.X.dtype != np.uint8:
                raise ValueError(f"{name} must hold original uint8 NCHW images")
    else:  # pragma: no cover - the guard above already limits this path
        raise ValueError("split-derived support covers text and image datasets only")

    train = bundle.train
    if support_size > len(train.X) - 2:
        raise ValueError("support_size must leave at least two training rows")
    positions = np.sort(
        np.random.RandomState(support_seed).choice(len(train.X), support_size, replace=False)
    )
    keep = np.ones(len(train.X), dtype=bool)
    keep[positions] = False
    if any(set(np.unique(train.labels[mask])) != {0, 1} for mask in (positions, keep)):
        raise ValueError("single-class support or remaining train draw; no label-based retry")

    original_counts = {name: len(part.X) for name, part in parts.items()}
    original_counts["prior_estimation"] = 0
    remaining = DatasetPart(train.X[keep], train.labels[keep], "clean", train.indices[keep])
    support = DatasetPart(
        train.X[positions], train.labels[positions], "clean", train.indices[positions], False
    )
    bundle = DatasetBundle(remaining, bundle.pu_val, bundle.clean_val, bundle.test)

    if modality == "image":
        _, spec = fit_survey_image_preprocessing(
            remaining.X, dataset=dataset, train_augmentation="none"
        )
        preprocessing = spec.to_manifest()
        preprocessing["stored_features"] = "raw_uint8_normalization_not_applied"
        preprocessing["source"] = "refit_on_remaining_train"
    else:
        preprocessing = dict(preprocessing)
        preprocessing["stored_features"] = "sbert_embeddings_l2_normalized"
        preprocessing["source"] = "frozen_split"

    class_prior = split_manifest.get("class_prior")
    if not isinstance(class_prior, dict):
        raise ValueError("frozen split manifest must record class_prior")
    population = class_prior.get("population")
    if isinstance(population, bool) or not isinstance(population, int | float):
        raise ValueError("frozen split population prior must be a real scalar")
    if not 0 < float(population) < 1:
        raise ValueError("frozen split population prior must be in (0, 1)")

    options = {
        "dataset": dataset,
        "source": source,
        "prior_estimation_indices": [],
        "original_train_count": len(train.X),
        "support_removed_from_train": True,
        "group_policy": group_policy,
        "preparation_seed": support_seed,
        "sampling_rule": _RULE,
    }
    if dataset == "adni":
        if group_policy != _ADNI_GROUP_POLICY:
            raise ValueError(
                "ADNI requires group_policy='unverifiable': the reviewed substitute source "
                "has no subject IDs, so subject isolation cannot be checked"
            )
        if not isinstance(owner_approval, str) or not owner_approval.strip():
            raise ValueError("ADNI unverifiable support requires the owner approval id")
        options.update(
            group_policy_reason=_ADNI_GROUP_REASON,
            subject_isolation="not_checkable",
            known_risks=list(_ADNI_KNOWN_RISKS),
            owner_approval=owner_approval,
        )
    else:
        if group_policy != "not_applicable":
            raise ValueError("text split support requires group_policy='not_applicable'")
        if owner_approval is not None:
            raise ValueError("owner approval is only recorded for the ADNI subject gap")

    parts = {name: getattr(bundle, name) for name in ROLES} | {"support": support}
    manifest = {
        "schema_version": _SCHEMA,
        "formal_admission": False,
        "comparison_group": "extra_clean_support",
        "budget_approval": "not_granted_by_preparer",
        "split_policy": "frozen_four_way_then_uniform_support_from_train",
        "dataset": dataset,
        "seed": split_seed,
        "modality": modality,
        "support_seed": support_seed,
        "support_seed_policy": SUPPORT_SEED_POLICY,
        "source": source,
        "source_mode": "frozen_split",
        "preprocessing": preprocessing,
        "parts": {name: _part_identity(part) for name, part in parts.items()},
        "indices": {name: part.indices.tolist() for name, part in parts.items()},
        "role_sizes": {name: len(part.X) for name, part in parts.items()},
        "role_positive_rates": {name: float(part.labels.mean()) for name, part in parts.items()},
        "class_prior": {
            **class_prior,
            "train": float(bundle.train.labels.mean()),
        },
        "lineage": {
            "base_split_manifest_path": str(manifest_path.resolve()),
            "base_split_manifest_sha256": file_hash(manifest_path),
            "base_split_manifest_content_sha256": strict_canonical_hash(split_manifest),
            "original_train_indices": train.indices.tolist(),
            "original_train_count": len(train.X),
            "source_mode": "frozen_split",
            "split_role_indices_sha256": declared_digest,
            "split_role_X_sha256": {
                name: array_hash(part.X) for name, part in parts.items() if name != "support"
            },
        },
        "support_source_audit": {
            "support_origin": "carved_from_original_pool",
            "original_role_counts": original_counts,
            "support_source_verified": False,
            "support_source_receipt": source,
        },
        "support_adapter_options": options,
    }
    manifest["indices_sha256"] = strict_canonical_hash(manifest["indices"])
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
    dataset = manifest["dataset"]
    if manifest.get("modality") not in (None, SUPPORT_MODALITY[dataset]):
        raise ValueError("support product modality disagrees with its dataset")
    seed = manifest.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("support product must record its split seed")
    support_seed = manifest.get("support_seed", options.get("preparation_seed"))
    if type(support_seed) is not int or not 0 <= support_seed < 2**32:
        raise ValueError("support product must record its support seed")
    policy = manifest.get("support_seed_policy")
    if policy is not None and (
        not isinstance(policy, str) or policy not in (SUPPORT_SEED_POLICY, _EXPLICIT_SEED_POLICY)
    ):
        raise ValueError("support seed policy must be one of the declared policies")
    if policy == SUPPORT_SEED_POLICY and support_seed != seed:
        raise ValueError("support_seed policy requires support_seed to equal the split seed")
    source_mode = manifest.get("source_mode", options.get("source_mode", "raw_arrays"))
    if source_mode not in ("raw_arrays", "frozen_split"):
        raise ValueError("support source mode must be raw_arrays or frozen_split")
    group_policy = options.get("group_policy")
    if dataset == "adni":
        if group_policy != _ADNI_GROUP_POLICY:
            raise ValueError("ADNI support requires group_policy='unverifiable'")
        if options.get("group_policy_reason") != _ADNI_GROUP_REASON:
            raise ValueError("ADNI unverifiable support requires its source reason")
        if options.get("subject_isolation") != "not_checkable":
            raise ValueError("ADNI support must record subject isolation as not_checkable")
        risks = options.get("known_risks")
        if (
            not isinstance(risks, list)
            or not risks
            or any(not isinstance(risk, str) or not risk.strip() for risk in risks)
        ):
            raise ValueError("ADNI unverifiable support must record its known risks")
        approval = options.get("owner_approval")
        if not isinstance(approval, str) or not approval.strip():
            raise ValueError("ADNI unverifiable support requires the owner approval id")
    elif group_policy != "not_applicable":
        raise ValueError("non-ADNI support requires group_policy='not_applicable'")
    if (
        options["dataset"] != dataset
        or options["source"] != manifest["source"]
        or options["support_removed_from_train"] is not True
        or options["prior_estimation_indices"] != []
        or options["original_train_count"] != len(original)
        or options["sampling_rule"] != _RULE
        or options["preparation_seed"] != support_seed
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
    payload.pop("receipt_sha256", None)
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
