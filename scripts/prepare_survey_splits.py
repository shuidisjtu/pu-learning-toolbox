# ruff: noqa: N803, N806

"""Survey split preparation (execution plan 1.4 — pilot data products).

Builds the user-prepared four-way split products for the seed datasets
(execution plan: CIFAR-10 + Spambase + IMDB first) from
``data/raw/<dataset>/`` and writes, per seed:

    data/splits/<dataset>/split_<seed>/
        train.npz / pu_val.npz / clean_val.npz / test.npz   # X, y, indices
        split_manifest.json                                  # + preprocessing

Splits follow protocol §2.4 (二)3 via ``prepare_survey_dataset``:
90% train / 5% pu_val / 5% clean_val (stratified, seed-driven), official
test split when the dataset has one (CIFAR-10, IMDB) and a stratified 20%
derived test otherwise (Spambase).  PU label views are NOT stored here:
the runner generates them later (SCAR contract), so the products always
carry REAL binary labels {0, 1} — this is the user-prepared part of the
protocol division of labour (see execution-plan decision D2).

Preprocessing per modality (statistics fitted on the train partition only
and frozen into the manifest):

- tabular (Spambase): train-fitted per-feature z-score, float32 features
- text (IMDB): locked SBERT encoding (revision pinned, content-addressed
  cache), 384-d float32 features
- image (CIFAR-10): uint8 NCHW source arrays; channel statistics/augment
  config fitted on train only and recorded in the manifest (the training
  pipeline applies the frozen scaling via ``transform_survey_images``)

Usage::

    uv run python scripts/prepare_survey_splits.py \\
        --datasets spambase,cifar10,imdb --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart
from pu_toolbox.experiment.datasets import prepare_survey_dataset
from pu_toolbox.experiment.image import fit_survey_image_preprocessing
from pu_toolbox.experiment.text import SBERT_MODEL_NAME, encode_survey_texts

SBERT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def load_spambase(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``spambase.data``: 57 features + binary label column."""
    values = np.loadtxt(raw_dir / "spambase" / "spambase.data", delimiter=",")
    return values[:, :-1].astype(np.float64), values[:, -1].astype(int)


def load_cifar10(raw_dir: Path):
    """Load CIFAR-10 train/test images (NCHW uint8) and raw class ids."""
    from torchvision.datasets import CIFAR10

    root = str(raw_dir / "cifar10")
    train = CIFAR10(root=root, train=True, download=False)
    test = CIFAR10(root=root, train=False, download=False)
    # PIL images are HWC; the survey pipeline expects NCHW uint8 source arrays.
    X_train = np.stack([np.asarray(image) for image, _ in train]).transpose(0, 3, 1, 2)
    X_test = np.stack([np.asarray(image) for image, _ in test]).transpose(0, 3, 1, 2)
    y_train = np.asarray([label for _, label in train])
    y_test = np.asarray([label for _, label in test])
    return X_train, y_train, X_test, y_test


def load_imdb_texts(raw_dir: Path) -> tuple[list[str], list[int], list[str], list[int]]:
    """List IMDB review texts and labels from the packed aclImdb archive."""
    import tarfile

    archive_path = raw_dir / "imdb" / "aclImdb_v1.tar.gz"
    texts_train: list[str] = []
    labels_train: list[int] = []
    texts_test: list[str] = []
    labels_test: list[int] = []

    def _collect(member_root: str, texts: list[str], labels: list[int]) -> None:
        with tarfile.open(archive_path, "r:gz") as archive:
            names = [
                (name, 1)
                for name in archive.getnames()
                if name.startswith(f"{member_root}/pos/") and name.endswith(".txt")
            ]
            names += [
                (name, 0)
                for name in archive.getnames()
                if name.startswith(f"{member_root}/neg/") and name.endswith(".txt")
            ]
            for name, label in names:
                texts.append(archive.extractfile(name).read().decode("utf-8", errors="replace"))
                labels.append(label)

    _collect("aclImdb/train", texts_train, labels_train)
    _collect("aclImdb/test", texts_test, labels_test)
    return texts_train, labels_train, texts_test, labels_test


def _standardizer_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-fitted per-feature mean/std (tabular preprocessing)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(np.abs(std) < 1e-12, 1.0, std)  # constant features stay put
    return mean, std


def _standardizer_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _remap_bundle(bundle, transform_fn) -> DatasetBundle:
    def _map(part: DatasetPart) -> DatasetPart:
        return DatasetPart(
            X=transform_fn(part.X),
            labels=part.labels,
            view=part.view,
            indices=part.indices,
            for_selection=part.for_selection,
        )

    return DatasetBundle(
        train=_map(bundle.train),
        pu_val=_map(bundle.pu_val),
        clean_val=_map(bundle.clean_val),
        test=_map(bundle.test),
    )


def save_split_products(bundle: DatasetBundle, manifest: dict[str, Any], run_dir: Path) -> None:
    """Write the four .npz partitions and the merged split manifest."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for role in ("train", "pu_val", "clean_val", "test"):
        part = getattr(bundle, role)
        np.savez_compressed(
            run_dir / f"{role}.npz",
            X=part.X,
            y=part.labels,
            indices=part.indices,
        )
    manifest_path = run_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sizes = [
        int(len(getattr(bundle, role).labels)) for role in ("train", "pu_val", "clean_val", "test")
    ]
    print(
        f"  wrote {run_dir.name}: sizes={sizes} "
        f"manifest_sha256={hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]}"
    )


def prepare_tabular(
    spec_name: str, X_source: np.ndarray, y_source: np.ndarray, *, seed: int, run_dir: Path
) -> None:
    bundle, manifest = prepare_survey_dataset(X_source, y_source, dataset=spec_name, seed=seed)
    mean, std = _standardizer_fit(bundle.train.X)
    scaled = _remap_bundle(bundle, lambda X: _standardizer_apply(X, mean, std))
    manifest["preprocessing"] = {
        "kind": "z-score",
        "fitted": "train_only",
        "feature_mean": [float(value) for value in mean],
        "feature_std": [float(value) for value in std],
    }
    save_split_products(scaled, manifest, run_dir)


def prepare_image(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    run_dir: Path,
) -> None:
    bundle, manifest = prepare_survey_dataset(
        X_train, y_train, dataset="cifar10", seed=seed, X_test=X_test, y_test=y_test
    )
    _, preprocessing = fit_survey_image_preprocessing(
        bundle.train.X, dataset="cifar10", train_augmentation="simaugment"
    )
    manifest["preprocessing"] = preprocessing.to_manifest()
    # Raw uint8 NCHW arrays are stored; the frozen scaling is applied by the
    # training pipeline (transform_survey_images + encoder), never twice.
    save_split_products(bundle, manifest, run_dir)


def prepare_text(
    texts_train: list[str],
    labels_train: list[int],
    texts_test: list[str],
    labels_test: list[int],
    *,
    seed: int,
    run_dir: Path,
    cache_dir: Path,
    encoder=None,
) -> None:
    """Split IMDB and store SBERT-encoded 384-d features per role."""
    y_source = np.asarray(labels_train, dtype=int)
    y_test = np.asarray(labels_test, dtype=int)
    # Placeholder 1-col features: only indices/labels of the split matter here.
    bundle, manifest = prepare_survey_dataset(
        np.arange(len(texts_train))[:, None],
        y_source,
        dataset="imdb",
        seed=seed,
        X_test=np.arange(len(texts_test))[:, None],
        y_test=y_test,
    )
    role_texts = []
    for role in ("train", "pu_val", "clean_val"):
        part = getattr(bundle, role)
        role_texts.extend(texts_train[i] for i in part.X[:, 0].astype(int))
    role_texts.extend(texts_test[i] for i in bundle.test.X[:, 0].astype(int))

    embeddings, encoder_meta = encode_survey_texts(
        role_texts, cache_dir=cache_dir, revision=SBERT_REVISION, encoder=encoder
    )
    encoded_parts = []
    cursor = 0
    for part in (bundle.train, bundle.pu_val, bundle.clean_val, bundle.test):
        slice_end = cursor + len(part.labels)
        encoded_parts.append(
            DatasetPart(
                X=embeddings[cursor:slice_end],
                labels=part.labels,
                view=part.view,
                indices=part.indices,
                for_selection=part.for_selection,
            )
        )
        cursor = slice_end
    encoded_bundle = DatasetBundle(*encoded_parts)
    manifest["preprocessing"] = {
        "kind": "sbert",
        "model_name": SBERT_MODEL_NAME,
        "revision": SBERT_REVISION,
        **{key: value for key, value in encoder_meta.items() if key != "cache_hit"},
    }
    save_split_products(encoded_bundle, manifest, run_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets", default="spambase,cifar10,imdb", help="comma-separated dataset names"
    )
    parser.add_argument("--seeds", default="0,1,2,3,4", help="comma-separated split seeds")
    parser.add_argument("--raw-dir", default="data/raw", help="raw data root")
    parser.add_argument("--out-dir", default="data/splits", help="split products root")
    parser.add_argument("--cache-dir", default="data/cache", help="SBERT embedding cache")
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    dataset_names = [value.strip() for value in args.datasets.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not dataset_names:
        print("error: --datasets must not be empty", file=sys.stderr)
        return 1

    for dataset in dataset_names:
        print(f"== {dataset} ==")
        for seed in seeds:
            run_dir = out_dir / dataset / f"split_{seed}"
            try:
                if dataset == "spambase":
                    X, y = load_spambase(raw_dir)
                    prepare_tabular(dataset, X, y, seed=seed, run_dir=run_dir)
                elif dataset == "cifar10":
                    X, y, X_test, y_test = load_cifar10(raw_dir)
                    prepare_image(X, y, X_test, y_test, seed=seed, run_dir=run_dir)
                elif dataset == "imdb":
                    texts_train, labels_train, texts_test, labels_test = load_imdb_texts(raw_dir)
                    prepare_text(
                        texts_train,
                        labels_train,
                        texts_test,
                        labels_test,
                        seed=seed,
                        run_dir=run_dir,
                        cache_dir=cache_dir,
                    )
                else:
                    print(f"error: unknown dataset {dataset!r}", file=sys.stderr)
                    return 1
            except (OSError, ValueError, ImportError) as exc:
                print(f"error: {dataset} seed {seed}: {exc}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
