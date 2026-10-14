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

from pu_toolbox.core.training_views import ROLES
from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart
from pu_toolbox.experiment.datasets import prepare_survey_dataset
from pu_toolbox.experiment.image import fit_survey_image_preprocessing
from pu_toolbox.experiment.text import SBERT_MODEL_NAME, encode_survey_texts

SBERT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

#: The datasets this pipeline prepares.  ``main`` rejects anything outside it
#: and the guard test requires reviewed catalog provenance for everything
#: inside it, so both read the same list: adding a dataset here without adding
#: its provenance fails a test rather than a run.
SPLIT_PIPELINE_DATASETS: tuple[str, ...] = (
    "spambase",
    "cifar10",
    "imdb",
    "adni",
    "mnist",
    "fashion_mnist",
    "twenty_newsgroups",
    "connect_4",
)
NEW_SPLIT_DATASETS = ("mnist", "fashion_mnist", "twenty_newsgroups", "connect_4")
NEWS_PREFIXES = ("alt", "comp", "misc", "rec", "sci", "soc", "talk")
NEWS_CATEGORIES = (
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
)
CONNECT4_CELLS = tuple(f"{column}{row}" for column in "abcdefg" for row in range(1, 7))
CONNECT4_VALUES = ("b", "o", "x")

#: The ADNI stand-in's classes in the order of the paper's Table B.1, which fixes the
#: ids 0-3 the protocol maps as ``0 vs 1,2,3``, and the image count of each class in the
#: pinned mirror: a different count means a different set of files.
ADNI_CLASSES = ("NonDemented", "VeryMildDemented", "ModerateDemented", "MildDemented")
ADNI_EXPECTED_COUNTS = {
    "NonDemented": 3200,
    "VeryMildDemented": 2240,
    "ModerateDemented": 64,
    "MildDemented": 896,
}
ADNI_IMAGE_SIZE = (128, 128)
_ADNI_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def load_mnist_arrays(raw_dir: Path, dataset: str):
    """Load validated uint8 MNIST-family arrays without merging official pools."""
    if dataset not in ("mnist", "fashion_mnist"):
        raise ValueError(f"not a MNIST-family dataset: {dataset!r}")
    keys = ("train_images", "train_labels", "test_images", "test_labels")
    with np.load(raw_dir / dataset / "raw_arrays.npz", allow_pickle=False) as source:
        if set(source.files) != set(keys):
            raise ValueError(f"{dataset}: expected exactly the keys {keys}")
        arrays = [source[key] for key in keys]
    for pool, images, labels in (("train", *arrays[:2]), ("test", *arrays[2:])):
        if images.dtype != np.uint8:
            raise ValueError(f"{dataset} {pool}: images must be uint8")
        if images.ndim != 3 or images.shape[1:] != (28, 28):
            raise ValueError(f"{dataset} {pool}: images must have shape (N,28,28)")
        if not len(images):
            raise ValueError(f"{dataset} {pool}: images must be non-empty")
        if labels.ndim != 1:
            raise ValueError(f"{dataset} {pool}: labels must be one-dimensional")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{dataset} {pool}: labels must have integer dtype")
        if len(labels) != len(images):
            raise ValueError(f"{dataset} {pool}: image/label length mismatch")
        if np.any((labels < 0) | (labels > 9)):
            raise ValueError(f"{dataset} {pool}: labels must be in 0..9")
    return arrays[0][:, None], arrays[1], arrays[2][:, None], arrays[3]


def load_twenty_newsgroups(raw_dir: Path):
    """Read all original 20News documents, sorted by category and filename."""
    pools = []
    for pool in ("train", "test"):
        root = raw_dir / "twenty_newsgroups" / f"20news-bydate-{pool}"
        entries = list(root.iterdir())
        if {entry.name for entry in entries} != set(NEWS_CATEGORIES) or any(
            not entry.is_dir() for entry in entries
        ):
            raise ValueError(f"20News {pool}: expected the 20 known categories")
        texts, labels = [], []
        for category in sorted(NEWS_CATEGORIES):
            members = sorted((root / category).iterdir(), key=lambda path: path.name)
            if not members:
                raise ValueError(f"20News {pool}: empty category {category}")
            for member in members:
                if not member.is_file():
                    raise ValueError(f"20News {pool}: not a document file: {member}")
                texts.append(member.read_bytes().decode("latin-1"))
                labels.append(NEWS_PREFIXES.index(category.split(".")[0]))
        pools.extend((texts, labels))
    return tuple(pools)


def load_connect4(raw_dir: Path):
    """Encode fixed b/o/x columns per cell; retain win/loss/draw source labels."""
    rows, labels = [], []
    path = raw_dir / "connect_4" / "connect-4.data"
    with path.open(encoding="ascii") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.rstrip("\r\n").split(",")
            if len(fields) != 43:
                raise ValueError(f"Connect-4 line {line_number}: expected 43 fields")
            if any(value not in CONNECT4_VALUES for value in fields[:42]):
                raise ValueError(f"Connect-4 line {line_number}: invalid cell value")
            if fields[-1] not in ("win", "loss", "draw"):
                raise ValueError(f"Connect-4 line {line_number}: invalid outcome")
            rows.append([CONNECT4_VALUES.index(value) for value in fields[:42]])
            labels.append(fields[-1])
    if not rows:
        raise ValueError("Connect-4: empty source")
    encoded = np.eye(3, dtype=np.float64)[np.asarray(rows)].reshape(len(rows), 126)
    return encoded, np.asarray(labels)


def source_encoding(dataset: str) -> dict[str, Any] | None:
    """Input representation, independent of fitted preprocessing or model identity."""
    if dataset in ("mnist", "fashion_mnist"):
        return {
            "input_format": "raw_arrays.npz",
            "source_layout": "NHW",
            "stored_layout": "NCHW",
            "dtype": "uint8",
            "official_test": True,
        }
    if dataset == "twenty_newsgroups":
        return {
            "input_format": "20news-bydate directories",
            "text_encoding": "latin-1",
            "remove": [],
            "ordering": "category_then_filename_lexicographic",
            "category_to_class": {
                category: NEWS_PREFIXES.index(category.split(".")[0])
                for category in NEWS_CATEGORIES
            },
        }
    if dataset == "connect_4":
        return {
            "input_format": "connect-4.data",
            "kind": "fixed_one_hot",
            "cells": list(CONNECT4_CELLS),
            "cell_values": list(CONNECT4_VALUES),
            "columns": [f"{cell}={value}" for cell in CONNECT4_CELLS for value in CONNECT4_VALUES],
            "source_labels": ["win", "loss", "draw"],
        }
    return None


def _file_identity(path: Path, relative_path: str) -> dict[str, Any]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
            size += len(chunk)
    return {"relative_path": relative_path, "sha256": hasher.hexdigest(), "bytes": size}


def source_input_identity(raw_dir: Path, dataset: str) -> dict[str, Any] | None:
    """Bind manifests to actual reader files, not merely their upstream archives."""
    root = raw_dir / dataset
    if dataset in ("mnist", "fashion_mnist"):
        paths = [root / "raw_arrays.npz"]
    elif dataset == "connect_4":
        paths = [root / "connect-4.data"]
    elif dataset == "twenty_newsgroups":
        paths = [
            path
            for pool in ("train", "test")
            for category in sorted((root / f"20news-bydate-{pool}").iterdir())
            for path in category.iterdir()
        ]
    else:
        return None
    files = [
        _file_identity(path, path.relative_to(root).as_posix())
        for path in sorted(paths, key=lambda path: path.relative_to(root).as_posix())
    ]
    listing = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "digest_scope": "reader_input_files",
        "digest_algorithm": "sha256_of_canonical_json_sorted_relative_path_sha256_bytes",
        "sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(),
        "files": files,
    }


def _attach_source_evidence(manifest, dataset, raw_handoff=None, input_identity=None):
    encoding = source_encoding(dataset)
    if encoding is not None:
        manifest["source_encoding"] = encoding
        if input_identity is not None:
            encoding["input_identity"] = input_identity
    if raw_handoff is not None:
        manifest["provenance"]["raw_handoff"] = raw_handoff


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


def load_adni(
    raw_dir: Path, *, expected_counts: dict[str, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load the ADNI stand-in from its four class archives (NCHW uint8, class ids 0-3).

    The archives are read in place and never extracted.  Reading and preprocessing follow
    PU-Bench's ``AlzheimerMRI`` loader (grayscale, LANCZOS resize to 128x128); its class
    lists and its own split are not used.  macOS litter in the zips (``__MACOSX/``,
    dot-files) is ignored, members are taken in sorted order so the arrays do not depend on
    how the zip was written, and a class whose image count differs from the pinned mirror
    is an error rather than a silently different dataset.
    """
    import io
    import zipfile

    from PIL import Image

    counts = ADNI_EXPECTED_COUNTS if expected_counts is None else expected_counts
    images: list[np.ndarray] = []
    labels: list[int] = []
    for class_id, name in enumerate(ADNI_CLASSES):
        with zipfile.ZipFile(raw_dir / "adni" / f"{name}.zip") as archive:
            members = sorted(
                info.filename
                for info in archive.infolist()
                if not info.is_dir()
                and info.filename.startswith(f"{name}/")
                and not info.filename.rsplit("/", 1)[-1].startswith(".")
                and info.filename.lower().endswith(_ADNI_IMAGE_SUFFIXES)
            )
            if len(members) != counts[name]:
                raise ValueError(
                    f"adni {name}: found {len(members)} images, expected {counts[name]}"
                )
            for member in members:
                with Image.open(io.BytesIO(archive.read(member))) as image:
                    resized = image.convert("L").resize(ADNI_IMAGE_SIZE, Image.LANCZOS)
                images.append(np.asarray(resized, dtype=np.uint8))
                labels.append(class_id)
    return np.stack(images)[:, np.newaxis, :, :], np.asarray(labels)


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
    run_dir.mkdir(parents=True, exist_ok=manifest.get("dataset") not in NEW_SPLIT_DATASETS)
    for role in ROLES:
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
    sizes = [int(len(getattr(bundle, role).labels)) for role in ROLES]
    print(
        f"  wrote {run_dir.name}: sizes={sizes} "
        f"manifest_sha256={hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]}"
    )


def load_download_record(raw_dir: Path, dataset: str) -> dict[str, Any] | None:
    """The local fetch record for one dataset, or ``None`` when there is none.

    ``data/raw`` is machine-local, and its digests describe the archive *this*
    machine downloaded, so they belong in the artifact rather than in the
    reviewed catalog.  A record that exists but cannot answer for the download
    is a defect, not an absent record, and must not be read as one.
    """
    return load_source_records(raw_dir, dataset)[0]


def load_source_records(raw_dir: Path, dataset: str):
    """Return separate download and raw-handoff records without inventing a time."""
    path = raw_dir / dataset / "provenance.json"
    if not path.is_file():
        return None, None
    receipt_bytes = path.read_bytes()
    payload = json.loads(receipt_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    declared = payload.get("dataset")
    if declared is not None and declared != dataset:
        raise ValueError(
            f"{path} records a download of {declared!r}, not {dataset!r}; its "
            "digest would otherwise be written into the wrong manifest"
        )
    if "files" in payload and "sha256" not in payload:
        if dataset in NEW_SPLIT_DATASETS and not payload.get("collected_at"):
            return None, _raw_handoff(path, payload, receipt_bytes=receipt_bytes)
        return _aggregate_download_record(path, payload), None
    return payload, None


def _raw_handoff(path: Path, payload: dict[str, Any], *, receipt_bytes: bytes) -> dict[str, Any]:
    """Validate every handed-off file against local bytes; preserve the full receipt."""
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{path}: raw handoff requires a non-empty files list")
    seen = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: file entries must be objects")
        name, digest, size = item.get("filename"), item.get("sha256"), item.get("bytes")
        if (
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or any(value in name for value in ("/", "\\", ":"))
            or name in seen
        ):
            raise ValueError(f"{path}: invalid or duplicate filename {name!r}")
        seen.add(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in digest)
        ):
            raise ValueError(f"{path}: invalid SHA256 for {name}")
        if type(size) is not int or size < 0:
            raise ValueError(f"{path}: invalid bytes for {name}")
        source = path.parent / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"{path}: missing or unsafe source file {name}")
        actual = _file_identity(source, name)
        if actual["bytes"] != size or actual["sha256"] != digest.lower():
            raise ValueError(f"{path}: raw handoff digest/bytes mismatch for {name}")
    return {
        "receipt": payload,
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "receipt_digest_scope": "original_receipt_bytes",
        "files_verified": True,
        "downloaded_at": None,
        "download_time_status": "unknown",
    }


def _aggregate_download_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """One download record for a dataset shipped as several archives.

    The digest is taken over ``<sha256>  <filename>`` lines in filename order, so it
    changes if any archive changes and not with the order the collector listed them.
    """
    if not payload.get("collected_at"):
        raise ValueError(f"{path} lists archives but records no collected_at")
    files = sorted(payload["files"], key=lambda item: item["filename"])
    listing = "".join(f"{item['sha256']}  {item['filename']}\n" for item in files)
    record = {
        "sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(),
        "bytes": sum(int(item["bytes"]) for item in files),
        "downloaded_at": payload["collected_at"],
    }
    if payload.get("mirror_url"):
        record["mirror_url"] = payload["mirror_url"]
    return record


def prepare_tabular(
    spec_name: str,
    X_source: np.ndarray,
    y_source: np.ndarray,
    *,
    seed: int,
    run_dir: Path,
    download: dict[str, Any] | None = None,
    raw_handoff: dict[str, Any] | None = None,
    input_identity: dict[str, Any] | None = None,
) -> None:
    bundle, manifest = prepare_survey_dataset(
        X_source, y_source, dataset=spec_name, seed=seed, download=download
    )
    mean, std = _standardizer_fit(bundle.train.X)
    scaled = _remap_bundle(bundle, lambda X: _standardizer_apply(X, mean, std))
    manifest["preprocessing"] = {
        "kind": "z-score",
        "fitted": "train_only",
        "feature_mean": [float(value) for value in mean],
        "feature_std": [float(value) for value in std],
    }
    _attach_source_evidence(manifest, spec_name, raw_handoff, input_identity)
    save_split_products(scaled, manifest, run_dir)


def prepare_image(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    *,
    seed: int,
    run_dir: Path,
    download: dict[str, Any] | None = None,
    dataset: str = "cifar10",
    raw_handoff: dict[str, Any] | None = None,
    input_identity: dict[str, Any] | None = None,
) -> None:
    """Prepare one image dataset; a dataset with no official test passes none (ADNI)."""
    bundle, manifest = prepare_survey_dataset(
        X_train,
        y_train,
        dataset=dataset,
        seed=seed,
        X_test=X_test,
        y_test=y_test,
        download=download,
    )
    # Split preparation applies no augmentation: the products store raw uint8 and
    # the training pipeline owns augmentation (protocol-locked "none" for the
    # pilot).  Recording the estimator-side default here would misdescribe what
    # these artifacts actually are.
    _, preprocessing = fit_survey_image_preprocessing(
        bundle.train.X, dataset=dataset, train_augmentation="none"
    )
    manifest["preprocessing"] = preprocessing.to_manifest()
    # Raw uint8 NCHW arrays are stored; the frozen scaling is applied by the
    # training pipeline (transform_survey_images + encoder), never twice.
    _attach_source_evidence(manifest, dataset, raw_handoff, input_identity)
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
    download: dict[str, Any] | None = None,
    dataset: str = "imdb",
    raw_handoff: dict[str, Any] | None = None,
    input_identity: dict[str, Any] | None = None,
) -> None:
    """Split text data and store SBERT-encoded 384-d features per role."""
    y_source = np.asarray(labels_train, dtype=int)
    y_test = np.asarray(labels_test, dtype=int)
    # Placeholder 1-col features: only indices/labels of the split matter here.
    bundle, manifest = prepare_survey_dataset(
        np.arange(len(texts_train))[:, None],
        y_source,
        dataset=dataset,
        seed=seed,
        X_test=np.arange(len(texts_test))[:, None],
        y_test=y_test,
        download=download,
    )
    # Three roles, not core's ``ROLES``: these rows are carved from
    # ``texts_train`` and encoded from it, while ``test`` comes from
    # ``texts_test`` and is appended on its own line below.  The set means
    # "roles sourced from the train pool" -- a different concept from the four
    # partition roles; widening it to ``ROLES`` would index ``texts_train``
    # with test rows.
    role_texts = []
    for role in ("train", "pu_val", "clean_val"):
        part = getattr(bundle, role)
        role_texts.extend(texts_train[i] for i in part.X[:, 0].astype(int))
    role_texts.extend(texts_test[i] for i in bundle.test.X[:, 0].astype(int))

    embeddings, encoder_meta = encode_survey_texts(
        role_texts, cache_dir=cache_dir, revision=SBERT_REVISION, encoder=encoder
    )
    # normalize_embeddings=False disables an extra encode-time pass; the pinned
    # SentenceTransformer has a Normalize module in its model pipeline. Record
    # the effective output, including for injected test encoders, not an
    # inference from the encode() argument alone.
    squared_norms = np.einsum("ij,ij->i", embeddings, embeddings, dtype=np.float64)
    is_unit_norm = bool(np.allclose(squared_norms, 1.0, rtol=1e-4, atol=1e-4))
    backend_is_model = encoder_meta["encoder_backend"] == "sentence-transformers"
    if backend_is_model and not is_unit_norm:
        raise ValueError("pinned SBERT pipeline output is not L2-normalized as expected")
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
        "effective_output_normalization": "l2_unit_norm" if is_unit_norm else "not_l2_unit_norm",
        "normalization_source": (
            "model_pipeline_module" if backend_is_model else "injected_encoder_output"
        ),
        **{key: value for key, value in encoder_meta.items() if key != "cache_hit"},
    }
    _attach_source_evidence(manifest, dataset, raw_handoff, input_identity)
    save_split_products(encoded_bundle, manifest, run_dir)


def _load_dataset(dataset: str, raw_dir: Path):
    """A dataset's raw arrays, in the order its ``prepare_*`` takes them.

    Separate from ``main`` so the one thing that is genuinely per-dataset
    happens exactly once: everything downstream of it varies by seed.
    """
    if dataset in ("mnist", "fashion_mnist"):
        return load_mnist_arrays(raw_dir, dataset)
    if dataset == "twenty_newsgroups":
        return load_twenty_newsgroups(raw_dir)
    if dataset == "connect_4":
        return load_connect4(raw_dir)
    if dataset == "spambase":
        return load_spambase(raw_dir)
    if dataset == "cifar10":
        return load_cifar10(raw_dir)
    if dataset == "imdb":
        return load_imdb_texts(raw_dir)
    if dataset == "adni":
        return load_adni(raw_dir)
    raise ValueError(f"unknown dataset {dataset!r}")


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
    unknown = [name for name in dataset_names if name not in SPLIT_PIPELINE_DATASETS]
    if unknown:
        print(
            f"error: unknown dataset(s) {unknown}; choose from {list(SPLIT_PIPELINE_DATASETS)}",
            file=sys.stderr,
        )
        return 1

    for dataset in dataset_names:
        if dataset in NEW_SPLIT_DATASETS:
            for seed in seeds:
                target = out_dir / dataset / f"split_{seed}"
                if target.exists():
                    print(f"error: refusing to overwrite existing split {target}", file=sys.stderr)
                    return 1
    for dataset in dataset_names:
        print(f"== {dataset} ==")
        try:
            download, raw_handoff = load_source_records(raw_dir, dataset)
            # Loaded once for every seed, not once per seed: the seed decides
            # how the pool is split, never what is in it.  Loading inside the
            # loop re-read all 813 MB of CIFAR-10 pickles and re-extracted
            # fifty thousand IMDB reviews from the archive once per seed.  The
            # archive is the expensive one -- measured at roughly two and a
            # half minutes per pass, against about five seconds for the pickles
            # -- and it was being paid five times per rebuild even when every
            # encoding came from cache.
            loaded = _load_dataset(dataset, raw_dir)
            input_identity = source_input_identity(raw_dir, dataset)
        except (OSError, ValueError, ImportError) as exc:
            print(f"error: {dataset}: {exc}", file=sys.stderr)
            return 1
        for seed in seeds:
            run_dir = out_dir / dataset / f"split_{seed}"
            try:
                if dataset in ("spambase", "connect_4"):
                    prepare_tabular(
                        dataset,
                        *loaded,
                        seed=seed,
                        run_dir=run_dir,
                        download=download,
                        raw_handoff=raw_handoff,
                        input_identity=input_identity,
                    )
                elif dataset in ("cifar10", "adni", "mnist", "fashion_mnist"):
                    prepare_image(
                        *loaded,
                        seed=seed,
                        run_dir=run_dir,
                        download=download,
                        dataset=dataset,
                        raw_handoff=raw_handoff,
                        input_identity=input_identity,
                    )
                else:
                    prepare_text(
                        *loaded,
                        seed=seed,
                        run_dir=run_dir,
                        cache_dir=cache_dir,
                        download=download,
                        dataset=dataset,
                        raw_handoff=raw_handoff,
                        input_identity=input_identity,
                    )
            except (OSError, ValueError, ImportError) as exc:
                print(f"error: {dataset} seed {seed}: {exc}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
