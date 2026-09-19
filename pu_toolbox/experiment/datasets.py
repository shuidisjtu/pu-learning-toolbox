# ruff: noqa: N803, N806

"""Survey dataset catalog and deterministic four-way split preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split

from pu_toolbox.utils.serialization import canonical_hash

from .bundle import DatasetBundle, DatasetPart, validate_bundle

Modality = Literal["image", "text", "tabular"]


@dataclass(frozen=True)
class DatasetProvenance:
    """Where one survey dataset was published, and what its publisher states about it.

    ``license_name`` is empty for a source that states no licence, and
    ``license_checked_on`` is required so that emptiness cannot be read as an
    unfinished entry: a date with no name means someone read the page and the
    page names nothing.  Where a licence *is* stated it is transcribed together
    with the URL it was read from; the status a manifest reports is derived from
    the name rather than stored next to it, so the two cannot disagree.

    ``label_semantics`` carries what the source labels *mean*.  A manifest's
    ``positive_classes``/``negative_classes`` say which ids fall on which side,
    and an id on its own does not say what the class is.
    """

    source_url: str
    version: str
    citation: str
    label_semantics: str
    #: The day the licence was read off that page, so a reader knows how stale
    #: the reading is rather than treating it as permanent.
    license_checked_on: str
    license_name: str = ""
    license_url: str = ""
    license_notes: str = ""

    def __post_init__(self) -> None:
        if bool(self.license_name) != bool(self.license_url):
            raise ValueError(
                "a stated licence needs the URL it was read from, and a URL "
                "without a name is not a licence"
            )
        if not self.license_checked_on:
            raise ValueError("provenance must record when its licence was checked")


@dataclass(frozen=True)
class SurveyDatasetSpec:
    """Locked binary mapping and test-source policy for one survey dataset."""

    name: str
    modality: Modality
    positive_classes: tuple[int | str, ...]
    negative_classes: tuple[int | str, ...]
    has_official_test: bool
    #: Reviewed publishing facts, recorded into every split manifest (P1.2).
    #: Absent for datasets whose split pipeline has not been built yet.
    provenance: DatasetProvenance | None = None


_CATALOG = {
    "mnist": SurveyDatasetSpec("mnist", "image", (0, 2, 4, 6, 8), (1, 3, 5, 7, 9), True),
    "fashion_mnist": SurveyDatasetSpec(
        "fashion_mnist", "image", (0, 2, 3, 4, 6), (1, 5, 7, 8, 9), True
    ),
    "cifar10": SurveyDatasetSpec(
        "cifar10",
        "image",
        (0, 1, 8, 9),
        (2, 3, 4, 5, 6, 7),
        True,
        provenance=DatasetProvenance(
            source_url="https://www.cs.toronto.edu/~kriz/cifar.html",
            version="CIFAR-10 python distribution (Krizhevsky, 2009 tech report)",
            citation=(
                "Krizhevsky, A. (2009). Learning Multiple Layers of Features "
                "from Tiny Images. Technical report, University of Toronto."
            ),
            label_semantics=(
                "class ids 0-9 = airplane, automobile, bird, cat, deer, dog, "
                "frog, horse, ship, truck; the positive classes are the four "
                "vehicles, {0, 1, 8, 9}"
            ),
            license_checked_on="2026-09-19",
            license_notes=(
                "The distribution page states no licence; it asks that the tech "
                "report be cited. Reusers record terms inconsistently (public "
                "domain / MIT / unknown) and none of those traces to the "
                "publisher, so none is transcribed here."
            ),
        ),
    ),
    "adni": SurveyDatasetSpec("adni", "image", (0,), (1, 2, 3), False),
    "imdb": SurveyDatasetSpec(
        "imdb",
        "text",
        (1,),
        (0,),
        True,
        provenance=DatasetProvenance(
            source_url="https://ai.stanford.edu/~amaas/data/sentiment/",
            version="aclImdb_v1 (Maas et al., ACL 2011)",
            citation=(
                "Maas, Daly, Pham, Huang, Ng, Potts (2011). Learning Word "
                "Vectors for Sentiment Analysis. ACL-HLT 2011, 142-150."
            ),
            label_semantics=(
                "label 1 = positive sentiment, 0 = negative; only strongly "
                "polarised reviews are labelled (score >= 7/10 or <= 4/10)"
            ),
            license_checked_on="2026-09-19",
            license_notes=(
                "The distribution page states no licence; it asks that the ACL 2011 paper be cited."
            ),
        ),
    ),
    "twenty_newsgroups": SurveyDatasetSpec(
        "twenty_newsgroups", "text", (0, 1, 2, 3), (4, 5, 6), True
    ),
    "spambase": SurveyDatasetSpec(
        "spambase",
        "tabular",
        (1,),
        (0,),
        False,
        provenance=DatasetProvenance(
            source_url="https://archive.ics.uci.edu/dataset/94/spambase",
            version="UCI repository id 94 (donated 1999-06-30)",
            citation=(
                "Hopkins, Reeber, Forman, Suermondt (1999). Spambase. UCI "
                "Machine Learning Repository. https://doi.org/10.24432/C53G6X"
            ),
            label_semantics=("last column: 1 = spam (the positive class), 0 = not spam"),
            license_checked_on="2026-09-19",
            license_name="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/legalcode",
            license_notes=("The dataset page states the licence and its attribution requirement."),
        ),
    ),
    "connect_4": SurveyDatasetSpec("connect_4", "tabular", ("win",), ("loss", "draw"), False),
}

_ALIASES = {
    "20news": "twenty_newsgroups",
    "connect-4": "connect_4",
    "f-mnist": "fashion_mnist",
    "f_mnist": "fashion_mnist",
}


def survey_dataset_catalog() -> dict[str, SurveyDatasetSpec]:
    """Return a copy of the eight-dataset protocol catalog."""
    return dict(_CATALOG)


def binaryize_survey_labels(
    labels: np.ndarray,
    dataset: str | SurveyDatasetSpec,
) -> np.ndarray:
    """Apply the protocol-locked positive/negative class mapping."""
    spec = _resolve_spec(dataset)
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("survey labels must be one-dimensional.")
    included = np.isin(labels, spec.positive_classes + spec.negative_classes)
    if not np.all(included):
        unknown = np.unique(labels[~included]).tolist()
        raise ValueError(
            f"{spec.name} contains labels outside its locked binary mapping: {unknown!r}."
        )
    binary = np.isin(labels, spec.positive_classes).astype(int)
    if np.unique(binary).size != 2:
        raise ValueError(f"{spec.name} data must contain both mapped classes.")
    return binary


def prepare_survey_dataset(
    X_source: np.ndarray,
    y_source: np.ndarray,
    *,
    dataset: str | SurveyDatasetSpec,
    seed: int,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    source_indices: np.ndarray | None = None,
    test_indices: np.ndarray | None = None,
    download: dict[str, Any] | None = None,
) -> tuple[DatasetBundle, dict[str, Any]]:
    """Prepare the survey's train/PU-val/clean-val/test clean-label bundle.

    Datasets with an official test split require ``X_test``/``y_test``.
    Spambase, Connect-4 and ADNI instead derive a stratified 20% test split
    from ``X_source``. The remaining training source is split 90%/5%/5%.
    """
    spec = _resolve_spec(dataset)
    X_source = np.asarray(X_source)
    y_source = np.asarray(y_source)
    _validate_arrays(X_source, y_source, role="source")
    y_source_binary = binaryize_survey_labels(y_source, spec)
    source_ids = _indices_or_default(source_indices, len(X_source), role="source")

    if spec.has_official_test:
        if X_test is None or y_test is None:
            raise ValueError(f"{spec.name} requires its official X_test and y_test split.")
        X_test_array = np.asarray(X_test)
        y_test_array = np.asarray(y_test)
        _validate_arrays(X_test_array, y_test_array, role="test")
        if X_test_array.shape[1:] != X_source.shape[1:]:
            raise ValueError("source and official test sample shapes must match.")
        y_test_binary = binaryize_survey_labels(y_test_array, spec)
        default_test_ids = np.arange(len(X_source), len(X_source) + len(X_test_array))
        final_test_ids = _indices_or_default(
            test_indices,
            len(X_test_array),
            role="test",
            default=default_test_ids,
        )
        development_positions = np.arange(len(X_source))
        final_test_positions = np.arange(len(X_test_array))
        test_source = "official"
    else:
        if X_test is not None or y_test is not None or test_indices is not None:
            raise ValueError(f"{spec.name} derives test from source; do not pass test arrays.")
        development_positions, final_test_positions = _stratified_split(
            np.arange(len(X_source)),
            y_source_binary,
            test_size=0.2,
            seed=seed,
            stage="derived test",
        )
        X_test_array = X_source
        y_test_binary = y_source_binary
        final_test_ids = source_ids
        test_source = "stratified_source_20_percent"

    development_labels = y_source_binary[development_positions]
    train_positions, validation_positions = _stratified_split(
        development_positions,
        development_labels,
        test_size=0.1,
        seed=seed,
        stage="validation pool",
    )
    validation_labels = y_source_binary[validation_positions]
    pu_val_positions, clean_val_positions = _stratified_split(
        validation_positions,
        validation_labels,
        test_size=0.5,
        seed=seed,
        stage="PU/clean validation",
    )

    bundle = DatasetBundle(
        train=_part(X_source, y_source_binary, source_ids, train_positions),
        pu_val=_part(X_source, y_source_binary, source_ids, pu_val_positions),
        clean_val=_part(X_source, y_source_binary, source_ids, clean_val_positions),
        test=DatasetPart(
            X=X_test_array[final_test_positions],
            labels=y_test_binary[final_test_positions],
            view="clean",
            indices=final_test_ids[final_test_positions],
            for_selection=False,
        ),
    )
    validate_bundle(bundle)

    role_indices = {
        role: _json_indices(getattr(bundle, role).indices)
        for role in ("train", "pu_val", "clean_val", "test")
    }
    manifest = {
        "schema_version": "1.0",
        "dataset": spec.name,
        "modality": spec.modality,
        "seed": int(seed),
        "positive_classes": list(spec.positive_classes),
        "negative_classes": list(spec.negative_classes),
        "test_source": test_source,
        "split_policy": "stratified_90_train_5_pu_val_5_clean_val",
        "role_sizes": {role: len(indices) for role, indices in role_indices.items()},
        "role_positive_rates": {
            role: float(np.mean(getattr(bundle, role).labels))
            for role in ("train", "pu_val", "clean_val", "test")
        },
        "indices": role_indices,
        # The same helper the transfer verifier recomputes with, so a
        # recomputation can never disagree with what was written here for a
        # reason other than the data having changed.
        "indices_sha256": canonical_hash(role_indices),
        # Protocol §3.1 defines the population prior as the positive rate of
        # the complete binarised pool *before* the stratified split, calls it
        # data-generation metadata, and forbids back-inferring it from any
        # subset.  The full pool is only visible here, so this is the only
        # place it can be recorded rather than reconstructed.  The unlabelled
        # rate is deliberately absent: it is a property of a run's label view,
        # not of the split, and recording it here would invite reading a
        # c-dependent number as a dataset constant.
        "class_prior": {
            "population": float(np.mean(y_source_binary)),
            "population_basis": "binarised source pool before the stratified split",
            "train": float(np.mean(bundle.train.labels)),
        },
        "provenance": _provenance_block(spec, download),
    }
    return bundle, manifest


#: The fields a local download record must carry to enter a split manifest.
#: Everything else in ``data/raw/*/provenance.json`` is either already in the
#: reviewed catalog or free prose; these three are what this machine fetched.
_DOWNLOAD_FIELDS = ("sha256", "bytes", "downloaded_at")
#: How the local record may name the URL the bytes actually came from.  The
#: three files spell it differently, and where a mirror was used that is the
#: URL that answers for the digest.
_RETRIEVAL_KEYS = ("mirror_url", "source_url", "canonical_source_url")


def _provenance_block(spec: SurveyDatasetSpec, download: dict[str, Any] | None) -> dict[str, Any]:
    """Assemble the split manifest's provenance block (P1.2).

    The publishing facts come from the catalog, so every machine that prepares
    this dataset reports the same ones; the download record is passed in
    because it describes one particular fetch.  A dataset with no reviewed
    facts records that instead of an empty block, which would read as an
    answer -- and still reports its download, because what this machine
    verified about the bytes does not depend on anyone having reviewed the
    source.

    ``positive_classes``/``negative_classes`` stay where they are, at the top
    level of the manifest; ``label_semantics`` says what those ids mean.
    """
    if spec.provenance is None:
        return {
            "status": "not_recorded",
            "reason": "no reviewed provenance for this dataset",
            "download": _download_block(download),
        }
    provenance = spec.provenance
    return {
        "status": "recorded",
        "source_url": provenance.source_url,
        "version": provenance.version,
        "citation": provenance.citation,
        "label_semantics": provenance.label_semantics,
        "license": {
            "status": "stated" if provenance.license_name else "not_stated_by_source",
            "name": provenance.license_name,
            "url": provenance.license_url,
            "source_url": provenance.source_url,
            "checked_on": provenance.license_checked_on,
            "notes": provenance.license_notes,
        },
        "download": _download_block(download),
    }


def _download_block(download: dict[str, Any] | None) -> dict[str, Any]:
    if download is None:
        return {"status": "not_recorded"}
    missing = [name for name in _DOWNLOAD_FIELDS if name not in download]
    if missing:
        raise ValueError(f"download record is missing {missing}")
    block = {name: download[name] for name in _DOWNLOAD_FIELDS}
    # Which URL the bytes came from is not decoration: a manifest that prints a
    # landing page beside a digest taken from a mirror invites the reader to
    # recompute the digest from the wrong host.
    for key in _RETRIEVAL_KEYS:
        if download.get(key):
            block["retrieved_from"] = download[key]
            break
    return block


def _resolve_spec(dataset: str | SurveyDatasetSpec) -> SurveyDatasetSpec:
    if isinstance(dataset, SurveyDatasetSpec):
        return dataset
    canonical = _ALIASES.get(dataset, dataset)
    try:
        return _CATALOG[canonical]
    except KeyError as exc:
        choices = sorted(_CATALOG)
        raise ValueError(f"unknown survey dataset {dataset!r}; choose from {choices}.") from exc


def _validate_arrays(X: np.ndarray, y: np.ndarray, *, role: str) -> None:
    if X.ndim < 2:
        raise ValueError(f"{role} features must be batch-first with ndim >= 2.")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"{role} features and labels must align.")


def _indices_or_default(
    indices: np.ndarray | None,
    size: int,
    *,
    role: str,
    default: np.ndarray | None = None,
) -> np.ndarray:
    result = np.asarray(default if indices is None and default is not None else indices)
    if indices is None and default is None:
        result = np.arange(size)
    if result.ndim != 1 or len(result) != size:
        raise ValueError(f"{role}_indices must be one-dimensional and align with its data.")
    if len(set(result.tolist())) != size:
        raise ValueError(f"{role}_indices must be unique.")
    return result


def _stratified_split(
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    test_size: float,
    seed: int,
    stage: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        left, right = train_test_split(
            positions,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    except ValueError as exc:
        raise ValueError(
            f"cannot create the stratified {stage} split; provide more samples per class."
        ) from exc
    return np.sort(left), np.sort(right)


def _part(
    X: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    positions: np.ndarray,
) -> DatasetPart:
    return DatasetPart(
        X=X[positions],
        labels=labels[positions],
        view="clean",
        indices=indices[positions],
    )


def _json_indices(indices: np.ndarray) -> list[int | str]:
    return [item.item() if isinstance(item, np.generic) else item for item in indices]
