# ruff: noqa: N803, N806

import numpy as np
import pytest

from pu_toolbox.experiment import (
    binaryize_survey_labels,
    prepare_survey_dataset,
    survey_dataset_catalog,
)
from pu_toolbox.experiment.bundle import validate_bundle

pytestmark = pytest.mark.unit


def _numeric_arrays(n_samples=400, n_features=3, n_classes=2):
    rng = np.random.default_rng(12)
    return rng.normal(size=(n_samples, n_features)), np.arange(n_samples) % n_classes


def test_basic_catalog_contains_locked_eight_datasets():
    catalog = survey_dataset_catalog()
    assert set(catalog) == {
        "adni",
        "cifar10",
        "connect_4",
        "fashion_mnist",
        "imdb",
        "mnist",
        "spambase",
        "twenty_newsgroups",
    }
    assert catalog["mnist"].positive_classes == (0, 2, 4, 6, 8)
    assert catalog["spambase"].has_official_test is False


@pytest.mark.parametrize(
    ("dataset", "labels", "expected"),
    [
        ("f-mnist", np.arange(10), [1, 0, 1, 1, 1, 0, 1, 0, 0, 0]),
        ("cifar10", np.arange(10), [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]),
        ("connect-4", np.array(["win", "loss", "draw"]), [1, 0, 0]),
        ("20news", np.arange(7), [1, 1, 1, 1, 0, 0, 0]),
    ],
)
def test_param_binary_mapping_matches_protocol(dataset, labels, expected):
    assert binaryize_survey_labels(labels, dataset).tolist() == expected


def test_determ_official_test_split_is_reproducible_and_disjoint():
    X_source, y_source = _numeric_arrays(n_classes=10)
    X_test, y_test = _numeric_arrays(n_samples=100, n_classes=10)
    first, first_manifest = prepare_survey_dataset(
        X_source,
        y_source,
        dataset="mnist",
        seed=7,
        X_test=X_test,
        y_test=y_test,
    )
    second, second_manifest = prepare_survey_dataset(
        X_source,
        y_source,
        dataset="mnist",
        seed=7,
        X_test=X_test,
        y_test=y_test,
    )

    validate_bundle(first)
    assert first_manifest == second_manifest
    assert first_manifest["role_sizes"] == {
        "train": 360,
        "pu_val": 20,
        "clean_val": 20,
        "test": 100,
    }
    assert first_manifest["test_source"] == "official"
    assert np.array_equal(first.train.indices, second.train.indices)
    assert min(first.test.indices) == len(X_source)


def test_basic_derived_test_builds_four_way_bundle():
    X, y = _numeric_arrays()
    bundle, manifest = prepare_survey_dataset(X, y, dataset="spambase", seed=3)

    validate_bundle(bundle)
    assert manifest["role_sizes"] == {
        "train": 288,
        "pu_val": 16,
        "clean_val": 16,
        "test": 80,
    }
    assert manifest["test_source"] == "stratified_source_20_percent"
    assert len(manifest["indices_sha256"]) == 64
    assert all(0.4 <= rate <= 0.6 for rate in manifest["role_positive_rates"].values())


def test_edge_rejects_unknown_labels_missing_test_and_tiny_splits():
    X, _ = _numeric_arrays(n_samples=20)
    with pytest.raises(ValueError, match="outside its locked binary mapping"):
        binaryize_survey_labels(np.array([0, 1, 7]), "20news")
    with pytest.raises(ValueError, match="official X_test"):
        prepare_survey_dataset(X, np.arange(20) % 10, dataset="mnist", seed=0)
    with pytest.raises(ValueError, match="provide more samples"):
        prepare_survey_dataset(X, np.arange(20) % 2, dataset="spambase", seed=0)
