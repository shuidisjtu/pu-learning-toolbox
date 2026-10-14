# ruff: noqa: N803, N806, S101
"""Offline MNIST-family source contracts and official test isolation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def prep_script():
    path = Path(__file__).resolve().parents[3] / "scripts/prepare_survey_splits.py"
    spec = importlib.util.spec_from_file_location("prep_mnist", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arrays():
    return {
        "train_images": np.arange(80 * 28 * 28, dtype=np.uint8).reshape(80, 28, 28),
        "train_labels": np.tile(np.arange(10), 8),
        "test_images": np.full((10, 28, 28), 251, dtype=np.uint8),
        "test_labels": np.arange(10),
    }


def _write(raw_dir, dataset, arrays):
    target = raw_dir / dataset
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / "raw_arrays.npz", **arrays)


@pytest.mark.parametrize("dataset", ["mnist", "fashion_mnist"])
def test_basic_load_arrays_preserves_official_pools(prep_script, tmp_path, dataset):
    arrays = _arrays()
    _write(tmp_path, dataset, arrays)
    X, y, Xt, yt = prep_script.load_mnist_arrays(tmp_path, dataset)
    assert X.shape == (80, 1, 28, 28)
    assert X.dtype == np.uint8
    np.testing.assert_array_equal(X[:, 0], arrays["train_images"])
    np.testing.assert_array_equal(y, arrays["train_labels"])
    np.testing.assert_array_equal(Xt[:, 0], arrays["test_images"])
    np.testing.assert_array_equal(yt, arrays["test_labels"])


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("train_images", None, "keys"),
        ("train_images", np.zeros((80, 28, 28), dtype=float), "uint8"),
        ("test_images", np.zeros((10, 1, 28, 28), dtype=np.uint8), "28"),
        ("train_labels", np.tile(np.arange(10, dtype=float), 8), "integer"),
        ("test_labels", np.arange(10)[:, None], "one-dimensional"),
        ("test_labels", np.arange(10) + 1, "0..9"),
        ("test_labels", np.arange(9), "length"),
        ("test_images", np.zeros((0, 28, 28), dtype=np.uint8), "non-empty"),
        ("test_labels", np.array([0] * 9 + [-1]), "0..9"),
        ("train_labels", np.array([True] * 80), "integer"),
    ],
)
def test_param_invalid_arrays_are_refused(prep_script, tmp_path, key, value, match):
    arrays = _arrays()
    if value is None:
        arrays.pop(key)
    else:
        arrays[key] = value
    _write(tmp_path, "mnist", arrays)
    with pytest.raises(ValueError, match=match):
        prep_script.load_mnist_arrays(tmp_path, "mnist")


def test_edge_object_arrays_cannot_load_pickle(prep_script, tmp_path):
    arrays = _arrays()
    arrays["train_images"] = np.array([object()], dtype=object)
    _write(tmp_path, "mnist", arrays)
    with pytest.raises(ValueError, match="allow_pickle"):
        prep_script.load_mnist_arrays(tmp_path, "mnist")


@pytest.mark.parametrize(
    "dataset,positive", [("mnist", {0, 2, 4, 6, 8}), ("fashion_mnist", {0, 2, 3, 4, 6})]
)
def test_determ_image_products_keep_test_mapping_and_seed(prep_script, tmp_path, dataset, positive):
    _write(tmp_path, dataset, _arrays())
    arrays = prep_script.load_mnist_arrays(tmp_path, dataset)
    for suffix in ("a", "b"):
        prep_script.prepare_image(*arrays, dataset=dataset, seed=3, run_dir=tmp_path / suffix)
    for role in ("train", "pu_val", "clean_val", "test"):
        with (
            np.load(tmp_path / "a" / f"{role}.npz") as a,
            np.load(tmp_path / "b" / f"{role}.npz") as b,
        ):
            for key in ("X", "y", "indices"):
                np.testing.assert_array_equal(a[key], b[key])
    with np.load(tmp_path / "a/test.npz") as test:
        np.testing.assert_array_equal(test["X"], arrays[2])
        np.testing.assert_array_equal(test["y"], [int(i in positive) for i in range(10)])
    manifest = json.loads((tmp_path / "a/split_manifest.json").read_text())
    assert manifest["source_encoding"]["input_format"] == "raw_arrays.npz"
    assert manifest["preprocessing"]["in_channels"] == 1
