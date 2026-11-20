"""Contract tests for the shared file-based CLI data readers."""

import numpy as np
import pytest

from pu_toolbox.cli._data import _load_features, _load_label_column, _read_csv

pytestmark = pytest.mark.unit


def test_basic_csv_features_and_labels(tmp_path):
    features = tmp_path / "X.csv"
    labels = tmp_path / "y.csv"
    features.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    labels.write_text("label\n1\n0\n", encoding="utf-8")
    np.testing.assert_array_equal(_load_features(features), [[1, 2], [3, 4]])
    np.testing.assert_array_equal(_load_label_column(labels, "labels"), [1, 0])
    assert list(_read_csv(features, "data").columns) == ["a", "b"]


def test_param_image_features_are_four_dimensional_float32(tmp_path):
    path = tmp_path / "images.npy"
    image = np.arange(24).reshape(2, 3, 2, 2)
    np.save(path, image)
    loaded = _load_features(path)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, image)


def test_edge_bad_csv_and_missing_file(tmp_path):
    headerless = tmp_path / "no_header.csv"
    headerless.write_text("1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        _load_features(headerless)
    wide = tmp_path / "wide.csv"
    wide.write_text("label,extra\n1,0\n0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="single column"):
        _load_label_column(wide, "labels")
    with pytest.raises(FileNotFoundError, match="labels file not found"):
        _load_label_column(tmp_path / "missing.csv", "labels")


@pytest.mark.parametrize("contents", ["a\nNaN\n", "a\ninf\n"])
def test_edge_nonfinite_features_rejected(tmp_path, contents):
    path = tmp_path / "X.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="NaN or Inf"):
        _load_features(path)


def test_edge_wrong_image_dimensions_rejected(tmp_path):
    path = tmp_path / "images.npy"
    np.save(path, np.ones((2, 3)))
    with pytest.raises(ValueError, match="4-D"):
        _load_features(path)
