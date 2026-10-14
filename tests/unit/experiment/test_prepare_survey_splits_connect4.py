# ruff: noqa: N803, N806, S101
"""Connect-4's fixed encoding and train-only standardization."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def prep_script():
    path = Path(__file__).resolve().parents[3] / "scripts/prepare_survey_splits.py"
    spec = importlib.util.spec_from_file_location("prep_connect4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(raw_dir, lines):
    directory = raw_dir / "connect_4"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "connect-4.data").write_text("\n".join(lines) + "\n", encoding="ascii")


def test_basic_connect4_fixed_columns_and_raw_labels(prep_script, tmp_path):
    lines = [
        ",".join([symbol] * 42 + [label])
        for symbol, label in zip("box", ("win", "loss", "draw"), strict=True)
    ]
    _write(tmp_path, lines)
    X, y = prep_script.load_connect4(tmp_path)
    assert X.shape == (3, 126)
    assert y.tolist() == ["win", "loss", "draw"]
    for row in range(3):
        np.testing.assert_array_equal(X[row].reshape(42, 3), np.tile(np.eye(3)[row], (42, 1)))


@pytest.mark.parametrize(
    "line",
    [
        ",".join(["b"] * 41 + ["win"]),
        ",".join(["b"] * 43 + ["win"]),
        ",".join(["q"] + ["b"] * 41 + ["win"]),
        ",".join(["b"] * 42 + ["unknown"]),
        "",
    ],
)
def test_param_bad_rows_are_refused_with_line_number(prep_script, tmp_path, line):
    _write(tmp_path, [line])
    with pytest.raises(ValueError, match="line 1"):
        prep_script.load_connect4(tmp_path)


def test_edge_absent_categories_still_have_all_126_columns(prep_script, tmp_path):
    _write(tmp_path, [",".join(["x"] * 42 + ["draw"])])
    X, _ = prep_script.load_connect4(tmp_path)
    assert X.shape == (1, 126)
    assert np.count_nonzero(X) == 42
    assert np.all(X[0, 2::3] == 1)


def test_determ_connect4_train_statistics_and_constant_columns(prep_script, tmp_path):
    lines = []
    for i in range(200):
        cells = ["b"] * 42
        cells[0] = "box"[i % 3]
        cells[41] = "box"[(i // 3) % 3]
        lines.append(",".join(cells + [("win", "loss", "draw")[i % 3]]))
    _write(tmp_path, lines)
    X, y = prep_script.load_connect4(tmp_path)
    for suffix in ("a", "b"):
        prep_script.prepare_tabular("connect_4", X, y, seed=2, run_dir=tmp_path / suffix)
    manifest = json.loads((tmp_path / "a/split_manifest.json").read_text())
    with np.load(tmp_path / "a/train.npz") as train:
        source = X[train["indices"]]
        np.testing.assert_allclose(manifest["preprocessing"]["feature_mean"], source.mean(axis=0))
        expected_std = source.std(axis=0)
        expected_std[expected_std < 1e-12] = 1
        np.testing.assert_allclose(manifest["preprocessing"]["feature_std"], expected_std)
    for role in ("train", "pu_val", "clean_val", "test"):
        with (
            np.load(tmp_path / "a" / f"{role}.npz") as a,
            np.load(tmp_path / "b" / f"{role}.npz") as b,
        ):
            assert a["X"].dtype == np.float32
            assert np.isfinite(a["X"]).all()
            np.testing.assert_array_equal(a["X"], b["X"])
            np.testing.assert_allclose(
                a["X"], (X[a["indices"]] - source.mean(axis=0)) / expected_std, atol=1e-6
            )
    assert manifest["source_encoding"]["cell_values"] == ["b", "o", "x"]
    assert manifest["source_encoding"]["columns"][0:3] == ["a1=b", "a1=o", "a1=x"]
    assert manifest["source_encoding"]["columns"][-3:] == ["g6=b", "g6=o", "g6=x"]
