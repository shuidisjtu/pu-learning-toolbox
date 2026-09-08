# tests/unit/experiment/test_prepare_survey_splits.py

# ruff: noqa: N803, N806, S101

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts/prepare_survey_splits.py"


@pytest.fixture(scope="module")
def prep_script():
    spec = importlib.util.spec_from_file_location("prepare_survey_splits", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeEncoder:
    """Deterministic encode for offline tests: 384-d unit vectors."""

    def encode(self, texts, **_kwargs):
        return np.ones((len(texts), 384), dtype=np.float32)


def test_basic_prepare_tabular_writes_scaled_products(prep_script, tmp_path):
    rng = np.random.RandomState(0)
    X = rng.randn(500, 57)
    y = np.array([1] * 150 + [0] * 350)
    run_dir = tmp_path / "split_0"

    prep_script.prepare_tabular("spambase", X, y, seed=0, run_dir=run_dir)

    train = np.load(run_dir / "train.npz")
    assert train["X"].dtype == np.float32
    assert train["X"].shape[-1] == 57
    # z-score applied on the train partition: near-zero mean, unit std.
    assert abs(float(train["X"].mean())) < 1e-4
    assert abs(float(train["X"].std()) - 1.0) < 1e-3
    manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["preprocessing"]["kind"] == "z-score"
    assert len(manifest["preprocessing"]["feature_mean"]) == 57


def test_basic_prepare_image_keeps_uint8_and_records_preprocessing(prep_script, tmp_path):
    rng = np.random.RandomState(0)
    X = rng.randint(0, 256, size=(500, 3, 32, 32), dtype=np.uint8)
    # positive classes {0,1,8,9} vs negative {2-7}: 150 pos + 350 neg.
    y = np.array([0] * 100 + [9] * 50 + [2] * 350)
    X_test = rng.randint(0, 256, size=(8, 3, 32, 32), dtype=np.uint8)
    y_test = np.array([0] * 4 + [2] * 4)
    run_dir = tmp_path / "split_0"

    prep_script.prepare_image(X, y, X_test, y_test, seed=0, run_dir=run_dir)

    train = np.load(run_dir / "train.npz")
    assert train["X"].dtype == np.uint8
    assert train["X"].shape[1:] == (3, 32, 32)
    manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["preprocessing"]["normalization"]["source"] == ("train_only_channel_statistics")
    assert manifest["preprocessing"]["input_size"] == [32, 32]
    assert len(manifest["preprocessing"]["train_data_sha256"]) == 64


def test_basic_prepare_text_writes_sbert_features(prep_script, tmp_path):
    texts_train = [
        f"review {i} good movie" if i % 2 else f"review {i} bad movie" for i in range(40)
    ]
    labels_train = [1 if i % 2 else 0 for i in range(40)]
    texts_test = [f"test review {i}" for i in range(8)]
    labels_test = [1, 0] * 4
    run_dir = tmp_path / "split_0"

    prep_script.prepare_text(
        texts_train,
        labels_train,
        texts_test,
        labels_test,
        seed=0,
        run_dir=run_dir,
        cache_dir=tmp_path / "cache",
        encoder=_FakeEncoder(),
    )

    test = np.load(run_dir / "test.npz")
    assert test["X"].shape == (8, 384)
    manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["preprocessing"]["kind"] == "sbert"
    assert manifest["preprocessing"]["revision"].startswith("1110a243")
