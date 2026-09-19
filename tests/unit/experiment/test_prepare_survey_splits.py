# tests/unit/experiment/test_prepare_survey_splits.py

# ruff: noqa: N803, N806, S101

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pu_toolbox.experiment import survey_dataset_catalog

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


_DOWNLOAD = {
    "sha256": "b" * 64,
    "bytes": 125537,
    "downloaded_at": "2026-09-08T10:01:00+08:00",
}


def test_basic_download_record_is_read_from_the_raw_directory(prep_script, tmp_path):
    (tmp_path / "spambase").mkdir()
    (tmp_path / "spambase/provenance.json").write_text(json.dumps(_DOWNLOAD), encoding="utf-8")
    assert prep_script.load_download_record(tmp_path, "spambase") == _DOWNLOAD


def test_edge_absent_download_record_is_none_and_a_malformed_one_is_refused(prep_script, tmp_path):
    """Absent and unreadable are different answers, and only one of them is benign."""
    assert prep_script.load_download_record(tmp_path, "spambase") is None
    (tmp_path / "spambase").mkdir()
    (tmp_path / "spambase/provenance.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        prep_script.load_download_record(tmp_path, "spambase")


def test_basic_split_products_carry_the_reviewed_provenance(prep_script, tmp_path):
    rng = np.random.RandomState(0)
    X = rng.randn(500, 57)
    y = np.array([1] * 150 + [0] * 350)

    prep_script.prepare_tabular(
        "spambase", X, y, seed=0, run_dir=tmp_path / "split_0", download=_DOWNLOAD
    )

    manifest = json.loads((tmp_path / "split_0/split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["download"]["sha256"] == _DOWNLOAD["sha256"]
    assert manifest["provenance"]["license"]["name"] == "CC BY 4.0"
    assert manifest["provenance"]["source_url"]


def test_param_a_download_record_belonging_to_another_dataset_is_refused(prep_script, tmp_path):
    """A copied record would otherwise write one dataset's digest into another's manifest."""
    (tmp_path / "connect_4").mkdir()
    (tmp_path / "connect_4/provenance.json").write_text(
        json.dumps(dict(_DOWNLOAD, dataset="spambase")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="download of 'spambase'"):
        prep_script.load_download_record(tmp_path, "connect_4")


def test_basic_every_dataset_the_pipeline_builds_has_reviewed_provenance(prep_script):
    """The pipeline's dataset list and the catalog's reviewed facts must agree.

    ``main`` rejects anything outside this list, so extending the list without
    adding catalog provenance is how P1.2 would regress again -- silently, and
    only in the artifact.
    """
    catalog = survey_dataset_catalog()
    assert prep_script.SPLIT_PIPELINE_DATASETS
    for name in prep_script.SPLIT_PIPELINE_DATASETS:
        assert catalog[name].provenance is not None, name


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
    # Split preparation applies no augmentation: products store raw uint8 and the
    # training pipeline owns augmentation (protocol-locked "none" in the pilot).
    # Recording the estimator-side default here would misdescribe the artifacts.
    assert manifest["preprocessing"]["augmentation"]["train"]["name"] == "none"
    # Statistics must describe THIS split's train partition, not a constant:
    # different seeds draw different train subsets, so the values must track data.
    recorded = manifest["preprocessing"]["normalization"]
    expected_mean = train["X"].astype(np.float32).reshape(len(train["X"]), 3, -1).mean(axis=(0, 2))
    assert np.allclose(recorded["mean"], expected_mean / 255.0, atol=1e-6)


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
    assert manifest["preprocessing"]["normalize_embeddings"] is False
    assert manifest["preprocessing"]["effective_output_normalization"] == "not_l2_unit_norm"
    assert manifest["preprocessing"]["normalization_source"] == "injected_encoder_output"


def test_basic_prepare_text_records_effective_model_pipeline_normalization(
    prep_script, tmp_path, monkeypatch
):
    def fake_pinned_encoder(texts, **_kwargs):
        features = np.full((len(texts), 384), 1 / np.sqrt(384), dtype=np.float32)
        return features, {"encoder_backend": "sentence-transformers", "normalize_embeddings": False}

    monkeypatch.setattr(prep_script, "encode_survey_texts", fake_pinned_encoder)
    prep_script.prepare_text(
        [f"review {i}" for i in range(40)],
        [i % 2 for i in range(40)],
        [f"test {i}" for i in range(8)],
        [i % 2 for i in range(8)],
        seed=0,
        run_dir=tmp_path / "split_0",
        cache_dir=tmp_path / "cache",
    )
    manifest = json.loads((tmp_path / "split_0/split_manifest.json").read_text())
    assert manifest["preprocessing"]["normalize_embeddings"] is False
    assert manifest["preprocessing"]["effective_output_normalization"] == "l2_unit_norm"
    assert manifest["preprocessing"]["normalization_source"] == "model_pipeline_module"


def test_edge_prepare_text_rejects_nonunit_pinned_model_output(prep_script, tmp_path, monkeypatch):
    def fake_pinned_encoder(texts, **_kwargs):
        return np.ones((len(texts), 384), dtype=np.float32), {
            "encoder_backend": "sentence-transformers",
            "normalize_embeddings": False,
        }

    monkeypatch.setattr(prep_script, "encode_survey_texts", fake_pinned_encoder)
    with pytest.raises(ValueError, match="not L2-normalized"):
        prep_script.prepare_text(
            [f"review {i}" for i in range(40)],
            [i % 2 for i in range(40)],
            [f"test {i}" for i in range(8)],
            [i % 2 for i in range(8)],
            seed=0,
            run_dir=tmp_path / "split_0",
            cache_dir=tmp_path / "cache",
        )
    assert not (tmp_path / "split_0/split_manifest.json").exists()
