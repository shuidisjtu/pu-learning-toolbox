"""Public collection is bounded, verified and never overwrites an input."""

import gzip
import hashlib
import importlib.util
import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "collect_data", Path(__file__).parents[3] / "scripts/collect_survey_public_data.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
pytestmark = pytest.mark.unit


def configure(monkeypatch, *, expected):
    monkeypatch.setattr(
        collector,
        "resources",
        lambda name: [("test.zip", "https://example.invalid/archive", "sha256", expected)],
    )
    monkeypatch.setattr(collector, "urlopen", lambda *a, **k: io.BytesIO(b"raw"))


def test_basic_determ_download_receipt_and_verified_reuse(tmp_path, monkeypatch):
    configure(monkeypatch, expected=hashlib.sha256(b"raw").hexdigest())
    receipt = collector.collect("mnist", tmp_path)
    monkeypatch.setattr(collector, "urlopen", lambda *a, **k: pytest.fail("unexpected download"))
    assert collector.collect("mnist", tmp_path) == receipt
    assert receipt["files"][0]["publisher_checksum_verified"]
    assert not receipt["extracted"]
    assert "source_version" not in receipt


def test_param_publisher_mismatch_not_published(tmp_path, monkeypatch):
    configure(monkeypatch, expected="0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        collector.collect("mnist", tmp_path)
    assert not list(tmp_path.rglob("*.zip"))
    assert not list(tmp_path.rglob(".download-*"))


def test_edge_local_receipt_is_not_publisher_checksum(tmp_path, monkeypatch):
    configure(monkeypatch, expected=None)
    first = collector.collect("connect_4", tmp_path)
    second = collector.collect("connect_4", tmp_path)
    assert first == second
    assert not second["files"][0]["publisher_checksum_verified"]
    target = tmp_path / "connect_4/test.zip"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="existing file preserved"):
        collector.collect("connect_4", tmp_path)
    assert target.read_bytes() == b"changed"
    assert json.loads((target.parent / "provenance.json").read_text(encoding="utf-8")) == first


def test_basic_adni_substitute_is_pinned_to_a_commit_and_a_digest():
    entries = collector.resources("adni")
    assert [name for name, *_ in entries] == [
        "NonDemented.zip",
        "VeryMildDemented.zip",
        "ModerateDemented.zip",
        "MildDemented.zip",
    ]
    for _, url, algorithm, digest in entries:
        assert algorithm == "sha256"
        assert len(digest) == 64
        assert collector.ADNI_MIRROR_COMMIT in url
        assert "/main/" not in url


def test_edge_a_project_pinned_digest_is_not_reported_as_the_publisher_s(tmp_path, monkeypatch):
    configure(monkeypatch, expected=hashlib.sha256(b"raw").hexdigest())
    receipt = collector.collect("adni", tmp_path)
    record = receipt["files"][0]
    assert record["checksum_source"] == "project_pin"
    assert not record["publisher_checksum_verified"]
    assert record["pinned_sha256"] == hashlib.sha256(b"raw").hexdigest()


def test_param_an_unknown_dataset_is_still_refused():
    with pytest.raises(ValueError, match="unsupported"):
        collector.resources("not_a_dataset")


def test_determ_the_collection_time_is_kept_when_a_verified_download_is_reused(
    tmp_path, monkeypatch
):
    configure(monkeypatch, expected=hashlib.sha256(b"raw").hexdigest())
    first = collector.collect("adni", tmp_path)
    second = collector.collect("adni", tmp_path)

    assert first["collected_at"] == second["collected_at"]
    assert first["mirror_url"].endswith(collector.ADNI_MIRROR_COMMIT)


def test_basic_mnist_collection_receipt_and_offline_partition_read(tmp_path, monkeypatch):
    from pu_toolbox.preprocessing.dataset_fetch import fetch_dataset
    from pu_toolbox.preprocessing.dataset_sources import read_dataset

    images = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 1, 28, 28)
    labels = np.array([3, 8], dtype=np.uint8)
    payloads = {
        "train-images-idx3-ubyte.gz": gzip.compress(
            struct.pack(">IIII", 2051, 2, 28, 28) + images.tobytes(), mtime=0
        ),
        "train-labels-idx1-ubyte.gz": gzip.compress(
            struct.pack(">II", 2049, 2) + labels.tobytes(), mtime=0
        ),
    }
    base_url = "https://example.invalid/mnist/"
    config = {
        "source_id": "mnist",
        "version": "idx-v1",
        "format": "idx_images",
        "options": {"image_shape": [28, 28], "label_values": list(range(10))},
        "partitions": {
            "train": {"features": list(payloads)[0], "labels": list(payloads)[1], "max_rows": 2}
        },
        "artifacts": {
            name: {
                "url": base_url + name,
                "checksum": {"algorithm": "sha256", "digest": hashlib.sha256(data).hexdigest()},
            }
            for name, data in payloads.items()
        },
    }
    requested = []

    def fake_urlopen(url, **kwargs):
        requested.append(url)
        return io.BytesIO(payloads[url.removeprefix(base_url)])

    monkeypatch.setattr("pu_toolbox.preprocessing.dataset_fetch.urlopen", fake_urlopen)
    receipt = fetch_dataset(config, tmp_path / "mnist")
    assert requested == [base_url + name for name in payloads]
    assert receipt["dataset"] == "mnist"
    assert receipt["source_version"] == "idx-v1"
    for record in receipt["files"]:
        name = record["filename"]
        assert record["sha256"] == hashlib.sha256(payloads[name]).hexdigest()
        assert not record["publisher_checksum_verified"]
        assert record["checksum_source"] == "config"
    saved_receipt = (tmp_path / "mnist" / "provenance.json").read_text(encoding="utf-8")
    assert json.loads(saved_receipt) == receipt

    monkeypatch.setattr(
        "pu_toolbox.preprocessing.dataset_fetch.urlopen",
        lambda *a, **k: pytest.fail("unexpected download"),
    )
    config_path = tmp_path / "dataset.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert fetch_dataset(config_path, tmp_path / "mnist") == receipt
    source = read_dataset(config, tmp_path / "mnist", partition="train")
    assert source.features.dtype == np.uint8
    np.testing.assert_array_equal(source.features, images)
    np.testing.assert_array_equal(source.labels, labels)
    assert source.source_id == "mnist@idx-v1"
    assert source.partition == "train"
    assert source.sample_ids.tolist() == ["mnist@idx-v1:train:0", "mnist@idx-v1:train:1"]


def test_edge_generic_fetch_validates_all_entries_before_writing(tmp_path, monkeypatch):
    from pu_toolbox.preprocessing.dataset_fetch import fetch_dataset

    config = {
        "source_id": "example",
        "version": "v1",
        "format": "npz_arrays",
        "partitions": {"train": {}},
        "artifacts": {
            "first": {
                "url": "https://example.invalid/first",
                "checksum": {"algorithm": "sha256", "digest": hashlib.sha256(b"raw").hexdigest()},
            },
            "second": {
                "url": "http://example.invalid/second",
                "checksum": {"algorithm": "sha256", "digest": "0" * 64},
            },
        },
    }
    monkeypatch.setattr(
        "pu_toolbox.preprocessing.dataset_fetch.urlopen",
        lambda *a, **k: pytest.fail("unexpected download"),
    )
    with pytest.raises(ValueError):
        fetch_dataset(config, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_edge_generic_fetch_preserves_mismatched_existing_file(tmp_path, monkeypatch):
    from pu_toolbox.preprocessing.dataset_fetch import fetch_dataset

    target = tmp_path / "data.bin"
    target.write_bytes(b"existing")
    config = {
        "source_id": "example",
        "version": "v1",
        "format": "npz_arrays",
        "partitions": {"train": {}},
        "artifacts": {
            "data.bin": {
                "url": "https://example.invalid/file",
                "checksum": {"algorithm": "md5", "digest": hashlib.md5(b"different").hexdigest()},
            }
        },
    }
    monkeypatch.setattr(
        "pu_toolbox.preprocessing.dataset_fetch.urlopen",
        lambda *a, **k: pytest.fail("unexpected download"),
    )
    with pytest.raises(ValueError, match="existing file preserved"):
        fetch_dataset(config, tmp_path)
    assert target.read_bytes() == b"existing"
    assert not (tmp_path / "provenance.json").exists()


def test_edge_survey_prevalidates_paths_before_creating_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        collector,
        "resources",
        lambda _: [("../escape", "https://example.invalid/file", "sha256", "0" * 64)],
    )
    monkeypatch.setattr(collector, "urlopen", lambda *a, **k: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="path"):
        collector.collect("mnist", tmp_path)
    assert not (tmp_path / "mnist").exists()


@pytest.mark.parametrize(
    "algorithm,digest", [("md5", ""), ("sha256", "x" * 64), ("sha1", "0" * 40)]
)
def test_param_generic_fetch_rejects_invalid_checksum_before_writing(
    tmp_path, monkeypatch, algorithm, digest
):
    from pu_toolbox.preprocessing.dataset_fetch import fetch_dataset

    config = {
        "source_id": "example",
        "version": "v1",
        "format": "npz_arrays",
        "partitions": {"train": {}},
        "artifacts": {
            "data": {
                "url": "https://example.invalid/file",
                "checksum": {"algorithm": algorithm, "digest": digest},
            }
        },
    }
    monkeypatch.setattr(
        "pu_toolbox.preprocessing.dataset_fetch.urlopen",
        lambda *a, **k: pytest.fail("unexpected download"),
    )
    with pytest.raises(ValueError, match="checksum"):
        fetch_dataset(config, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
