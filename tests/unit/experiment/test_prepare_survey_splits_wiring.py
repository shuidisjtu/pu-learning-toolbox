# ruff: noqa: N803, N806, S101
"""New dataset CLI routing, refusal to overwrite, and independent handoff evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit
NEW = ("mnist", "fashion_mnist", "twenty_newsgroups", "connect_4")


@pytest.fixture(scope="module")
def prep_script():
    path = Path(__file__).resolve().parents[3] / "scripts/prepare_survey_splits.py"
    spec = importlib.util.spec_from_file_location("prep_wiring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipt(raw_dir, dataset="mnist", **changes):
    directory = raw_dir / dataset
    directory.mkdir(parents=True, exist_ok=True)
    data = b"fixture original archive"
    (directory / "source.bin").write_bytes(data)
    payload = {
        "dataset": dataset,
        "files": [
            {
                "filename": "source.bin",
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        ],
        "reviewed_on": "2026-10-14",
        "notes": "download time unavailable",
        **changes,
    }
    (directory / "provenance.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_basic_handoff_preserves_payload_without_inventing_download(prep_script, tmp_path):
    payload = _receipt(tmp_path)
    download, handoff = prep_script.load_source_records(tmp_path, "mnist")
    assert download is None
    assert handoff["receipt"] == payload
    assert (
        handoff["receipt_sha256"]
        == hashlib.sha256((tmp_path / "mnist/provenance.json").read_bytes()).hexdigest()
    )
    assert handoff["receipt_digest_scope"] == "original_receipt_bytes"
    assert handoff["downloaded_at"] is None
    assert handoff["download_time_status"] == "unknown"
    assert prep_script.load_download_record(tmp_path, "mnist") is None
    X = np.arange(80 * 28 * 28, dtype=np.uint8).reshape(80, 1, 28, 28)
    prep_script.prepare_image(
        X,
        np.tile(np.arange(10), 8),
        X[:10],
        np.arange(10),
        dataset="mnist",
        seed=0,
        run_dir=tmp_path / "split",
        raw_handoff=handoff,
    )
    manifest = json.loads((tmp_path / "split/split_manifest.json").read_text())
    assert manifest["provenance"]["download"]["status"] == "not_recorded"
    assert manifest["provenance"]["raw_handoff"] == handoff


@pytest.mark.parametrize(
    "change",
    [
        {"filename": "../source.bin"},
        {"filename": "C:/source.bin"},
        {"filename": "source\\bin"},
        {"sha256": "short"},
        {"sha256": "z" * 64},
        {"bytes": -1},
        {"bytes": True},
        {"bytes": "23"},
        {"sha256": "a" * 64},
        {"bytes": 1},
    ],
)
def test_param_handoff_refuses_invalid_or_mismatched_files(prep_script, tmp_path, change):
    payload = _receipt(tmp_path)
    payload["files"][0].update(change)
    (tmp_path / "mnist/provenance.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        prep_script.load_source_records(tmp_path, "mnist")


@pytest.mark.parametrize(
    "files", [[], {}, [None], [{"filename": "missing", "sha256": "a" * 64, "bytes": 0}]]
)
def test_param_handoff_bad_file_lists_are_not_silently_ignored(prep_script, tmp_path, files):
    _receipt(tmp_path, files=files)
    with pytest.raises(ValueError):
        prep_script.load_source_records(tmp_path, "mnist")


def test_edge_handoff_duplicate_files_are_refused(prep_script, tmp_path):
    payload = _receipt(tmp_path)
    payload["files"] *= 2
    (tmp_path / "mnist/provenance.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate"):
        prep_script.load_source_records(tmp_path, "mnist")


def test_edge_adni_missing_time_remains_an_error(prep_script, tmp_path):
    _receipt(tmp_path, "adni")
    with pytest.raises(ValueError, match="collected_at"):
        prep_script.load_source_records(tmp_path, "adni")


def test_determ_handoff_is_order_stable_and_verifies_actual_bytes(prep_script, tmp_path):
    _receipt(tmp_path)
    first = prep_script.load_source_records(tmp_path, "mnist")
    second = prep_script.load_source_records(tmp_path, "mnist")
    assert first == second
    (tmp_path / "mnist/source.bin").write_bytes(b"fixture modified archive")
    with pytest.raises(ValueError, match="mismatch"):
        prep_script.load_source_records(tmp_path, "mnist")


@pytest.mark.parametrize(
    "dataset,loader",
    [
        ("mnist", "load_mnist_arrays"),
        ("fashion_mnist", "load_mnist_arrays"),
        ("twenty_newsgroups", "load_twenty_newsgroups"),
        ("connect_4", "load_connect4"),
    ],
)
def test_basic_load_dispatch_uses_new_readers(prep_script, monkeypatch, tmp_path, dataset, loader):
    calls = []
    monkeypatch.setattr(prep_script, loader, lambda *args: calls.append(args) or ("loaded",))
    assert prep_script._load_dataset(dataset, tmp_path) == ("loaded",)
    assert (
        calls == [(tmp_path, dataset)] if loader == "load_mnist_arrays" else calls == [(tmp_path,)]
    )


@pytest.mark.parametrize(
    "dataset,prepare",
    [
        ("mnist", "prepare_image"),
        ("fashion_mnist", "prepare_image"),
        ("twenty_newsgroups", "prepare_text"),
        ("connect_4", "prepare_tabular"),
    ],
)
def test_basic_cli_routes_new_datasets_and_forwards_evidence(
    prep_script, monkeypatch, tmp_path, dataset, prepare
):
    calls = []
    handoff = {"receipt": {"files": []}}
    monkeypatch.setattr(prep_script, "load_source_records", lambda *args: (None, handoff))
    monkeypatch.setattr(prep_script, "_load_dataset", lambda *args: ("fixture",))
    identity = {"sha256": "a" * 64}
    monkeypatch.setattr(prep_script, "source_input_identity", lambda *args: identity)
    monkeypatch.setattr(prep_script, prepare, lambda *args, **kwargs: calls.append((args, kwargs)))
    assert (
        prep_script.main(["--datasets", dataset, "--seeds", "0,1", "--out-dir", str(tmp_path)]) == 0
    )
    assert len(calls) == 2
    assert [call[1]["seed"] for call in calls] == [0, 1]
    assert all(call[1]["raw_handoff"] == handoff for call in calls)
    assert all(call[1]["input_identity"] == identity for call in calls)
    if prepare != "prepare_tabular":
        assert all(call[1]["dataset"] == dataset for call in calls)
    else:
        assert calls[0][0][0] == dataset


def test_edge_existing_new_split_is_refused_before_loading(prep_script, monkeypatch, tmp_path):
    directory = tmp_path / "mnist/split_1"
    directory.mkdir(parents=True)
    sentinel = directory / "train.npz"
    sentinel.write_bytes(b"must remain")
    monkeypatch.setattr(
        prep_script, "_load_dataset", lambda *args: pytest.fail("must preflight before loading")
    )
    assert (
        prep_script.main(["--datasets", "mnist", "--seeds", "0,1", "--out-dir", str(tmp_path)]) == 1
    )
    assert sentinel.read_bytes() == b"must remain"
    assert not (tmp_path / "mnist/split_0").exists()


def test_edge_direct_save_refuses_new_dataset_overwrite(prep_script, tmp_path):
    X = np.arange(80 * 28 * 28, dtype=np.uint8).reshape(80, 1, 28, 28)
    kwargs = {"dataset": "mnist", "seed": 0, "run_dir": tmp_path / "split"}
    prep_script.prepare_image(X, np.tile(np.arange(10), 8), X[:10], np.arange(10), **kwargs)
    before = (tmp_path / "split/train.npz").read_bytes()
    with pytest.raises(FileExistsError):
        prep_script.prepare_image(X, np.tile(np.arange(10), 8), X[:10], np.arange(10), **kwargs)
    assert (tmp_path / "split/train.npz").read_bytes() == before


def test_determ_cli_default_dataset_list_remains_pilot(prep_script, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(prep_script, "load_source_records", lambda *args: (None, None))
    monkeypatch.setattr(
        prep_script, "_load_dataset", lambda dataset, *_args: calls.append(dataset) or ()
    )
    for name in ("prepare_image", "prepare_text", "prepare_tabular"):
        monkeypatch.setattr(prep_script, name, lambda *args, **kwargs: None)
    assert prep_script.main(["--seeds", "0", "--out-dir", str(tmp_path)]) == 0
    assert calls == ["spambase", "cifar10", "imdb"]
