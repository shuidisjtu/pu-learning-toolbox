# ruff: noqa: S101
"""Actual reader input bytes have their own identity, separate from archive receipts."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def prep_script():
    path = Path(__file__).resolve().parents[3] / "scripts/prepare_survey_splits.py"
    spec = importlib.util.spec_from_file_location("prep_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "dataset,name",
    [
        ("mnist", "raw_arrays.npz"),
        ("fashion_mnist", "raw_arrays.npz"),
        ("connect_4", "connect-4.data"),
    ],
)
def test_basic_identity_hashes_reader_file_not_archive(prep_script, tmp_path, dataset, name):
    directory = tmp_path / dataset
    directory.mkdir()
    (directory / name).write_bytes(b"actual input")
    (directory / "archive.gz").write_bytes(b"different archive")
    identity = prep_script.source_input_identity(tmp_path, dataset)
    assert identity["files"] == [
        {"relative_path": name, "bytes": 12, "sha256": hashlib.sha256(b"actual input").hexdigest()}
    ]
    assert identity["digest_scope"] == "reader_input_files"
    (directory / name).write_bytes(b"changed input")
    assert prep_script.source_input_identity(tmp_path, dataset)["sha256"] != identity["sha256"]


def _news(root, reverse=False):
    paths = [
        "20news-bydate-train/alt.atheism/2",
        "20news-bydate-train/alt.atheism/10",
        "20news-bydate-test/comp.graphics/1",
    ]
    for relative in reversed(paths) if reverse else paths:
        path = root / "twenty_newsgroups" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode() + b"\xff\r\n")


def test_determ_news_identity_sorted_by_relative_path_and_bytes(prep_script, tmp_path):
    _news(tmp_path / "a")
    _news(tmp_path / "b", reverse=True)
    first = prep_script.source_input_identity(tmp_path / "a", "twenty_newsgroups")
    assert first == prep_script.source_input_identity(tmp_path / "b", "twenty_newsgroups")
    paths = [item["relative_path"] for item in first["files"]]
    assert paths == sorted(paths)
    for item in first["files"]:
        data = (tmp_path / "a/twenty_newsgroups" / item["relative_path"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(data).hexdigest()
        assert item["bytes"] == len(data)
    path = tmp_path / "a/twenty_newsgroups" / paths[0]
    path.write_bytes(b"modified extracted document")
    assert (
        prep_script.source_input_identity(tmp_path / "a", "twenty_newsgroups")["sha256"]
        != first["sha256"]
    )


def test_edge_missing_actual_input_is_not_an_absent_receipt(prep_script, tmp_path):
    with pytest.raises(OSError):
        prep_script.source_input_identity(tmp_path, "mnist")


def test_param_unrelated_dataset_has_no_new_input_identity(prep_script, tmp_path):
    assert prep_script.source_input_identity(tmp_path, "adni") is None


def test_basic_identity_attached_separately_from_encoder(prep_script):
    identity = {"digest_scope": "reader_input_files", "sha256": "a" * 64}
    manifest = {"provenance": {}, "preprocessing": {"kind": "sbert"}}
    prep_script._attach_source_evidence(manifest, "twenty_newsgroups", input_identity=identity)
    assert manifest["source_encoding"]["input_identity"] == identity
    assert manifest["preprocessing"] == {"kind": "sbert"}
