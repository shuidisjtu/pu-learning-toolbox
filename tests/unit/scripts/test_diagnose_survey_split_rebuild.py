"""Cross-platform diagnostics cannot turn a changed receipt into acceptance."""

import importlib.util
import json
import shutil
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pu_toolbox.experiment.split_archive import build_index, dump_index
from pu_toolbox.utils.serialization import canonical_hash, file_hash

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def script():
    path = Path(__file__).resolve().parents[3] / "scripts/diagnose_survey_split_rebuild.py"
    spec = importlib.util.spec_from_file_location("rebuild_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tree(tmp_path):
    source, rebuilt = tmp_path / "source", tmp_path / "rebuilt"
    for seed in (0, 1):
        directory = source / "mnist" / f"split_{seed}"
        directory.mkdir(parents=True)
        indices = {}
        for offset, role in enumerate(("train", "pu_val", "clean_val", "test")):
            ids = np.arange(2) + offset * 2
            indices[role] = ids.tolist()
            np.savez_compressed(
                directory / f"{role}.npz",
                X=np.arange(4, dtype=np.float32).reshape(2, 2) + seed,
                y=np.array([0, 1]),
                indices=ids,
            )
        manifest = dict(
            indices_sha256=canonical_hash(indices),
            indices=indices,
            role_sizes={role: 2 for role in indices},
        )
        (directory / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    index_path = tmp_path / "index.json"
    dump_index(build_index(source), index_path)
    shutil.copytree(source, rebuilt)
    return rebuilt, index_path


def change_creator(path, system=0):
    data = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as archive:
        offset = archive.start_dir
        for _ in archive.infolist():
            data[offset + 5] = system
            offset += 46 + sum(struct.unpack_from("<HHH", data, offset + 28))
    path.write_bytes(data)


def test_basic_byte_identical_rebuild_is_not_formal_archive_acceptance(script, tree):
    root, index = tree
    result = script.diagnose(root, index)
    assert result["strict_unpacked_reference_index"]["ok"]
    assert result["summary"]["role_files"] == {"byte_identical": 8}
    assert result["summary"]["role_indices"] == {"identical": 2}
    assert not result["original_archive_receipt_verified"] and not result["formal_acceptance"]


def test_basic_determ_creator_only_diagnosed_without_writing_or_relaxing_cli(script, tree, capsys):
    root, index = tree
    for p in root.glob("*/*/*.npz"):
        change_creator(p)
    before = {str(p): file_hash(p) for p in root.glob("*/*/*")}
    result = script.diagnose(root, index)
    assert not result["strict_unpacked_reference_index"]["ok"]
    assert result["summary"]["role_files"] == {"zip_creator_system_only": 8}
    assert result["summary"]["role_indices"] == {"identical": 2}
    assert not result["files_rewritten"]
    assert result == script.diagnose(root, index)
    assert before == {str(p): file_hash(p) for p in root.glob("*/*/*")}
    assert script.main(["--root", str(root), "--index", str(index)]) == 2
    assert not json.loads(capsys.readouterr().out)["formal_acceptance"]


def test_param_actual_array_changes_cannot_be_explained_by_creator_bytes(script, tree):
    root, index = tree
    p = root / "mnist/split_0/train.npz"
    with np.load(p, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    arrays["X"][0, 0] += 1
    np.savez_compressed(p, **arrays)
    result = script.diagnose(root, index)
    assert result["splits"][0]["files"]["train.npz"]["status"] == "different"
    assert not result["strict_unpacked_reference_index"]["ok"]


def test_edge_missing_whole_split_remains_in_expected_inventory(script, tree):
    root, index = tree
    shutil.rmtree(root / "mnist/split_1")
    result = script.diagnose(root, index)
    assert result["summary"]["expected_splits"] == 2
    assert result["summary"]["role_files"] == {"byte_identical": 4, "missing": 4}
    assert result["summary"]["manifests"] == {"byte_identical": 1, "missing": 1}
    assert result["summary"]["role_indices"]["not_verified"] == 1
    assert not result["strict_unpacked_reference_index"]["ok"]


def test_edge_other_zip_metadata_is_not_hidden(script, tree):
    root, index = tree
    p = root / "mnist/split_0/test.npz"
    raw = bytearray(p.read_bytes())
    with zipfile.ZipFile(p) as z:
        raw[z.start_dir + 12] ^= 1  # date byte, not creator system
    p.write_bytes(raw)
    result = script.diagnose(root, index)
    assert result["splits"][0]["files"]["test.npz"]["status"] == "different"


def test_edge_corrupt_deflate_stream_is_a_finding_not_a_crash(script, tree):
    root, index = tree
    p = root / "mnist/split_0/train.npz"
    raw = bytearray(p.read_bytes())
    name_size, extra_size = struct.unpack_from("<HH", raw, 26)
    raw[30 + name_size + extra_size] ^= 255
    p.write_bytes(raw)
    result = script.diagnose(root, index)
    record = result["splits"][0]["files"]["train.npz"]
    assert record["status"] == "different" and "zip_diagnostic_error" in record


def test_param_uncompressed_diagnostic_size_is_bounded(script, tree, monkeypatch):
    root, index = tree
    p = root / "mnist/split_0/train.npz"
    change_creator(p)
    monkeypatch.setattr(script, "MAX_UNCOMPRESSED_BYTES", 1)
    result = script.diagnose(root, index)
    record = result["splits"][0]["files"]["train.npz"]
    assert record["status"] == "different" and "bounded" in record["zip_diagnostic_error"]


@pytest.mark.parametrize(
    "change", ["dataset_path", "split_path", "extra_file", "bool_size", "empty"]
)
def test_param_malformed_reference_names_and_coverage_fail_before_reads(script, tree, change):
    root, index = tree
    data = json.loads(index.read_text())
    if change == "dataset_path":
        data["datasets"]["../outside"] = data["datasets"].pop("mnist")
    elif change == "split_path":
        data["datasets"]["mnist"]["../outside"] = data["datasets"]["mnist"].pop("split_0")
    elif change == "extra_file":
        data["datasets"]["mnist"]["split_0"]["files"]["../outside"] = {}
    elif change == "bool_size":
        data["datasets"]["mnist"]["split_0"]["files"]["test.npz"]["bytes"] = True
    else:
        data["datasets"] = {}
    index.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        script.diagnose(root, index)


def test_param_symlink_rebuild_file_is_rejected(script, tree):
    root, index = tree
    p = root / "mnist/split_0/test.npz"
    sibling = root / "mnist/split_1/test.npz"
    p.unlink()
    p.symlink_to(sibling)
    with pytest.raises(ValueError, match="symlink"):
        script.diagnose(root, index)


def test_edge_unexpected_split_is_still_a_strict_error(script, tree):
    root, index = tree
    shutil.copytree(root / "mnist/split_0", root / "mnist/split_99")
    result = script.diagnose(root, index)
    assert result["summary"]["role_files"] == {"byte_identical": 8}
    assert not result["strict_unpacked_reference_index"]["ok"]
    assert any(
        p["kind"] == "unexpected_split"
        for p in result["strict_unpacked_reference_index"]["problems"]
    )
