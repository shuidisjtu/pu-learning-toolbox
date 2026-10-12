# tests/unit/experiment/test_prepare_survey_splits_adni.py
"""The ADNI substitute (Kaggle 4-class Alzheimer MRI) loads and splits like the other images.

The paper's Table B.1 names the classes ``NonDemented (0)``, ``VeryMildDemented (1)``,
``ModerateDemented (2)``, ``MildDemented (3)``; the protocol maps ``0 vs 1,2,3`` so
NonDemented is the positive class.  PU-Bench's own loader picks Moderate+Mild as
positive, which is **not** this protocol, so only its reading and preprocessing are
reused here, never its class lists.
"""

# ruff: noqa: N803, N806, S101

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pu_toolbox.experiment import survey_dataset_catalog

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts/prepare_survey_splits.py"
COUNTS = {"NonDemented": 8, "VeryMildDemented": 6, "ModerateDemented": 3, "MildDemented": 5}


@pytest.fixture(scope="module")
def prep_script():
    spec = importlib.util.spec_from_file_location("prepare_survey_splits_adni", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jpeg(seed: int, size=(40, 30)) -> bytes:
    pixels = np.random.RandomState(seed).randint(0, 256, size=(size[1], size[0]), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_zips(raw_dir: Path, counts=COUNTS, *, junk=True, reverse=False) -> Path:
    """One zip per class as the mirror ships them, with the macOS litter included."""
    target = raw_dir / "adni"
    target.mkdir(parents=True, exist_ok=True)
    for index, (name, count) in enumerate(counts.items()):
        members = [(f"{name}/{i}.jpg", _jpeg(index * 100 + i)) for i in range(count)]
        if junk:
            members += [(f"__MACOSX/{name}/._{i}.jpg", b"not an image") for i in range(count)]
            members += [(f"__MACOSX/._{name}", b"junk"), (f"{name}/notes.txt", b"text")]
        if reverse:
            members.reverse()
        with zipfile.ZipFile(target / f"{name}.zip", "w") as archive:
            for member, data in members:
                archive.writestr(member, data)
    return raw_dir


def test_basic_load_adni_returns_nchw_uint8_with_protocol_class_ids(prep_script, tmp_path):
    raw_dir = _write_zips(tmp_path)

    X, y = prep_script.load_adni(raw_dir, expected_counts=COUNTS)

    assert X.dtype == np.uint8
    assert X.shape == (sum(COUNTS.values()), 1, 128, 128)
    assert [int((y == i).sum()) for i in range(4)] == list(COUNTS.values())


def test_edge_load_adni_ignores_macos_litter_and_non_images(prep_script, tmp_path):
    raw_dir = _write_zips(tmp_path, junk=True)

    X, _ = prep_script.load_adni(raw_dir, expected_counts=COUNTS)

    assert len(X) == sum(COUNTS.values())


@pytest.mark.parametrize("name", list(COUNTS))
def test_param_load_adni_a_class_with_the_wrong_count_is_an_error(prep_script, tmp_path, name):
    raw_dir = _write_zips(tmp_path)

    with pytest.raises(ValueError, match=name):
        prep_script.load_adni(raw_dir, expected_counts={**COUNTS, name: COUNTS[name] + 1})


def test_edge_load_adni_a_missing_archive_is_an_os_error(prep_script, tmp_path):
    raw_dir = _write_zips(tmp_path)
    (raw_dir / "adni" / "MildDemented.zip").unlink()

    with pytest.raises(OSError):
        prep_script.load_adni(raw_dir, expected_counts=COUNTS)


def test_determ_load_adni_does_not_depend_on_member_order(prep_script, tmp_path):
    first = prep_script.load_adni(_write_zips(tmp_path / "a"), expected_counts=COUNTS)
    second = prep_script.load_adni(
        _write_zips(tmp_path / "b", reverse=True), expected_counts=COUNTS
    )

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_basic_prepare_image_splits_adni_without_an_official_test(prep_script, tmp_path):
    rng = np.random.RandomState(0)
    X = rng.randint(0, 256, size=(400, 1, 128, 128), dtype=np.uint8)
    y = np.array([0] * 200 + [1] * 120 + [2] * 20 + [3] * 60)
    run_dir = tmp_path / "split_0"

    prep_script.prepare_image(X, y, seed=0, run_dir=run_dir, dataset="adni")

    manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["positive_classes"] == [0]
    assert manifest["negative_classes"] == [1, 2, 3]
    assert manifest["preprocessing"]["input_size"] == [128, 128]
    assert manifest["preprocessing"]["in_channels"] == 1
    assert np.load(run_dir / "test.npz")["X"].shape[0] == 80


def test_basic_the_adni_catalog_entry_records_what_the_files_really_are():
    provenance = survey_dataset_catalog()["adni"].provenance

    assert provenance is not None
    assert "kaggle.com/datasets/tourist55" in provenance.source_url
    assert "NonDemented" in provenance.label_semantics
    assert "not ADNI" in provenance.label_semantics
    assert len(provenance.version) >= 40


def _receipt(raw_dir: Path, *, files=None, collected_at="2026-10-12T10:00:00+08:00") -> Path:
    """A collector receipt: several archives, each with its own digest."""
    files = files or [
        {"filename": "NonDemented.zip", "sha256": "a" * 64, "bytes": 10},
        {"filename": "MildDemented.zip", "sha256": "b" * 64, "bytes": 5},
    ]
    payload = {"dataset": "adni", "files": files, "mirror_url": "https://example.invalid/m"}
    if collected_at:
        payload["collected_at"] = collected_at
    target = raw_dir / "adni"
    target.mkdir(parents=True, exist_ok=True)
    (target / "provenance.json").write_text(json.dumps(payload), encoding="utf-8")
    return raw_dir


def test_basic_a_multi_archive_receipt_becomes_one_download_record(prep_script, tmp_path):
    record = prep_script.load_download_record(_receipt(tmp_path), "adni")

    assert record["bytes"] == 15
    assert record["downloaded_at"] == "2026-10-12T10:00:00+08:00"
    assert len(record["sha256"]) == 64
    assert record["mirror_url"] == "https://example.invalid/m"


def test_determ_the_aggregate_digest_does_not_depend_on_file_order(prep_script, tmp_path):
    files = [
        {"filename": "NonDemented.zip", "sha256": "a" * 64, "bytes": 10},
        {"filename": "MildDemented.zip", "sha256": "b" * 64, "bytes": 5},
    ]
    first = prep_script.load_download_record(_receipt(tmp_path / "a", files=files), "adni")
    second = prep_script.load_download_record(_receipt(tmp_path / "b", files=files[::-1]), "adni")

    assert first["sha256"] == second["sha256"]


def test_edge_a_receipt_with_a_changed_digest_changes_the_aggregate(prep_script, tmp_path):
    changed = [
        {"filename": "NonDemented.zip", "sha256": "c" * 64, "bytes": 10},
        {"filename": "MildDemented.zip", "sha256": "b" * 64, "bytes": 5},
    ]
    first = prep_script.load_download_record(_receipt(tmp_path / "a"), "adni")
    second = prep_script.load_download_record(_receipt(tmp_path / "b", files=changed), "adni")

    assert first["sha256"] != second["sha256"]


def test_edge_a_multi_archive_receipt_without_a_time_is_a_defect(prep_script, tmp_path):
    with pytest.raises(ValueError, match="collected_at"):
        prep_script.load_download_record(_receipt(tmp_path, collected_at=None), "adni")
