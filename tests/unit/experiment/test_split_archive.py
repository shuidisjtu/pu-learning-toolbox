# tests/unit/experiment/test_split_archive.py

# ruff: noqa: N803, N806, S101

"""What the transfer index records, and what ``verify`` refuses to accept.

The artifacts here are written by the real producer rather than by hand, so a
test failing means the archive tool and ``prepare_survey_splits`` have drifted
apart -- not that a fixture went stale.

The failures pinned here are the ones a receiver actually meets.  A byte flip
that leaves the size alone is what a bad copy looks like, and it makes numpy
raise from inside a deflate stream -- an exception no tidy tuple anticipates.
A split directory that arrived twice is a landing tree the operator would
otherwise train on without noticing.  An index asked to vouch for archives it
never described is the one check the whole procedure leans on.
"""

import importlib.util
import json
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest

from pu_toolbox.experiment import split_archive
from pu_toolbox.experiment.split_archive import (
    build_index,
    canonical_hash,
    dump_index,
    file_sha256,
    load_index,
    pack,
    verify,
)

pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
MANIFEST = "split_manifest.json"
ROLE_FILES = ("train.npz", "pu_val.npz", "clean_val.npz", "test.npz")
ROLES = ("train", "pu_val", "clean_val", "test")


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prep_script():
    return _load_script("prepare_survey_splits", "prepare_survey_splits.py")


@pytest.fixture
def products(tmp_path, prep_script):
    """Two real spambase splits under ``tmp_path/splits``."""
    root = tmp_path / "splits"
    rng = np.random.RandomState(0)
    X = rng.randn(500, 57)
    y = np.array([1] * 150 + [0] * 350)
    for seed in (0, 1):
        prep_script.prepare_tabular(
            "spambase", X, y, seed=seed, run_dir=root / "spambase" / f"split_{seed}"
        )
    return root


def _unpack(archive_path: Path, destination: Path) -> None:
    """Extract member by member: ``extractall``'s filter argument is 3.12+."""
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())


def _kinds(report: dict) -> list[str]:
    return sorted(problem["kind"] for problem in report["problems"])


# --- what the index records --------------------------------------------------


def test_basic_index_covers_every_file_and_recomputes_the_indices_digest(products):
    index = build_index(products, commit="c" * 40, protocol_version="survey-v1")
    assert set(index["datasets"]["spambase"]) == {"split_0", "split_1"}
    entry = index["datasets"]["spambase"]["split_0"]
    assert set(entry["files"]) == {MANIFEST, *ROLE_FILES}
    manifest = json.loads((products / "spambase/split_0" / MANIFEST).read_text(encoding="utf-8"))
    # Derived from the .npz here, so an index that copied the manifest's own
    # claim would fail this rather than pass it.
    from_npz = {}
    for role in ROLES:
        with np.load(products / f"spambase/split_0/{role}.npz", allow_pickle=False) as payload:
            from_npz[role] = payload["indices"].tolist()
    assert canonical_hash(from_npz) == manifest["indices_sha256"]
    assert entry["indices_sha256"] == manifest["indices_sha256"]
    assert entry["role_sizes"] == manifest["role_sizes"]
    assert entry["manifest_sha256"] == file_sha256(products / "spambase/split_0" / MANIFEST)
    assert index["produced_by_commit"] == "c" * 40


def test_basic_a_packed_delivery_verifies_after_unpacking(products, tmp_path):
    """The receiver's whole trip: unpack the tar, then check against the index."""
    out = tmp_path / "dist"
    dump_index(
        pack(products, out, commit="c" * 40, protocol_version="survey-v1"),
        out / "index.json",
    )
    landing = tmp_path / "landing"
    _unpack(out / "spambase-splits.tar", landing)

    report = verify(landing, load_index(out / "index.json"), archive_dir=out)
    assert report["ok"] is True
    assert report["problems"] == []
    assert report["checked"] == {"splits": 2, "files": 10}


# --- parameter errors --------------------------------------------------------


def test_param_a_dataset_that_is_not_there_is_refused(products):
    with pytest.raises(ValueError, match="no such dataset directory"):
        build_index(products, datasets=["mnist"])


def test_param_a_split_missing_a_role_file_is_refused_at_pack_time(products):
    """An incomplete set must not be certified by an archive of it."""
    (products / "spambase/split_1/test.npz").unlink()
    with pytest.raises(ValueError, match="is missing test.npz"):
        build_index(products)


def test_param_verify_rejects_something_that_is_not_an_index(products):
    with pytest.raises(ValueError, match="not a split artifact index"):
        verify(products, {"note": "custom reference"})
    # Guessing at a newer index would report a delivery as verified against a
    # document this tool did not understand.
    with pytest.raises(ValueError, match="this tool reads"):
        verify(products, {"schema_version": "9.9", "datasets": {}})


# --- the failures the tool exists for ----------------------------------------


def test_edge_a_manifest_beside_the_wrong_data_fails_at_pack_time(products):
    """The runner copies indices_sha256 through unexamined; this is the check."""
    first = products / "spambase/split_0" / MANIFEST
    second = products / "spambase/split_1" / MANIFEST
    first.write_bytes(second.read_bytes())
    with pytest.raises(ValueError, match="not the artifact that was produced"):
        build_index(products)


def test_edge_a_same_size_corruption_is_reported_rather_than_raising(products):
    """A bad copy leaves the size alone, and numpy then raises from inside it."""
    index = build_index(products)
    target = products / "spambase/split_0/train.npz"
    target.write_bytes(b"\x00" * target.stat().st_size)

    report = verify(products, index)
    assert _kinds(report) == ["digest_mismatch", "unreadable_indices"]
    assert report["ok"] is False


def test_edge_every_damaged_product_is_named_not_just_the_first(products):
    """An operator repairing a transfer needs the whole list, not the first."""
    index = build_index(products)
    for name in ("train.npz", "pu_val.npz"):
        target = products / "spambase/split_0" / name
        target.write_bytes(target.read_bytes()[: target.stat().st_size // 2])

    report = verify(products, index)
    # Two findings per file: stopping at the first unreadable product would
    # leave the second one unnamed.
    assert _kinds(report) == [
        "size_mismatch",
        "size_mismatch",
        "unreadable_indices",
        "unreadable_indices",
    ]
    assert report["ok"] is False


def test_edge_an_unreadable_file_is_reported_rather_than_raising(products, monkeypatch):
    """Held open by a sync client, or denied: not readable is not the same as fine."""
    index = build_index(products)

    def deny(path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(split_archive, "file_sha256", deny)
    report = verify(products, index)
    assert set(_kinds(report)) == {"unreadable_file"}
    assert report["ok"] is False


def test_edge_the_landing_tree_must_not_hold_what_the_index_does_not(products):
    """A copy renamed or duplicated in transit is not what was indexed."""
    index = build_index(products)
    shutil.rmtree(products / "spambase/split_1")
    shutil.copytree(products / "spambase/split_0", products / "spambase/split_77")
    (products / "spambase/split_0/notes.txt").write_text("scratch", encoding="utf-8")

    assert _kinds(verify(products, index)) == [
        "missing_split",
        "unexpected_file",
        "unexpected_split",
    ]


def test_edge_an_index_that_describes_nothing_is_not_a_pass(products):
    """Verifying nothing successfully is the one report a receiver must not get.

    Every split on disk also becomes unaccounted for, which is the same
    statement from the other side: nothing in this tree was indexed.
    """
    index = build_index(products)
    index["datasets"] = {}
    report = verify(products, index)
    assert _kinds(report) == ["index_has_no_splits", "unexpected_split", "unexpected_split"]
    assert report["ok"] is False


def test_edge_archives_are_checked_or_the_reason_is_reported(products, tmp_path):
    out = tmp_path / "dist"
    index = pack(products, out, commit="c" * 40)
    archive = out / "spambase-splits.tar"
    archive.write_bytes(archive.read_bytes() + b"\x00")
    assert _kinds(verify(products, index, archive_dir=out)) == ["archive_digest_mismatch"]

    # An index built without archives cannot vouch for one, and saying nothing
    # would report success for the artifact the procedure tells people to trust.
    assert _kinds(verify(products, build_index(products), archive_dir=out)) == [
        "index_has_no_archive"
    ]


# --- determinism -------------------------------------------------------------


def test_determ_two_packs_of_one_tree_produce_the_same_archive_bytes(products, tmp_path):
    first = pack(products, tmp_path / "one", commit="c" * 40)
    second = pack(products, tmp_path / "two", commit="c" * 40)
    assert (
        first["datasets"]["spambase"]["archive"]["sha256"]
        == second["datasets"]["spambase"]["archive"]["sha256"]
    )
    assert first["datasets"]["spambase"] == second["datasets"]["spambase"]


def test_determ_a_different_mtime_and_mode_do_not_change_the_archive(products, tmp_path):
    """The digest is published, so it must describe content, not the copy.

    Without zeroed headers this passes only because both packs read the same
    files in the same second; a second machine's timestamps would publish a
    digest nobody could reproduce.
    """
    first = pack(products, tmp_path / "one", commit="c" * 40)
    rewritten = tmp_path / "rewritten"
    shutil.copytree(products, rewritten)
    for path in rewritten.rglob("*"):
        if path.is_file():
            os.utime(path, (1_000_000, 1_000_000))
            os.chmod(path, 0o600)

    second = pack(rewritten, tmp_path / "two", commit="c" * 40)
    assert (
        first["datasets"]["spambase"]["archive"]["sha256"]
        == second["datasets"]["spambase"]["archive"]["sha256"]
    )
