"""Tests for the locked PUSB Table 2 dataset loader and sampling audit."""

# Test matrices follow sklearn's conventional X/y names.
# ruff: noqa: N803, N806

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from sklearn.datasets import dump_svmlight_file

from benchmarks.assigned_methods.pusb_table2_data import (
    audit_sampling_schedule,
    load_manifest,
    load_table2_dataset,
)

pytestmark = [pytest.mark.unit, pytest.mark.paper]


def _manifest(path, target, *, sha256, samples=4, features=2, status="locked"):
    document = {
        "schema_version": 1,
        "protocol": "pusb_table2_dataset_lock",
        "status": status,
        "datasets": {
            name: {
                "format": "svmlight",
                "downloads": [{"target": target, "sha256": sha256}],
                "expected_samples": samples,
                "expected_features": features,
                "class_counts": {"negative": 2, "positive": 2},
                "status": "locked",
            }
            for name in (
                "mushrooms",
                "shuttle",
                "pageblocks",
                "usps",
                "connect-4",
                "spambase",
            )
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_basic_loads_hashes_maps_labels_and_applies_official_scaling(tmp_path):
    target = tmp_path / "data.svm"
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 1.0], [4.0, 3.0]])
    dump_svmlight_file(X, np.array([1, 2, 1, 2]), str(target), zero_based=False)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = _manifest(tmp_path / "manifest.json", target.name, sha256=digest)

    loaded_X, y, provenance = load_table2_dataset("mushrooms", tmp_path, manifest_path=manifest)

    np.testing.assert_allclose(loaded_X.max(axis=0), 1.0)
    np.testing.assert_array_equal(y, [0, 1, 0, 1])
    assert provenance["class_counts"] == {"negative": 2, "positive": 2}


def test_edge_hash_mismatch_is_rejected(tmp_path):
    target = tmp_path / "data.svm"
    dump_svmlight_file(np.ones((4, 2)), np.array([1, 2, 1, 2]), str(target))
    manifest = _manifest(tmp_path / "manifest.json", target.name, sha256="0" * 64)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_table2_dataset("mushrooms", tmp_path, manifest_path=manifest)


def test_edge_unlocked_manifest_is_rejected(tmp_path):
    manifest = _manifest(
        tmp_path / "manifest.json", "missing", sha256="0" * 64, status="pending_download"
    )
    with pytest.raises(ValueError, match="not locked"):
        load_manifest(manifest)


def test_determ_sampling_schedule_uses_cross_cell_seed_sequence():
    y = np.tile([0, 1], 2500)
    first = audit_sampling_schedule(
        y,
        initial_seed=10,
        repetitions=3,
        class_priors=(0.5,),
        unlabeled_sizes=(100, 200),
        positive_size=20,
        test_size=100,
        holdout_size=1000,
    )
    second = audit_sampling_schedule(
        y,
        initial_seed=10,
        repetitions=3,
        class_priors=(0.5,),
        unlabeled_sizes=(100, 200),
        positive_size=20,
        test_size=100,
        holdout_size=1000,
    )
    assert first == second
    assert all(row["strictly_feasible_repetitions"] == 3 for row in first)


def test_edge_sampling_audit_reports_official_silent_truncation():
    y = np.r_[np.ones(450, dtype=int), np.zeros(3550, dtype=int)]
    rows = audit_sampling_schedule(
        y,
        initial_seed=3,
        repetitions=4,
        class_priors=(0.8,),
        unlabeled_sizes=(800,),
        positive_size=400,
        test_size=1000,
        holdout_size=3000,
    )

    assert rows[0]["strictly_feasible_repetitions"] == 0
    assert rows[0]["minimum_released_test_size"] < 1000
    assert "holdout_pool" in rows[0]["failure_reasons"]
