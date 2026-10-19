"""P3Mix source-anchor hashes and literature values stay distinct from pilot data."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ANCHORS_PATH = ROOT / "docs/research/pu_survey/data/p3mix_literature_anchors_20261019.json"


def test_basic_p3mix_literature_anchors_bind_the_uploaded_pdf_without_approving_results():
    record = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
    source_path = ROOT / record["source"]["file"]

    assert record["schema_version"] == "p3mix-literature-anchors-1"
    assert record["review_status"] == "pending_method_owner_review"
    assert len(record["source"]["sha256"]) == 64
    assert record["source"]["artifact_retained_in_repository"] is False
    if source_path.is_file():
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert record["source"]["sha256"] == digest
    assert record["scope"].startswith("Source-audit anchors only")
    assert record["references"]["frozen_comparison_matrix_modified"] is False
    assert record["metric"] == {
        "name": "classification_accuracy",
        "unit": "percent",
        "spread": "standard_deviation",
        "repeats": 5,
    }
    assert len(record["anchors"]) == 6

    by_key = {(row["task"], row["variant"]): row for row in record["anchors"]}
    expected = {
        ("CIFAR-10-1", "P3Mix"): (87.0, 1.1),
        ("CIFAR-10-2", "P3Mix"): (84.3, 0.6),
        ("CIFAR-10-1", "P3Mix-E"): (88.2, 0.4),
        ("CIFAR-10-2", "P3Mix-E"): (84.7, 0.5),
        ("CIFAR-10-1", "P3Mix-C"): (88.7, 0.4),
        ("CIFAR-10-2", "P3Mix-C"): (87.9, 0.5),
    }
    assert {key: (row["mean"], row["spread"]) for key, row in by_key.items()} == expected
    for variant in ("P3Mix-E", "P3Mix-C"):
        for task in ("CIFAR-10-1", "CIFAR-10-2"):
            assert len(by_key[(task, variant)]["locations"]) == 2
