"""P3Mix source-anchor hashes and literature values stay distinct from pilot data."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ANCHORS_PATH = ROOT / "docs/research/pu_survey/data/p3mix_literature_anchors_20261019.json"
EXPECTED_PDF_SHA256 = "8599cffab378b0e3dfb7586a55cef482587b7b29f187619833c00b0545d5f528"
EXPECTED_OPENREVIEW_ID = "NH29920YEmj"

pytestmark = pytest.mark.unit


def test_basic_p3mix_literature_anchors_bind_the_uploaded_pdf_without_approving_results():
    record = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
    source_path = ROOT / record["source"]["file"]

    assert record["schema_version"] == "p3mix-literature-anchors-2"
    assert record["review_status"] == "pending_method_owner_review"
    assert record["source"]["sha256"] == EXPECTED_PDF_SHA256
    assert record["source"]["openreview_id"] == EXPECTED_OPENREVIEW_ID
    assert record["source"]["artifact_retained_in_repository"] is False
    if source_path.is_file():
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert record["source"]["sha256"] == digest
    assert record["scope"].startswith("Source-audit anchors only")
    assert record["references"]["frozen_comparison_matrix_modified"] is False
    assert record["protocol"]["validation_definition"] == {
        "count": 500,
        "sampling_frame": "not specified in the paper",
        "class_composition": "not specified in the paper",
        "selection_use": "not specified in the paper",
    }
    assert any("sampling frame" in caveat for caveat in record["caveats"])
    provenance = record["comparison_baseline_provenance"]
    assert set(provenance["methods"]) == {
        "uPU", "nnPU", "nnPU+mixup", "Self-PU", "PAN", "VPU", "MIXPUL", "PULNS", "Supervised"
    }
    assert provenance["methods"]["uPU"]["repository_as_cited"] == (
        "https://github.com/kiryor/nnPUlearning"
    )
    assert provenance["methods"]["PULNS"]["repository_as_cited"] is None
    assert "in-house Python implementation" in provenance["methods"]["PULNS"][
        "implementation_claim"
    ]
    assert "https://github.com/weitongseu/MixPUL" in provenance["methods"]["MIXPUL"][
        "repository_identity_note"
    ]
    assert "does not establish the exact revision" in provenance["methods"]["MIXPUL"][
        "repository_identity_note"
    ]
    assert provenance["methods"]["Supervised"]["label_budget"].startswith(
        "not sufficiently specified"
    )
    assert len(provenance["unresolved_protocol"]) == 2
    assert record["metric"] == {
        "name": "classification_accuracy",
        "unit": "percent",
        "spread": "standard_deviation",
        "repeats": 5,
    }
    assert len(record["anchors"]) == 24

    by_key = {(row["task"], row["variant"]): row for row in record["anchors"]}
    expected = {
        ("CIFAR-10-1", "P3Mix"): (87.0, 1.1),
        ("CIFAR-10-2", "P3Mix"): (84.3, 0.6),
        ("CIFAR-10-1", "P3Mix-E"): (88.2, 0.4),
        ("CIFAR-10-2", "P3Mix-E"): (84.7, 0.5),
        ("CIFAR-10-1", "P3Mix-C"): (88.7, 0.4),
        ("CIFAR-10-2", "P3Mix-C"): (87.9, 0.5),
        ("CIFAR-10-1", "uPU"): (76.5, 2.5),
        ("CIFAR-10-2", "uPU"): (71.6, 1.4),
        ("CIFAR-10-1", "nnPU"): (84.7, 2.4),
        ("CIFAR-10-2", "nnPU"): (83.7, 0.6),
        ("CIFAR-10-1", "nnPU+mixup"): (87.2, 0.6),
        ("CIFAR-10-2", "nnPU+mixup"): (85.8, 1.2),
        ("CIFAR-10-1", "Self-PU"): (85.1, 0.8),
        ("CIFAR-10-2", "Self-PU"): (83.9, 2.6),
        ("CIFAR-10-1", "PAN"): (87.0, 0.3),
        ("CIFAR-10-2", "PAN"): (82.8, 1.0),
        ("CIFAR-10-1", "VPU"): (86.8, 1.2),
        ("CIFAR-10-2", "VPU"): (82.5, 1.1),
        ("CIFAR-10-1", "MIXPUL"): (87.0, 1.9),
        ("CIFAR-10-2", "MIXPUL"): (87.0, 1.1),
        ("CIFAR-10-1", "PULNS"): (87.2, 0.6),
        ("CIFAR-10-2", "PULNS"): (83.7, 2.9),
        ("CIFAR-10-1", "Supervised"): (91.3, 0.3),
        ("CIFAR-10-2", "Supervised"): (91.3, 0.3),
    }
    assert {key: (row["mean"], row["spread"]) for key, row in by_key.items()} == expected
    for variant in ("P3Mix-E", "P3Mix-C"):
        for task in ("CIFAR-10-1", "CIFAR-10-2"):
            assert len(by_key[(task, variant)]["locations"]) == 2

    for variant in ("uPU", "nnPU", "nnPU+mixup", "Self-PU", "PAN", "VPU", "MIXPUL", "PULNS"):
        for task in ("CIFAR-10-1", "CIFAR-10-2"):
            anchor = by_key[(task, variant)]
            assert anchor["method_type"] == "pu_method_comparison"
            assert len(anchor["locations"]) == 1
            assert anchor["locations"][0]["table"] == 2

    for task in ("CIFAR-10-1", "CIFAR-10-2"):
        supervised = by_key[(task, "Supervised")]
        assert (supervised["mean"], supervised["spread"]) == (91.3, 0.3)
        assert supervised["method_type"] == "fully_supervised_reference"
        assert len(supervised["locations"]) == 1
        assert supervised["locations"][0]["table"] == 2
        assert "label budget" in supervised["protocol_note"]
