"""PULNS literature values remain traceable and separate from Pilot results."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ANCHORS_PATH = ROOT / "docs/research/pu_survey/data/pulns_literature_anchors_20261019.json"

pytestmark = pytest.mark.contract


def test_basic_pulns_table3_anchors_preserve_source_protocol_and_unit_ambiguity():
    record = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))

    assert record["schema_version"] == "pulns-literature-anchors-1"
    assert record["review_status"] == "pending_method_owner_review"
    assert record["source"]["doi"] == "10.1609/aaai.v35i10.17064"
    assert record["source"]["official_pdf_url"].endswith("17064-13-20558-1-2-20210518.pdf")
    assert record["source"]["table"] == 3
    assert record["source"]["printed_page"] == 8788
    assert record["source"]["pdf_page_index"] == 5
    assert record["source"]["content_sha256"] is None
    assert "could not retrieve" in record["source"]["digest_unavailable_reason"]

    protocol = record["protocol"]
    assert (
        protocol["gamma_definition"]
        == "positive fraction within U; not labeled-positive frequency c"
    )
    assert protocol["independent_runs"] == 20
    assert protocol["mean_unit"] == "percent"
    assert protocol["spread_name"] == "standard deviation"
    assert protocol["spread_unit"] == "unresolved_by_paper_caption"
    assert record["interpretation"]["spread_is_standard_deviation"] is True
    assert record["interpretation"]["spread_unit_claimed_as_percent"] is False
    assert record["interpretation"]["spread_unit_claimed_as_fraction"] is False

    assert len(record["anchors"]) == 21
    by_key = {(row["dataset"], row["gamma"]): row for row in record["anchors"]}
    expected = {
        ("cifar10", 0.2): (7.98, 0.007),
        ("mnist", 0.2): (3.63, 0.006),
        ("mushrooms", 0.2): (3.13, 0.011),
        ("shuttle", 0.2): (2.74, 0.017),
        ("spambase", 0.2): (9.59, 0.076),
        ("usps", 0.2): (5.43, 0.008),
        ("landsat", 0.2): (4.22, 0.026),
        ("cifar10", 0.4): (11.44, 0.011),
        ("mnist", 0.4): (5.27, 0.009),
        ("mushrooms", 0.4): (4.30, 0.024),
        ("shuttle", 0.4): (4.43, 0.016),
        ("spambase", 0.4): (15.70, 0.014),
        ("usps", 0.4): (8.53, 0.010),
        ("landsat", 0.4): (7.03, 0.009),
        ("cifar10", 0.6): (11.73, 0.014),
        ("mnist", 0.6): (6.54, 0.016),
        ("mushrooms", 0.6): (5.72, 0.026),
        ("shuttle", 0.6): (5.39, 0.039),
        ("spambase", 0.6): (22.39, 0.013),
        ("usps", 0.6): (7.71, 0.012),
        ("landsat", 0.6): (8.03, 0.015),
    }
    assert {
        key: (row["mean_error_percent"], row["spread_printed"]) for key, row in by_key.items()
    } == expected

    boundary = record["comparison_boundary"]
    assert boundary["frozen_comparison_matrix_modified"] is False
    assert boundary["pilot_results_modified"] is False
    assert boundary["direct_numeric_comparison_eligible"] is False
