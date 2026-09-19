# ruff: noqa: N803, N806

"""Shared builders for the pre-registered comparison test modules.

Not collected by pytest (``python_files = ["test_*.py"]``).  The loader and
the verdict suites describe the same artifact from two angles and must agree
on what a well-formed matrix looks like, so the builders live here rather
than being copied into each -- a fixture that drifts between the two suites
would let one of them pass against a shape the other forbids.
"""

import json

import pytest

from pu_toolbox.experiment.survey_comparison import (
    ELIGIBILITY_CLASSES,
    load_comparison_protocol,
    resolve_comparison_unit,
)
from pu_toolbox.experiment.survey_protocol import digest


def survey_payload():
    """Minimal stand-in for the execution protocol; the real one is bound at runtime."""
    return {"protocol_version": "survey-v1.2", "review_status": "accepted"}


def source(**overrides):
    source = {
        "source_id": "pu_bench_2026",
        "kind": "paper_and_locked_repository",
        "paper_title": "PU-Bench",
        "paper_version": "v1",
        "repository": "XiXiphus/PU-Bench",
        "repository_commit": "2d95a19eefd72e66ff30128ec2f65e1d1d4cc077",
        "evidence_reference": "review record",
        "notes": "",
    }
    source.update(overrides)
    return source


def anchor(**overrides):
    anchor = {
        "anchor_id": "pu_bench_table1_nnpu_cifar10_c01_accuracy",
        "source_id": "pu_bench_2026",
        "paper_location": {"table": "1", "page": 6},
        "method_identity": "nnpu",
        "dataset": "cifar10",
        "labeling_mechanism": "scar",
        "c_token": "0.1",
        "selection_protocol": "oa",
        "metric": "accuracy",
        "mean_percent": 85.30,
        "spread_percent": 2.63,
        "uncertainty_kind": "std",
        "n_repeats": 10,
        "value_producer": "original_authors",
        "protocol_provenance": "certain",
        "review_state": "accepted",
        "verified_by": ["shuidisjtu"],
    }
    anchor.update(overrides)
    return anchor


def mapping(**overrides):
    mapping = {
        "mapping_id": "nnpu_cifar10_scar_c01_oa_native_cnn",
        "scope": "unit",
        "result_selector": {
            "method": "nnpu",
            "dataset": "cifar10",
            "labeling_mechanism": "scar",
            "c_token": "0.1",
            "selection_protocol": "oa",
            "metric": "accuracy",
            "training_path": "native_cnn",
        },
        "eligibility": "numeric",
        "anchor_ids": ["pu_bench_table1_nnpu_cifar10_c01_accuracy"],
        "protocol_differences": [],
        "rationale": "same dataset, metric and selection semantics",
        "review_state": "accepted",
        "verified_by": ["shuidisjtu"],
    }
    mapping.update(overrides)
    return mapping


def payload(*, survey=None, **overrides):
    survey = survey_payload() if survey is None else survey
    payload = {
        "schema_version": "1.0",
        "comparison_version": "survey-comparison-v1",
        "review_status": "pending_collaborator_review",
        "bound_survey_protocol": {
            "protocol_version": survey["protocol_version"],
            "protocol_sha256": digest(survey),
        },
        "metric_unit": "percentage_points",
        "result_input_unit": "fraction",
        "eligibility_classes": list(ELIGIBILITY_CLASSES),
        "decision_rules": {"limit_floor_pp": 3.0, "limit_multiplier": 2.0},
        "sources": [source()],
        "anchors": [anchor()],
        "mappings": [mapping()],
        "contradictions": [],
        "formal_blockers": ["collaborator_review"],
    }
    payload.update(overrides)
    return payload


def write(tmp_path, payload, name="survey_comparison_v1.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def rejected(tmp_path, payload, match, *, survey=None):
    """Loading must fail with `match`, proving this rule (not a neighbour) rejected it."""
    survey = survey_payload() if survey is None else survey
    path = write(tmp_path, payload)
    with pytest.raises(ValueError, match=match):
        load_comparison_protocol(path, survey=survey)


def result(**overrides):
    result = {
        "result_identity": {
            "method": "nnpu",
            "dataset": "cifar10",
            "labeling_mechanism": "scar",
            "c_token": "0.1",
            "selection_protocol": "oa",
            "metric": "accuracy",
            "training_path": "native_cnn",
        },
        "mean": 0.853,
        "std": 0.021,
        "n_repeats": 5,
        "metric_unit": "fraction",
    }
    result.update(overrides)
    return result


def resolved(payload):
    return resolve_comparison_unit(result()["result_identity"], comparison=payload)
