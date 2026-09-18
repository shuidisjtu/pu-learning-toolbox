# ruff: noqa: N803, N806
"""Comparison pre-registration: fail-closed loader, unit contract and verdicts."""

import copy
import json

import pytest

from pu_toolbox.experiment.survey_comparison import (
    ELIGIBILITY_CLASSES,
    comparison_digest,
    evaluate_anchor,
    load_comparison_protocol,
    resolve_comparison_unit,
)
from pu_toolbox.experiment.survey_protocol import digest

pytestmark = pytest.mark.unit


def _survey_payload():
    """Minimal stand-in for the execution protocol; the real one is bound at runtime."""
    return {"protocol_version": "survey-v1.2", "review_status": "accepted"}


def _source(**overrides):
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


def _anchor(**overrides):
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


def _mapping(**overrides):
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


def _payload(*, survey=None, **overrides):
    survey = _survey_payload() if survey is None else survey
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
        "sources": [_source()],
        "anchors": [_anchor()],
        "mappings": [_mapping()],
        "contradictions": [],
        "formal_blockers": ["collaborator_review"],
    }
    payload.update(overrides)
    return payload


def _write(tmp_path, payload, name="survey_comparison_v1.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _rejected(tmp_path, payload, match, *, survey=None):
    """Loading must fail with `match`, proving this rule (not a neighbour) rejected it."""
    survey = _survey_payload() if survey is None else survey
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match=match):
        load_comparison_protocol(path, survey=survey)


def _result(**overrides):
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


def _resolved(payload):
    return resolve_comparison_unit(_result()["result_identity"], comparison=payload)


# --- 1/15/16. load, digest, review gating and protocol binding ----------------


def test_basic_protocol_loads_and_reports_status(tmp_path):
    loaded = load_comparison_protocol(_write(tmp_path, _payload()), survey=_survey_payload())
    assert loaded["comparison_version"] == "survey-comparison-v1"
    assert loaded["review_status"] == "pending_collaborator_review"

    accepted = _payload(review_status="accepted", formal_blockers=[])
    loaded = load_comparison_protocol(_write(tmp_path, accepted), survey=_survey_payload())
    assert loaded["review_status"] == "accepted"


def test_determ_comparison_digest_is_key_order_independent():
    """The digest hashes canonical JSON, so serialization order cannot move it."""
    payload = _payload()
    reordered = dict(reversed(list(payload.items())))
    assert comparison_digest(payload) == comparison_digest(reordered)
    changed = _payload(review_status="changes_requested")
    assert comparison_digest(payload) != comparison_digest(changed)


def test_edge_binding_and_review_gates_fail_loud(tmp_path):
    payload = _payload()
    payload["bound_survey_protocol"]["protocol_sha256"] = "b" * 64
    _rejected(tmp_path, payload, "sha256")
    version_mismatch = _payload()
    version_mismatch["bound_survey_protocol"]["protocol_version"] = "survey-v1.1"
    _rejected(tmp_path, version_mismatch, "protocol_version")

    pending = _payload(review_status="accepted")
    pending["anchors"][0]["review_state"] = "pending_review"
    _rejected(tmp_path, pending, "review_state")


# --- 2. eligibility_classes is pinned, not free text -------------------------


def test_edge_eligibility_classes_must_match_the_spec_set_exactly(tmp_path):
    assert ELIGIBILITY_CLASSES == (
        "numeric",
        "magnitude_and_trend",
        "background_only",
        "no_direct_anchor",
        "blocked_pending_pa_criterion",
        "non_runnable",
    )
    extra = _payload(eligibility_classes=[*ELIGIBILITY_CLASSES, "looks_close"])
    _rejected(tmp_path, extra, "eligibility_classes")
    missing = _payload(eligibility_classes=list(ELIGIBILITY_CLASSES[:-1]))
    _rejected(tmp_path, missing, "eligibility_classes")


# --- 3/4/13. identity, containers, fields and enums --------------------------


def test_edge_duplicate_ids_and_wrong_container_types_are_rejected(tmp_path):
    _rejected(tmp_path, _payload(sources=[_source(), _source()]), "source_id")
    _rejected(tmp_path, _payload(anchors=[_anchor(), _anchor()]), "anchor_id")
    _rejected(tmp_path, _payload(mappings=[_mapping(), _mapping()]), "mapping_id")
    _rejected(tmp_path, _payload(sources={"pu_bench_2026": {}}), "sources")
    _rejected(tmp_path, _payload(sources=[_source(), "pu_bench_2026"]), "sources")


def test_edge_missing_fields_unknown_enums_and_bad_units_are_rejected(tmp_path):
    missing = _payload()
    del missing["anchors"][0]["mean_percent"]
    _rejected(tmp_path, missing, "mean_percent")

    bad_enum = _payload()
    bad_enum["mappings"][0]["eligibility"] = "probably_fine"
    _rejected(tmp_path, bad_enum, "eligibility")

    _rejected(tmp_path, _payload(metric_unit="percentish"), "metric_unit")

    for producer in (None, "somebody_else"):
        payload = _payload()
        if producer is None:
            del payload["anchors"][0]["value_producer"]
        else:
            payload["anchors"][0]["value_producer"] = producer
        _rejected(tmp_path, payload, "value_producer")


def test_edge_non_finite_and_out_of_range_anchor_values_are_rejected(tmp_path):
    non_finite = _payload()
    non_finite["anchors"][0]["mean_percent"] = float("nan")
    path = _write(tmp_path, non_finite)
    with pytest.raises(ValueError):
        load_comparison_protocol(path, survey=_survey_payload())

    over = _payload()
    over["anchors"][0]["mean_percent"] = 185.3
    _rejected(tmp_path, over, "mean_percent")

    negative_spread = _payload()
    negative_spread["anchors"][0]["spread_percent"] = -0.1
    _rejected(tmp_path, negative_spread, "spread_percent")


# --- 5/6. dangling references and numeric prerequisites ----------------------


def test_edge_dangling_anchor_references_and_numeric_prerequisites(tmp_path):
    dangling = _payload(mappings=[_mapping(anchor_ids=["missing_anchor"])])
    _rejected(tmp_path, dangling, "anchor")
    _rejected(tmp_path, _payload(mappings=[_mapping(anchor_ids=[])]), "anchor")
    for field in ("mean_percent", "spread_percent", "n_repeats"):
        payload = _payload()
        del payload["anchors"][0][field]
        _rejected(tmp_path, payload, field)


# --- 8/9/10. unit contract and the two uncertainty branches ------------------


def test_basic_fraction_result_is_converted_to_percentage_points():
    verdict = evaluate_anchor(_result(), _resolved(_payload()))
    assert verdict["mean_ours_pp"] == pytest.approx(85.3)
    assert verdict["spread_ours_pp"] == pytest.approx(2.1)


def test_basic_pooled_se_divides_for_std_and_not_for_sem():
    """The two branches must differ, or the implementation silently collapsed."""
    verdict_std = evaluate_anchor(_result(mean=0.893), _resolved(_payload()))
    # sqrt(2.63^2/10 + 2.1^2/5)
    assert verdict_std["se_pooled_pp"] == pytest.approx(1.254468, abs=1e-6)
    assert verdict_std["limit_pp"] == pytest.approx(3.0)

    sem_payload = _payload()
    sem_payload["anchors"][0]["uncertainty_kind"] = "sem"
    verdict_sem = evaluate_anchor(_result(mean=0.893), _resolved(sem_payload))
    # sqrt(2.63^2 + 2.1^2/5) -- the published spread already is the standard error
    assert verdict_sem["se_pooled_pp"] == pytest.approx(2.792651, abs=1e-6)
    assert verdict_sem["limit_pp"] == pytest.approx(5.585302, abs=1e-6)


def test_basic_threshold_outcome_differs_between_uncertainty_kinds():
    """delta_pp == 4.0 sits between the two limits; the branch must decide."""
    assert evaluate_anchor(_result(mean=0.893), _resolved(_payload()))["status"] == ("investigate")
    sem_payload = _payload()
    sem_payload["anchors"][0]["uncertainty_kind"] = "sem"
    assert evaluate_anchor(_result(mean=0.893), _resolved(sem_payload))["status"] == ("consistent")


def test_basic_error_rate_anchor_requires_an_explicit_conversion(tmp_path):
    converted = _payload()
    converted["anchors"][0].update(
        {
            "metric": "error_rate",
            "mean_percent": 19.0,
            "spread_percent": 1.4,
            "metric_conversion": {
                "to": "accuracy",
                "rule": "accuracy = 100 - error_rate",
                "mean_percent": 81.0,
                "spread_percent": 1.4,
            },
        }
    )
    verdict = evaluate_anchor(_result(mean=0.812), _resolved(converted))
    assert verdict["mean_anchor_pp"] == pytest.approx(81.0)

    unreported = _payload()
    unreported["anchors"][0]["metric"] = "error_rate"
    _rejected(tmp_path, unreported, "metric_conversion")


# --- 7/11/14. verdict discipline ---------------------------------------------


def test_edge_verdicts_stay_within_the_numeric_contract():
    payload = _payload()
    payload["mappings"][0]["eligibility"] = "magnitude_and_trend"
    with pytest.raises(ValueError, match="numeric"):
        evaluate_anchor(_result(), _resolved(payload))

    verdict = evaluate_anchor(_result(mean=0.70), _resolved(_payload()))
    assert verdict["status"] == "investigate"
    assert "implementation_error" not in verdict
    assert "conclusion" not in verdict


def test_edge_uncertain_provenance_cannot_be_numeric(tmp_path):
    payload = _payload()
    payload["anchors"][0]["protocol_provenance"] = "uncertain"
    _rejected(tmp_path, payload, "protocol_provenance")


# --- 12. contradiction record ------------------------------------------------


def test_edge_contradictions_must_bind_a_commit_and_resolve_to_code(tmp_path):
    base = _payload()
    # Downgrade the mapping so the uncertain anchor is admissible: the point of
    # this test is the contradiction record, not the numeric gate (covered by
    # test_edge_uncertain_provenance_cannot_be_numeric).
    base["anchors"][0]["protocol_provenance"] = "uncertain"
    base["mappings"][0]["eligibility"] = "magnitude_and_trend"

    without_commit = copy.deepcopy(base)
    without_commit["contradictions"] = [
        {
            "contradiction_id": "c1",
            "source_id": "pu_bench_2026",
            "field": "validation_policy",
            "paper_value": "macro-F1",
            "code_value": "val_proxy_acc",
            "resolution": "code",
            "impact": "cannot attribute the published numbers",
            "affected_anchors": ["pu_bench_table1_nnpu_cifar10_c01_accuracy"],
            "review_state": "accepted",
        }
    ]
    _rejected(tmp_path, without_commit, "code_commit")

    wrong_resolution = copy.deepcopy(without_commit)
    wrong_resolution["contradictions"][0]["code_commit"] = "a" * 40
    wrong_resolution["contradictions"][0]["resolution"] = "pending"
    _rejected(tmp_path, wrong_resolution, "resolution")

    dangling = copy.deepcopy(without_commit)
    dangling["contradictions"][0]["code_commit"] = "a" * 40
    dangling["contradictions"][0]["affected_anchors"] = ["nope"]
    _rejected(tmp_path, dangling, "affected_anchors")
