# ruff: noqa: N803, N806
"""Comparison pre-registration: fail-closed loader, review gating and binding.

The verdict arithmetic lives in ``test_survey_comparison_verdict.py``: this
module establishes what a well-formed matrix *is*, so a change here must
never be able to move a number.
"""

import copy

import pytest
from _survey_comparison_helpers import (
    anchor as _anchor,
)
from _survey_comparison_helpers import (
    mapping as _mapping,
)
from _survey_comparison_helpers import (
    payload as _payload,
)
from _survey_comparison_helpers import (
    rejected as _rejected,
)
from _survey_comparison_helpers import (
    source as _source,
)
from _survey_comparison_helpers import (
    survey_payload as _survey_payload,
)
from _survey_comparison_helpers import (
    write as _write,
)

from pu_toolbox.experiment.survey_comparison import (
    ELIGIBILITY_CLASSES,
    comparison_digest,
    load_comparison_protocol,
)

pytestmark = pytest.mark.unit


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

    pending = _payload(review_status="accepted", formal_blockers=[])
    pending["anchors"][0]["review_state"] = "pending_review"
    _rejected(tmp_path, pending, "review_state")

    blocked = _payload(review_status="accepted")
    _rejected(tmp_path, blocked, "formal_blockers")


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


def test_edge_accepted_comparison_requires_reviewed_contradictions(tmp_path):
    payload = _payload(review_status="accepted", formal_blockers=[])
    payload["anchors"][0]["protocol_provenance"] = "uncertain"
    payload["mappings"][0]["eligibility"] = "magnitude_and_trend"
    payload["contradictions"] = [
        {
            "contradiction_id": "c1",
            "source_id": "pu_bench_2026",
            "field": "validation_policy",
            "paper_value": "macro-F1",
            "code_value": "val_proxy_acc",
            "code_commit": "a" * 40,
            "resolution": "code",
            "impact": "published selection protocol is uncertain",
            "affected_anchors": ["pu_bench_table1_nnpu_cifar10_c01_accuracy"],
            "review_state": "pending_review",
        }
    ]
    _rejected(tmp_path, payload, "contradiction review_state")

    payload["contradictions"][0]["review_state"] = "accepted"
    loaded = load_comparison_protocol(_write(tmp_path, payload), survey=_survey_payload())
    assert loaded["review_status"] == "accepted"

    payload["contradictions"][0]["review_state"] = "unknown"
    _rejected(tmp_path, payload, "contradiction review_state")
