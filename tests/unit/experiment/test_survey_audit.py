"""The auditor's two contracts: never crash, and never blame one file for a group.

Both are about what a *finding* is allowed to say.

* A malformed manifest must become a finding.  The aggregation entry point raises
  on one, which is correct for an entry point whose whole output is an exit code
  -- but an audit that stopped at the first bad item would report one problem and
  hide the rest, and a traceback is not a finding a reviewer can act on.
* A group-level refusal may not be pinned on a manifest.  The gate raises on the
  first bad group and names the rule, not the file, so a check that picked one
  member would manufacture a culprit.  ``members`` is mandatory for unit- and
  group-scoped checks precisely so that cannot be done quietly.

The identity fields asserted here are the ones a real manifest carries: code
commit and the dependency lock are deliberately absent from the list, because the
survey runner records them per batch rather than per run.
"""

import pytest

from pu_toolbox.experiment.survey_audit import (
    AuditError,
    gate_error_check,
    make_check,
    overall_status,
    preflight_status,
)

pytestmark = pytest.mark.unit


def _complete_manifest():
    """Every field the preflight requires, in the shape a real manifest uses."""
    return {
        "execution_mode": "versioned_pilot",
        "protocol_version": "survey-v1.2",
        "protocol_sha256": "a" * 64,
        "execution_unit": {
            "method": "nnpu",
            "dataset": "spambase",
            "comparability_group": "spambase/native_2d/minibatch",
        },
        "training_path": "native_2d",
        "run_view": "ts-compatible",
        "representation": {
            "name": "tabular_mlp",
            "split_sha256": "b" * 64,
            "feature_sha256": {"train": "c" * 64},
        },
        "seed": 0,
        "split_ref": {"indices_sha256": "d" * 64},
        "generation": {"train": {"mechanism": "scar", "c_requested": 0.1}},
        "candidate_runs": [{"candidate_index": 0}],
        "resources": {
            "environment": {
                "python_version": "3.10.8",
                "torch_version": "2.13.0+cu130",
                "gpu_devices": [{"index": 0, "name": "NVIDIA vGPU-32GB"}],
            }
        },
    }


def _without(manifest, *paths):
    """A copy of *manifest* with each dotted path removed."""
    pruned = {key: value for key, value in manifest.items()}
    for path in paths:
        head, _, tail = path.partition(".")
        if not tail:
            pruned.pop(head, None)
            continue
        pruned[head] = {key: value for key, value in pruned[head].items()}
        pruned[head].pop(tail, None)
    return pruned


def test_basic_a_complete_manifest_passes_the_preflight():
    status, reasons, missing = preflight_status(_complete_manifest())

    assert status == "formal"
    assert reasons == []
    assert missing == []


def test_basic_a_manifest_missing_its_split_digest_is_not_reproducible():
    status, reasons, missing = preflight_status(_without(_complete_manifest(), "split_ref"))

    assert status == "not_reproducible"
    assert reasons == ["missing_manifest_field"]
    assert missing == ["split_ref.indices_sha256"]


def test_basic_a_manifest_missing_environment_identity_is_not_reproducible():
    # Protocol §5 clause 6 asks for Python/PyTorch/CUDA/GPU information, so this
    # is the same kind of finding as a missing split digest.
    status, reasons, _ = preflight_status(_without(_complete_manifest(), "resources.environment"))

    assert status == "not_reproducible"
    assert reasons == ["missing_manifest_field"]


def test_edge_the_missing_field_is_named_rather_than_only_counted():
    manifest = _without(_complete_manifest(), "representation.name", "seed")
    _, _, missing = preflight_status(manifest)

    assert missing == ["representation.name", "seed"]


def test_param_the_preflight_never_raises_on_a_malformed_manifest():
    for broken in ({}, {"execution_mode": "versioned_pilot"}, {"execution_unit": "not-a-dict"}):
        status, reasons, _ = preflight_status(broken)
        assert status == "not_reproducible"
        assert reasons


def test_param_a_manifest_whose_c_cannot_be_resolved_is_not_reproducible():
    manifest = _without(_complete_manifest(), "c_requested_token")
    manifest["generation"] = {"train": {"mechanism": "scar"}}

    status, reasons, _ = preflight_status(manifest)

    assert status == "not_reproducible"
    assert reasons == ["missing_c_token"]


def test_param_rejects_an_unknown_check_result():
    with pytest.raises(AuditError, match="unknown check result"):
        make_check(check_id="A01", result="ok", severity="info", scope="global", message="x")


def test_param_rejects_a_group_scoped_check_without_members():
    with pytest.raises(AuditError, match="must name its members"):
        make_check(check_id="A08", result="fail", severity="error", scope="group", message="x")


def test_param_rejects_a_manifest_scoped_check_naming_two_manifests():
    with pytest.raises(AuditError, match="exactly one manifest"):
        make_check(
            check_id="A04",
            result="fail",
            severity="error",
            scope="manifest",
            message="x",
            members=["a/manifest.json", "b/manifest.json"],
        )


def test_param_rejects_a_check_carrying_an_unknown_reason():
    with pytest.raises(AuditError, match="unknown reason code"):
        make_check(
            check_id="A08",
            result="fail",
            severity="error",
            scope="global",
            message="x",
            reasons=["because_i_said_so"],
        )


def test_basic_a_gate_refusal_becomes_a_group_check_naming_every_member():
    members = ["b1/manifest.json", "b1/other/manifest.json"]
    check = gate_error_check(
        check_id="A08",
        message="unit spans training paths ['a', 'b']; a comparability group must pin one",
        members=members,
    )

    assert check["result"] == "fail"
    assert check["scope"] == "group"
    assert check["members"] == members
    assert check["reasons"] == ["fairness_gate_blocked", "group_key_mismatch"]


def test_basic_overall_status_fails_on_a_failure_and_is_partial_when_one_did_not_run():
    def check(result):
        return make_check(check_id="A", result=result, severity="info", scope="global", message="x")

    assert overall_status([check("pass")]) == "pass"
    assert overall_status([check("pass"), check("warn")]) == "pass"
    assert overall_status([check("pass"), check("fail")]) == "fail"
    assert overall_status([check("pass"), check("not_run")]) == "partial"
    assert overall_status([]) == "pass"


def test_determ_overall_status_does_not_depend_on_check_order():
    def check(result):
        return make_check(check_id="A", result=result, severity="info", scope="global", message="x")

    forward = [check("pass"), check("not_run"), check("fail")]
    assert overall_status(forward) == overall_status(list(reversed(forward)))


def test_determ_the_preflight_is_a_pure_function_of_its_input():
    manifest = _complete_manifest()
    first = preflight_status(manifest)

    assert preflight_status(manifest) == first
    # It reads its argument and returns findings; it neither mutates the payload
    # nor remembers anything between calls.
    assert preflight_status(_complete_manifest()) == first
