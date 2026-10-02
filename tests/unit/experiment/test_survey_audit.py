"""The auditor must turn a malformed manifest into a finding, not a traceback.

The aggregation entry point raises on a manifest missing a field it reads, which
is correct for an entry point whose whole output is an exit code.  An audit that
stopped at the first bad item would report one problem and hide the rest, and a
traceback is not a finding a reviewer can act on -- so the preflight returns
reason codes and names the fields it could not find, and never raises.

The identity fields asserted here are the ones a real manifest carries.  Code
commit and the dependency lock are deliberately absent from the list: the survey
runner records them per batch rather than per run, so demanding them per manifest
would mark every legitimate artifact defective.

The check schema, its scopes and the gate roll-up have their own file,
``test_survey_audit_scope.py``.
"""

import pytest

from pu_toolbox.experiment.survey_audit import preflight_status
from pu_toolbox.experiment.survey_comparison import load_comparison_protocol

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
        # Paired with the view, as the runner writes it: only a calibrated run
        # reports ts-compatible, and a manifest missing the flag is refused rather
        # than assumed either way.
        "calibration_applied": True,
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


def test_determ_the_preflight_is_a_pure_function_of_its_input():
    manifest = _complete_manifest()
    first = preflight_status(manifest)

    assert preflight_status(manifest) == first
    # It reads its argument and returns findings; it neither mutates the payload
    # nor remembers anything between calls.
    assert preflight_status(_complete_manifest()) == first


def test_basic_a_manifest_the_frozen_matrix_covers_passes_the_preflight():
    status, reasons, _ = preflight_status(
        _complete_manifest(), comparison=load_comparison_protocol()
    )

    assert status == "formal"
    assert reasons == []


def test_param_an_oracle_claim_the_matrix_does_not_corroborate_is_reported():
    # The manifest asserts c-independence on an execution row the matrix holds as a
    # scar row; the two statements disagree, and neither is derived from the other.
    manifest = _complete_manifest()
    manifest["c_independent"] = True

    status, reasons, _ = preflight_status(manifest, comparison=load_comparison_protocol())

    assert status == "not_reproducible"
    assert "oracle_c_independent_conflict" in reasons


def test_determ_the_matrix_check_is_opt_in_and_leaves_the_structure_verdict_alone():
    entry = _without(_complete_manifest(), "split_ref")
    absent_fields = preflight_status(entry)

    # Structure is judged with or without the matrix; the identity the matrix holds
    # is only compared when a caller supplies it, so a caller that has none is not
    # silently credited with the check having run.
    assert preflight_status(entry, comparison=load_comparison_protocol()) == absent_fields
    assert absent_fields[1] == ["missing_manifest_field"]
