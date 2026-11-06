# ruff: noqa: N803, F811
"""Runtime eligibility preserves configured blockers and method identities."""

import pytest
from _survey_script_helpers import make_splits, survey_script  # noqa: F401

from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.survey_execution import assemble_model
from pu_toolbox.experiment.survey_protocol import (
    PROTOCOL_PATH,
    load_protocol,
    resolve_unit,
    validate_comparable_manifests,
)

pytestmark = pytest.mark.unit


def _canonical(survey_script, tmp_path, method="upu"):
    """A bound unit carrying the protocol's own seeds and c grid.

    Nothing deviates, so whatever lands in ``formal_blockers`` comes from the
    protocol and the runtime rules rather than from the request.
    """
    make_splits(tmp_path)
    parts = survey_script.load_split_parts(tmp_path)
    protocol = load_protocol()
    row = resolve_unit(protocol, "spambase", method)
    model = assemble_model(protocol, row, 3, seed=0, params={}, class_prior=0.3, device="cpu")
    config = {
        "architecture": "mlp",
        "c": 0.5,
        "c_requested_token": "0.5",
        "split_ref": {"dataset": "spambase", "seed": 0},
        "survey_protocol": {
            "path": str(PROTOCOL_PATH),
            "dataset": "spambase",
            "method": method,
            "mechanism": "scar",
            "seeds": protocol["seeds"],
            "c_tokens": protocol["c_tokens"]["scar"],
        },
    }
    return model, parts, config


def _manifest(survey_script, tmp_path, method="upu", **config_overrides):
    model, parts, config = _canonical(survey_script, tmp_path, method)
    config.update(config_overrides)
    return ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest


def test_basic_canonical_scar_unit_retains_protocol_blockers(survey_script, tmp_path):  # noqa: F811
    """A valid run cannot grant release to a protocol awaiting acceptance."""
    manifest = _manifest(survey_script, tmp_path)
    assert manifest["execution_mode"] == "versioned_pilot"
    assert manifest["protocol_deviation"] == []
    assert manifest["formal_blockers"] == [
        *load_protocol()["formal_blockers"],
        "profile_identity_extension_engineering_only",
        "collaborator_review",
    ]
    assert manifest["formal_eligible"] is False


def test_basic_self_pu_row_keeps_its_ablation_variant_label(survey_script, tmp_path):  # noqa: F811
    """Releasing the Self-PU blocker must not drop the scope record beside it.

    ``method_variant`` is what tells a reader the pilot's ``self_pu`` numbers
    are the no-meta ablation; it is a scope declaration, not a blocker, and it
    has to survive the release.
    """
    manifest = _manifest(survey_script, tmp_path, method="self_pu")
    assert (
        "SelfPU_clean_validation_meta_reweighting_OA_integration" not in manifest["formal_blockers"]
    )
    assert manifest["method_variant"] == "without_clean_validation_meta_reweighting"


@pytest.mark.parametrize(
    "deviation, override",
    [
        ("candidate_pool", {"candidates": [{"reg_lambda": 0.01}]}),
        ("seed_subset", None),
    ],
)
def test_param_every_real_deviation_still_blocks_the_release(
    survey_script, tmp_path, deviation, override
):  # noqa: F811
    """The release removes protocol standing, not the deviation record.

    A request that genuinely departs from the matrix must still land outside
    the formal tier, otherwise the release would have dissolved the gate
    instead of the blockers.
    """
    model, parts, config = _canonical(survey_script, tmp_path)
    if override is not None:
        config.update(override)
    else:
        config["survey_protocol"]["seeds"] = [0]
    manifest = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    assert deviation in manifest["protocol_deviation"]
    assert "protocol_deviation" in manifest["formal_blockers"]
    assert manifest["formal_eligible"] is False


def test_edge_pending_canonical_manifest_requires_diagnostic_mode(survey_script, tmp_path):  # noqa: F811
    manifest = _manifest(survey_script, tmp_path)
    with pytest.raises(ValueError, match="formal aggregation blocked"):
        validate_comparable_manifests([manifest])
    validate_comparable_manifests([manifest], require_formal=False)


def test_determ_pending_eligibility_is_reproducible(survey_script, tmp_path):  # noqa: F811
    """Two identical canonical runs report the same standing and digest."""
    first = _manifest(survey_script, tmp_path)
    second = _manifest(survey_script, tmp_path)
    expected = [
        *load_protocol()["formal_blockers"],
        "profile_identity_extension_engineering_only",
        "collaborator_review",
    ]
    assert first["formal_blockers"] == second["formal_blockers"] == expected
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert first["formal_eligible"] is second["formal_eligible"] is False
