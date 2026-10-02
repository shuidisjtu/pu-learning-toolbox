# tests/unit/experiment/test_survey_recipe_registry_manifest.py
"""Manifest-binding rules for the P4.1 recipe registry (D-P4.1-2, D-P4.1-9).

The binding is checked fail-closed.  B1-B4 predate the registry and are not
rewritten, so a manifest with no binding field at all is a legacy artifact and
passes untouched.  But a run that *claimed* the binding -- by declaring
``registry_bound`` or by carrying any one of the fields -- is held to the whole
set: a partial one is a defect, never a quiet downgrade to legacy.
"""

from __future__ import annotations

import pytest
from _survey_recipe_registry_helpers import (
    base_registry,
    bound_manifest,
    codes,
    inner_search,
    profile,
)

from pu_toolbox.experiment.survey_protocol import load_protocol
from pu_toolbox.experiment.survey_recipe_registry import validate_manifest_binding

pytestmark = pytest.mark.unit


def test_basic_registry_bound_manifest_validates_clean():
    value = base_registry()
    manifest = bound_manifest(value)
    assert validate_manifest_binding(manifest, value) == []


def test_basic_legacy_manifest_without_binding_fields_is_accepted():
    legacy = {"protocol_version": "survey-v1.2", "candidate_runs": []}
    assert validate_manifest_binding(legacy, base_registry()) == []


def test_basic_oracle_profile_binding_resolves():
    value = base_registry(methods=("pn_oracle",))
    manifest = bound_manifest(value, "pn_oracle")
    assert manifest["recipe_profile_id"] == "pn_oracle/survey_shared_pn_oracle"
    assert validate_manifest_binding(manifest, value) == []


def test_param_resolved_params_digest_must_match_the_candidate_expansion():
    protocol = load_protocol()
    value = base_registry()
    manifest = bound_manifest(value, protocol=protocol)
    assert validate_manifest_binding(manifest, value, protocol=protocol) == []

    tampered = {**manifest, "resolved_params_sha256": "0" * 64}
    findings = validate_manifest_binding(tampered, value, protocol=protocol)
    assert codes(findings) == ["resolved_params_digest_mismatch"]


def test_param_inner_search_id_null_is_legal_without_an_inner_search():
    value = base_registry()
    manifest = bound_manifest(value)
    assert manifest["inner_search_id"] is None
    assert validate_manifest_binding(manifest, value) == []


def test_param_pusb_kernel_manifest_records_its_inner_search_id():
    value = base_registry(
        profiles={"pusb_kernel": profile("pusb_kernel", inner_search=inner_search())}
    )
    manifest = bound_manifest(value, "pusb_kernel")
    assert manifest["inner_search_id"] == "pusb_kernel_sigma_reg_cv"
    assert validate_manifest_binding(manifest, value) == []


def test_edge_partial_binding_fields_are_rejected():
    value = base_registry()
    manifest = bound_manifest(value)
    del manifest["recipe_candidate_id"]
    assert codes(validate_manifest_binding(manifest, value)) == ["manifest_registry_field_missing"]


def test_edge_registry_bound_manifest_with_every_field_removed_is_rejected():
    declared = {"recipe_binding_status": "registry_bound"}
    assert codes(validate_manifest_binding(declared, base_registry())) == [
        "manifest_registry_field_missing"
    ]


def test_edge_legacy_unbound_carrying_registry_fields_is_rejected():
    value = base_registry()
    manifest = bound_manifest(value, recipe_binding_status="legacy_unbound")
    assert codes(validate_manifest_binding(manifest, value)) == ["partial_recipe_binding"]


def test_edge_unknown_binding_status_is_rejected():
    value = base_registry()
    manifest = bound_manifest(value, recipe_binding_status="unbound")
    assert codes(validate_manifest_binding(manifest, value)) == ["unknown_recipe_binding_status"]


def test_edge_registry_digest_mismatch_is_reported():
    value = base_registry()
    manifest = bound_manifest(value, recipe_registry_sha256="0" * 64)
    assert codes(validate_manifest_binding(manifest, value)) == [
        "manifest_registry_digest_mismatch"
    ]


def test_edge_unknown_candidate_id_is_reported():
    value = base_registry()
    manifest = bound_manifest(value, recipe_candidate_id="gone_v1")
    assert codes(validate_manifest_binding(manifest, value)) == ["unknown_recipe_candidate"]


def test_edge_inner_search_id_mismatch_is_reported():
    value = base_registry()
    manifest = bound_manifest(value, inner_search_id="unexpected_search")
    assert codes(validate_manifest_binding(manifest, value)) == ["inner_search_id_mismatch"]


def test_determ_same_registry_yields_the_same_resolved_params_digest():
    protocol = load_protocol()
    value = base_registry()
    first = bound_manifest(value, protocol=protocol)
    second = bound_manifest(value, protocol=protocol)
    assert first["resolved_params_sha256"] == second["resolved_params_sha256"]
    assert validate_manifest_binding(first, value, protocol=protocol) == []
    assert validate_manifest_binding(second, value, protocol=protocol) == []
