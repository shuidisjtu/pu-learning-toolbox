# tests/unit/experiment/test_survey_recipe_registry_sources.py
"""Source-evidence rules for the P4.1 recipe registry.

``source_refs`` is deliberately outside the registry digest (D-P4.1-4 rule 3),
because several citations are ``external_local`` paths that do not exist inside
the repository.  That makes this pass the only thing standing between a profile
and a citation pointing nowhere, so it is tested on its own: a reference must
resolve, must land on a file rather than a directory, and an external one must
say that it is external and cite lines.
"""

from __future__ import annotations

import json

import pytest
from _survey_recipe_registry_helpers import (
    DEFAULT_REF,
    EXTERNAL_REF,
    base_registry,
    candidate,
    codes,
    evidence,
    evidence_index,
    excluded_profile,
    external_evidence,
    profile,
)

from pu_toolbox.experiment.survey_protocol import load_protocol
from pu_toolbox.experiment.survey_recipe_registry import (
    canonical_digest,
    resolve_candidate,
    validate_registry,
)

pytestmark = pytest.mark.unit


def test_basic_source_refs_resolve_into_the_evidence_index():
    value = base_registry()
    assert value["profiles"]["nnpu"]["source_refs"] == [DEFAULT_REF]
    assert validate_registry(value, load_protocol()) == []


def test_basic_adopted_profile_without_source_refs_is_reported():
    value = base_registry(profiles={"nnpu": profile("nnpu", source_refs=[])})
    assert codes(validate_registry(value, load_protocol())) == ["source_refs_missing"]


def test_basic_in_repo_evidence_must_cite_a_file_not_a_directory():
    index = evidence_index(**{DEFAULT_REF: evidence(file="")})
    value = base_registry(source_evidence=index)
    findings = validate_registry(value, load_protocol())
    assert codes(findings) == ["source_evidence_shape_invalid"]
    assert findings[0].path == f"source_evidence.{DEFAULT_REF}.file"


def test_param_external_local_evidence_must_declare_itself_and_cite_lines():
    undeclared = base_registry(
        source_evidence=evidence_index(**{EXTERNAL_REF: external_evidence(reproducibility="")})
    )
    assert codes(validate_registry(undeclared, load_protocol())) == [
        "source_evidence_shape_invalid"
    ]
    unlined = base_registry(
        source_evidence=evidence_index(**{EXTERNAL_REF: external_evidence(lines="")})
    )
    findings = validate_registry(unlined, load_protocol())
    assert codes(findings) == ["source_evidence_shape_invalid"]
    assert findings[0].path == f"source_evidence.{EXTERNAL_REF}.lines"


def test_param_excluded_profile_may_omit_source_refs():
    value = base_registry(profiles={"kldce": excluded_profile("kldce", source_refs=[])})
    assert validate_registry(value, load_protocol()) == []


def test_edge_unknown_source_ref_is_reported():
    value = base_registry(profiles={"nnpu": profile("nnpu", source_refs=["not_indexed"])})
    assert codes(validate_registry(value, load_protocol())) == ["unknown_source_ref"]


def test_edge_missing_evidence_index_is_reported():
    value = base_registry(source_evidence=None)
    assert codes(validate_registry(value, load_protocol())) == ["source_evidence_shape_invalid"]


def test_edge_evidence_type_outside_the_closed_set_is_reported():
    index = evidence_index(**{DEFAULT_REF: evidence(type="url")})
    value = base_registry(source_evidence=index)
    findings = validate_registry(value, load_protocol())
    assert codes(findings) == ["source_evidence_shape_invalid"]
    assert findings[0].path == f"source_evidence.{DEFAULT_REF}.type"


def test_edge_source_refs_that_are_not_a_list_are_reported():
    value = base_registry(profiles={"nnpu": profile("nnpu", source_refs=DEFAULT_REF)})
    assert codes(validate_registry(value, load_protocol())) == ["source_evidence_shape_invalid"]


def test_determ_growing_the_evidence_index_never_moves_the_registry_digest():
    value = base_registry()
    baseline = canonical_digest(value)
    grown = json.loads(json.dumps(value))
    grown["source_evidence"]["a_new_external_ref"] = external_evidence()
    assert canonical_digest(grown) == baseline


def test_determ_locked_profile_with_only_external_evidence_warns():
    protocol = load_protocol()
    prof = profile(
        "nnpu",
        recipe_lifecycle="locked",
        source_refs=[EXTERNAL_REF],
        candidate_pool=[candidate("nnpu_default_v1", source_refs=[EXTERNAL_REF])],
    )
    prof["candidate_pool"][0]["resolved_snapshot"] = resolve_candidate(
        prof, prof["candidate_pool"][0], protocol
    )
    value = base_registry(profiles={"nnpu": prof})
    findings = validate_registry(value, protocol)
    assert codes(findings) == ["source_evidence_external_only"]
    assert findings[0].severity == "warning"
