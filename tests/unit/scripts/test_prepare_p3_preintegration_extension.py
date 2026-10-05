"""Handoff snapshots preserve facts without authorizing candidate experiments."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from pu_toolbox.experiment.survey_protocol import digest, load_protocol

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "extension_draft", ROOT / "scripts/prepare_p3_preintegration_extension.py"
)
draft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draft)
pytestmark = pytest.mark.unit


def ledger():
    return json.loads((ROOT / "pu_toolbox/experiment/method_ledger.json").read_text())


def test_basic_determ_generated_extension_matches_recorded_draft_and_sources():
    report = draft.build_extension(ledger(), load_protocol())
    recorded = json.loads(
        (
            ROOT / "docs/research/pu_survey/data/p3_preintegration_extension_v1_draft.json"
        ).read_text()
    )
    assert report == recorded
    assert draft.build_extension(ledger(), load_protocol()) == report
    assert report["frozen_protocol_sha256"] == digest(load_protocol())
    assert len(report["candidates"]) == 5
    for entry in report["candidates"]:
        for field in ("storage_evidence_ref", "gpu_evidence_ref"):
            if entry[field] is not None:
                assert (ROOT / entry[field]).is_file()
        assert entry["ledger_snapshot"] == {
            key: ledger()["methods"][entry["method"]][key] for key in entry["ledger_snapshot"]
        }


def test_edge_null_candidates_empty_signatures_no_p3mix_registration_or_source_mutation():
    facts, protocol = ledger(), load_protocol()
    before = copy.deepcopy((facts, protocol))
    report = draft.build_extension(facts, protocol)
    for entry in report["candidates"]:
        assert entry["admitted"] is False
        assert entry["candidate_pool"] is None and entry["budget_ref"] is None
        assert entry["candidate_protocol_ref"] is None and entry["owner_decisions"] == {}
    assert report["unregistered_components"][0]["registered"] is False
    pulns = next(row for row in report["candidates"] if row["method"] == "pulns")
    assert pulns["ledger_snapshot"]["requires_clean_support"] is True
    assert pulns["ledger_snapshot"]["pa_eligible"] is False
    report["candidates"][0]["ledger_snapshot"]["code_version"]["upstream_url"] = "changed"
    assert (facts, protocol) == before


@pytest.mark.parametrize("fault", ["missing_method", "missing_fact", "bad_flag", "old_admission"])
def test_param_inconsistent_source_or_old_admission_fails_closed(fault):
    facts, protocol = ledger(), load_protocol()
    if fault == "missing_method":
        del facts["methods"]["pan"]
    elif fault == "missing_fact":
        del facts["methods"]["pan"]["prior_semantics"]
    elif fault == "bad_flag":
        facts["methods"]["pan"]["calibration_applied"] = "false"
    else:
        protocol["method_profiles"]["pan"] = {}
    with pytest.raises(ValueError):
        draft.build_extension(facts, protocol)
