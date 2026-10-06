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
    return json.loads(
        (ROOT / "pu_toolbox/experiment/method_ledger.json").read_text(encoding="utf-8")
    )


def test_edge_unicode_ledger_read_is_independent_of_windows_default_encoding(monkeypatch):
    original = Path.read_text

    def windows_default(path, *args, **kwargs):
        kwargs.setdefault("encoding", "cp1252")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", windows_default)
    assert len(draft.build_extension(ledger(), load_protocol())["candidates"]) == 5


def test_basic_determ_generated_extension_matches_recorded_draft_and_sources():
    report = draft.build_extension(ledger(), load_protocol())
    recorded = json.loads(
        (ROOT / "docs/research/pu_survey/data/p3_preintegration_extension_v1_draft.json").read_text(
            encoding="utf-8"
        )
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
    assert pulns["ledger_snapshot"]["modality_backbone"]["code_capability"] == {
        "native_architectures": ["cnn", "mlp"],
        "input_ndims": [2, 4],
        "encoder_parameter": "encoder",
        "trains_encoder": True,
    }
    assert pulns["ledger_snapshot"]["calibration_applied"] is False
    assert "CNN_representation_and_sequential_policy_resource_spec" in pulns["specific_blockers"]
    assert pulns["ledger_snapshot"]["source_behavior_review_ref"].endswith(
        "pulns_source_cnn_review_20261007.md"
    )
    holistic = next(row for row in report["candidates"] if row["method"] == "holistic_pu")
    assert holistic["ledger_snapshot"]["warmup_selection_review_ref"].endswith(
        "holistic_lzo_selection_20261007.md"
    )
    assert (
        "lzo_positive_loss_recipe_and_external_selection_isolation" in holistic["specific_blockers"]
    )
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
