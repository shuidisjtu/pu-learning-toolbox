"""Production CI checks declared locks without manufacturing a freeze."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pu_toolbox.experiment.survey_protocol import load_protocol
from pu_toolbox.experiment.survey_recipe_registry import build_draft_registry, resolve_candidate

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import check_survey_recipe_registry as gate  # noqa: E402

pytestmark = pytest.mark.unit


def _locked():
    protocol = load_protocol()
    registry = build_draft_registry(protocol)
    registry["recipe_lifecycle"] = "locked"
    for profile in registry["profiles"].values():
        if profile["recipe_lifecycle"] != "excluded":
            profile["recipe_lifecycle"] = "locked"
            for candidate in profile["candidate_pool"]:
                candidate["resolved_snapshot"] = resolve_candidate(profile, candidate, protocol)
    return registry


def _run(tmp_path, registry):
    path = tmp_path / "synthetic-registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return gate.main(["--registry", str(path), "--require-locked"])


def test_basic_locked_synthetic_registry_passes_without_approving_project(tmp_path, capsys):
    assert _run(tmp_path, _locked()) == 0
    assert "registry sha256:" in capsys.readouterr().out


@pytest.mark.parametrize("state", ["draft", "under_review", "signed_off", "retired", "excluded"])
def test_param_nonlocked_registry_cannot_pass_production_gate(tmp_path, state, capsys):
    registry = _locked()
    registry["recipe_lifecycle"] = state
    assert _run(tmp_path, registry) == 1
    assert "lock check failed" in capsys.readouterr().out


@pytest.mark.parametrize("state", ["draft", "under_review", "signed_off", "retired"])
def test_param_nonlocked_profile_cannot_hide_under_locked_registry(tmp_path, state, capsys):
    registry = _locked()
    registry["profiles"]["nnpu"]["recipe_lifecycle"] = state
    assert _run(tmp_path, registry) == 1
    assert "profiles.nnpu.recipe_lifecycle" in capsys.readouterr().out


def test_edge_missing_production_registry_is_not_a_validated_lock(tmp_path, capsys):
    assert gate.main(["--registry", str(tmp_path / "missing.json"), "--require-locked"]) == 1
    assert "Nothing was validated" in capsys.readouterr().out


def test_edge_declared_lock_does_not_bypass_snapshot_validation(tmp_path, capsys):
    registry = _locked()
    registry["profiles"]["nnpu"]["candidate_pool"][0]["resolved_snapshot"] = {}
    assert _run(tmp_path, registry) == 1
    assert "resolved_snapshot_mismatch" in capsys.readouterr().out


def test_basic_ci_keeps_production_activation_separate_from_draft():
    workflow = (Path(__file__).resolve().parents[3] / ".github/workflows/tests.yml").read_text()
    assert "hashFiles('pu_toolbox/experiment/survey_recipe_registry_v1.json') != ''" in workflow
    assert "--require-present --require-locked" in workflow
    assert "survey_recipe_registry_v0_1_draft.json --require-present" in workflow


def test_determ_production_check_reports_identical_digest_without_rewriting(tmp_path, capsys):
    registry = _locked()
    assert _run(tmp_path, registry) == 0
    first = capsys.readouterr().out
    path = tmp_path / "synthetic-registry.json"
    before = path.read_bytes()
    assert gate.main(["--registry", str(path), "--require-locked"]) == 0
    assert capsys.readouterr().out == first
    assert path.read_bytes() == before
