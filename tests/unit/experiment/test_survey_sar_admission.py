"""Completed neural SAR batches need evidence-based admission outside support profiles."""

import copy
import json

import pytest
from test_survey_audit_checks import _admission_entries, _entry, checks

pytestmark = pytest.mark.unit


def _sar_entries(tmp_path, method):
    protocol, _, passed = _admission_entries(tmp_path)
    unit = next(
        row
        for row in protocol["execution_units"]
        if row["method"] == method
        and row["dataset"] == "spambase"
        and row["training_path"] == "native_2d"
    )
    from _survey_summary_helpers import manifest

    entries = []
    for token in protocol["c_tokens"]["sar_lbe_a"]:
        for seed in protocol["seeds"]:
            payload = manifest(method=method, c=float(token), seed=seed, mechanism="sar_lbe_a")
            payload.update(
                protocol_version=protocol["protocol_version"],
                protocol_sha256=checks.digest(protocol),
                execution_unit=unit,
                profile_identity_schema_version="survey-profile-identity-1",
                profile_id=method,
                algorithm_id=method,
                estimator_parameters=copy.deepcopy(protocol["method_profiles"][method]["params"]),
                budget=copy.deepcopy(protocol["budgets"][unit["budget"]]),
                formal_eligible=False,
                formal_blockers=[
                    *protocol["formal_blockers"],
                    "profile_identity_extension_engineering_only",
                ],
                protocol_deviation=[],
                selection_checkpoint_scope="independent_per_epoch",
                candidate_runs=[
                    {
                        "status": "succeeded",
                        "params": {},
                        "epoch_checkpoints": [
                            {
                                "epoch_position": epoch + 1,
                                "epoch_label": epoch,
                                "component": "model",
                                "validation_metrics": {
                                    "OA": {"val_accuracy": 0.7, "threshold": 0.5}
                                },
                            }
                            for epoch in range(200)
                        ],
                    }
                ],
            )
            if method == "dist_pu":
                payload["candidate_runs"][0]["training_provenance"] = {
                    "implementation_path": "clean_room_single_stage_full_batch",
                    "epochs_completed": 200,
                    "optimizer_steps": 200,
                }
            payload.update(checks.profile_registration_status(method, protocol=protocol))
            path = tmp_path / f"{method}_{token}_{seed}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            entries.append(_entry(payload, name=str(path)))
    return protocol, entries, passed


@pytest.mark.parametrize("method", ["nnpu", "dist_pu"])
def test_basic_determ_complete_sar_grid_discharges_without_mutating_evidence(tmp_path, method):
    protocol, entries, passed = _sar_entries(tmp_path, method)
    before = {str(e["path"]): e["path"].read_bytes() for e in entries}
    first = checks.reassess_admission(entries, protocol=protocol, checks=passed)
    assert all(item["eligible"] for item in first["runs"].values())
    assert checks.reassess_admission(entries, protocol=protocol, checks=passed) == first
    assert before == {str(e["path"]): e["path"].read_bytes() for e in entries}


@pytest.mark.parametrize("method", ["nnpu", "dist_pu"])
@pytest.mark.parametrize(
    "problem",
    ["epoch", "position", "validation", "params", "budget", "candidate", "grid", "plan", "unknown"],
)
def test_param_edge_sar_reassessment_preserves_real_evidence_gaps(tmp_path, method, problem):
    protocol, entries, passed = _sar_entries(tmp_path, method)
    payload = entries[0]["payload"]
    candidate = payload["candidate_runs"][0]
    if problem == "epoch":
        candidate["epoch_checkpoints"].pop()
    elif problem == "position":
        candidate["epoch_checkpoints"][0]["epoch_position"] = 2
    elif problem == "validation":
        candidate["epoch_checkpoints"][0]["validation_metrics"] = {}
    elif problem == "params":
        payload["estimator_parameters"]["batch_size"] = 1
    elif problem == "budget":
        payload["budget"]["epochs"] = 1
    elif problem == "candidate":
        candidate["params"] = {"learning_rate": 9}
    elif problem == "grid":
        entries.pop()
    elif problem == "plan":
        next(c for c in passed if c["check_id"] == "A04")["result"] = "not_run"
    else:
        payload["formal_blockers"].append("unknown_gate")
    report = checks.reassess_admission(entries, protocol=protocol, checks=passed)
    assert not report["runs"][str(entries[0]["path"])]["eligible"]


def test_edge_dist_pu_cannot_claim_frozen_budget_with_missing_optimizer_steps(tmp_path):
    protocol, entries, passed = _sar_entries(tmp_path, "dist_pu")
    entries[0]["payload"]["candidate_runs"][0]["training_provenance"]["optimizer_steps"] = 1
    report = checks.reassess_admission(entries, protocol=protocol, checks=passed)
    assert not report["runs"][str(entries[0]["path"])]["eligible"]


@pytest.mark.parametrize(
    "marker,valid",
    [
        ("TRAINING_COMPLETE 1/1; formal admission requires audit", True),
        ("TRAINING_COMPLETE 1/2; formal admission requires audit", False),
        ("TRAINING_COMPLETE 2/2; formal admission requires audit", False),
        ("MECHANISM_COMPLETE 10/10; resource review required", False),
    ],
)
def test_param_assigned_runner_completion_is_checked_against_logged_runs(tmp_path, marker, valid):
    log = tmp_path / "run.log"
    path = tmp_path / "seed_0/manifest.json"
    log.write_text(f"[nnpu] seed=0 -> {path.as_posix()}\n{marker}\n", encoding="utf-8")
    result = checks._compare_log(str(log), [_entry({}, name=str(path))], str(tmp_path))
    assert (not result) == valid
