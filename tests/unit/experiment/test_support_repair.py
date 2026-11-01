"""D53 repair decisions reuse auditor criteria without changing source results."""

import copy
import hashlib
from pathlib import Path

import pytest
from test_survey_audit_checks import _admission_entries, checks

from pu_toolbox.experiment.support_repair import plan_support_repairs

pytestmark = pytest.mark.unit


def fixture(tmp_path):
    protocol, entries, _ = _admission_entries(tmp_path)
    for entry in entries:
        entry["manifest_sha256"] = hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
    return protocol, entries


def plan(protocol, entries):
    return plan_support_repairs(
        protocol, entries, checkpoint_check=checks._checkpoint_reassessment_problems
    )


def test_basic_determ_complete_trajectory_is_not_admission_and_source_is_immutable(tmp_path):
    protocol, entries = fixture(tmp_path)
    before = copy.deepcopy(entries)
    raw = {str(e["path"]): Path(e["path"]).read_bytes() for e in entries}
    report = plan(protocol, entries)
    assert report == plan(protocol, entries) and entries == before
    assert report["counts"] == dict(trajectory_complete=15, missing=165)
    assert all(not r["formal_admission"] for r in report["fits"])
    assert raw == {str(e["path"]): Path(e["path"]).read_bytes() for e in entries}


def test_edge_warmup_gap_is_counted_per_epoch_not_only_total(tmp_path):
    protocol, entries = fixture(tmp_path)
    for e in entries:
        e["payload"]["candidate_runs"][0]["epoch_checkpoints"] = e["payload"]["candidate_runs"][0][
            "epoch_checkpoints"
        ][2:]
    report = plan(protocol, entries)
    affected = [r for r in report["fits"] if r["status"] == "rerun_required"]
    assert len(affected) == 15
    assert affected[0]["missing_epoch_components"] == [
        dict(epoch=0, component="model"),
        dict(epoch=1, component="model"),
    ]
    assert (
        affected[0]["expected_checkpoint_count"] == 20
        and affected[0]["recorded_checkpoint_count"] == 18
    )


def test_param_duplicate_attempts_require_explicit_reconciliation(tmp_path):
    protocol, entries = fixture(tmp_path)
    report = plan(protocol, [*entries, copy.deepcopy(entries[0])])
    assert report["counts"]["conflict"] == 1
    item = next(r for r in report["fits"] if r["status"] == "conflict")
    assert len(item["sources"]) == 2 and not item["formal_admission"]


def test_edge_incomplete_failed_attempt_does_not_shadow_success(tmp_path):
    protocol, entries = fixture(tmp_path)
    failed = copy.deepcopy(entries[0])
    failed["payload"]["failures"] = ["fit failed"]
    report = plan(protocol, [*entries, failed])
    assert report["counts"]["trajectory_complete"] == 15
    assert len(report["excluded_sources"]) == 1


def test_param_source_digest_is_required_before_a_reuse_decision(tmp_path):
    protocol, entries = fixture(tmp_path)
    del entries[0]["manifest_sha256"]
    with pytest.raises(ValueError, match="digest"):
        plan(protocol, entries)


@pytest.mark.parametrize("method", ["pulns", "self_pu_full_meta"])
def test_basic_pulns_episodes_and_full_meta_teacher_boundaries(tmp_path, method):
    protocol, entries = fixture(tmp_path)
    profile = protocol["method_profiles"][method]
    unit = next(
        r
        for r in protocol["execution_units"]
        if r["method"] == method and r["dataset"] == "spambase"
    )
    count = protocol["budgets"][profile["budget"]]["epochs"]
    labels = list(range(1, count + 1)) if method == "self_pu_full_meta" else list(range(count))
    components = ["teacher_1", "teacher_2"] if method == "self_pu_full_meta" else ["model"]
    for entry in entries:
        m = entry["payload"]
        m.update(
            profile_id=method,
            algorithm_id="self_pu" if method == "self_pu_full_meta" else method,
            execution_unit=unit,
            estimator_parameters=profile["params"],
            budget=protocol["budgets"][profile["budget"]],
        )
        candidate = m["candidate_runs"][0]
        candidate["training_provenance"] = {
            "captured_epochs" if method == "self_pu_full_meta" else "captured_episodes": labels
        }
        candidate["epoch_checkpoints"] = [
            dict(epoch_label=e, component=c, validation_metrics={"OA": {}, "PA+clean-support": {}})
            for e in labels
            for c in components
        ]
    report = plan(protocol, entries)
    assert report["counts"]["trajectory_complete"] == 15
    if method == "self_pu_full_meta":
        entries[0]["payload"]["candidate_runs"][0]["epoch_checkpoints"].pop(0)
        changed = next(
            r for r in plan(protocol, entries)["fits"] if r["status"] == "rerun_required"
        )
        assert changed["missing_epoch_components"] == [dict(epoch=1, component="teacher_1")]
