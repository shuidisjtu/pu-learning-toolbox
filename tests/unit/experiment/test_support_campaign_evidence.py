"""Per-fit correction QA reuses the official warmup/teacher audit predicate."""

import copy
from pathlib import Path

import pytest
from test_survey_audit_checks import _admission_entries, checks

from pu_toolbox.experiment import batch_evidence
from pu_toolbox.experiment import support_campaign as campaign

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    protocol, entries, _ = _admission_entries(tmp_path)
    payload = copy.deepcopy(entries[0]["payload"])
    unit = dict(dataset="spambase", method="lagam", training_path="native_2d", mechanism="scar")
    environment = dict(
        hardware=dict(processor="test CPU"),
        gpu_driver_version="test",
        thread_environment={
            k: "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    )
    packet = batch_evidence.build_evidence(
        unit=unit,
        seeds=[0],
        c_tokens=["0.1"],
        protocol_version=protocol["protocol_version"],
        protocol_sha256=checks.digest(protocol),
        code=dict(
            code_commit="f" * 40,
            code_commit_dirty=False,
            code_commit_state=batch_evidence.CODE_COMMIT_RECORDED,
        ),
        lock=dict(path="uv.lock", sha256="e" * 64, state="recorded"),
        environment=environment,
    )
    job = dict(
        source=str(ROOT),
        expected_keys=[["spambase", "lagam", "native_2d", "scar", "0.1", 0]],
        expected_recipe={},
        campaign_root=str(tmp_path),
        source_code_commit="f" * 40,
        dependency_lock_sha256="e" * 64,
        **{k: unit[k] for k in ("dataset", "method")},
        path="native_2d",
        seeds=[0],
        c_tokens=["0.1"],
        protocol_sha256=checks.digest(protocol),
    )
    payload.update(
        selection={"OA": {}, "PA+clean-support": {}},
        test_results={"OA": {}, "PA+clean-support": {}},
        extra_supervision=dict(all_role_isolation_verified=True, support_removed_from_train=True),
    )
    monkeypatch.setattr(campaign, "matches_recipe_binding", lambda *args: True)
    return job, payload, packet


def findings(evidence):
    job, payload, packet = evidence
    path = batch_evidence.write(Path(job["campaign_root"]) / "batch_evidence", packet)
    payload["batch"] = batch_evidence.manifest_block(packet, path)
    return campaign.repair_manifest_findings(
        job, payload, checkpoint_check=checks._checkpoint_reassessment_problems
    )


def test_basic_determ_complete_trajectory_and_actual_packet_pass(evidence):
    assert findings(evidence) == []
    assert findings(evidence) == []


def test_edge_missing_warmup_is_rejected_by_same_official_predicate(evidence):
    evidence[1]["candidate_runs"][0]["epoch_checkpoints"].pop(0)
    assert findings(evidence)


@pytest.mark.parametrize(
    "problem", ["seed", "lock", "code", "threads", "processor", "driver", "isolation", "track"]
)
def test_param_packet_and_supervision_gaps_fail_closed(evidence, problem):
    job, payload, packet = evidence
    if problem == "seed":
        packet["seeds"] = [1]
    elif problem == "lock":
        packet["dependency_lock"]["sha256"] = "a" * 64
    elif problem == "code":
        packet["code_commit_dirty"] = True
    elif problem == "threads":
        packet["environment"]["thread_environment"]["OMP_NUM_THREADS"] = "1"
    elif problem == "processor":
        packet["environment"]["hardware"]["processor"] = ""
    elif problem == "driver":
        packet["environment"].pop("gpu_driver_version")
    elif problem == "isolation":
        payload["extra_supervision"]["support_removed_from_train"] = False
    else:
        payload["selection"] = {"OA": {}, "PA": {}}
    packet["batch_id"] = batch_evidence.batch_identity(packet)
    assert findings((job, payload, packet))
