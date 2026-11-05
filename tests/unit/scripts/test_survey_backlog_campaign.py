"""Full backlog denominator, family isolation, immutable inputs and result gates."""

import copy
import json
from pathlib import Path

import pytest

from pu_toolbox.experiment.survey_protocol import load_protocol, resolve_protocol_path
from scripts import run_survey_backlog_campaign as script

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("status_exists", [False, True])
def test_edge_monitor_waits_for_first_status_and_uses_campaign_version(
    tmp_path, monkeypatch, capsys, status_exists
):
    (tmp_path / "campaign.json").write_text(json.dumps({"protocol_version": "survey-v3.2"}))
    if status_exists:
        (tmp_path / "status.json").write_text(
            json.dumps(dict(jobs=[], updated_at="test timestamp", gpu_allowlist=[]))
        )

    def stop_after_first_frame(seconds):
        assert seconds == 5
        raise InterruptedError("test frame complete")

    monkeypatch.setattr(script.time, "sleep", stop_after_first_frame)
    with pytest.raises(InterruptedError, match="test frame"):
        script.monitor(tmp_path)
    output = capsys.readouterr().out
    assert "survey-v3.2" in output
    assert ("waiting for first controller status" in output) is not status_exists
    if status_exists:
        assert "GPU pool=[]" in output


def jobs():
    old = load_protocol(resolve_protocol_path("survey-v3.0"))
    new = load_protocol(resolve_protocol_path("survey-v3.1"))
    locations = {row["dataset"]: f"/input/{row['dataset']}" for row in new["execution_units"]}
    return script.expand_jobs(new, old, locations, Path("/support"))


def test_basic_determ_full_grid_is_1845_fits_without_old_completed_units():
    first = jobs()
    assert first == jobs()
    assert len(first) == 1845
    assert len({j["label"] for j in first}) == 1845
    assert len({(j["dataset"], j["method"], j["path"]) for j in first}) == 59
    assert not any(j["method"] in {"nnpu", "upu", "pn_oracle", "genpu"} for j in first)
    assert len([j for j in first if j["method"] == "kldce"]) == 120
    assert len([j for j in first if j["method"] == "self_pu_full_meta"]) == 15
    assert {j["dataset"] for j in first if j["support"]} == {"cifar10"}
    assert len(script.ALLOWED_GPUS) == 4
    assert all(j["checkpoint_epochs"] == 21 for j in first if j["method"] == "cvir")


def test_basic_family_releases_only_after_own_canary_and_holds_after_any_failure():
    planned = jobs()
    canary = next(j for j in planned if j["canary"])
    family = [j for j in planned if j["family"] == canary["family"]]
    other = next(j for j in planned if not j["canary"] and j["family"] != canary["family"])
    script.release(planned)
    assert other["status"] == "waiting_canary"
    canary["status"] = "completed"
    script.release(planned)
    assert all(j["status"] == "pending" for j in family if not j["canary"])
    assert other["status"] == "waiting_canary"
    family[1]["status"] = "failed"
    script.release(planned)
    assert all(j["status"] == "held_family_failure" for j in family[2:])
    assert canary["status"] == "completed"


def accepted_fixture(tmp_path):
    job = next(j for j in jobs() if j["method"] == "cvir" and j["canary"])
    path = tmp_path / "runs" / job["label"] / "manifest.json"
    path.parent.mkdir(parents=True)
    weight = tmp_path / "selected.pt"
    weight.write_bytes(b"fixed test artifact")
    campaign = dict(protocol_sha256="protocol", registry_sha256="registry")
    manifest = dict(
        execution_unit=dict(
            dataset=job["dataset"], method=job["method"], training_path=job["path"]
        ),
        protocol_sha256="protocol",
        recipe_registry_sha256="registry",
        seed=job["seed"],
        c_requested_token=job["c"],
        failures=[],
        generation={role: {"mechanism": job["mechanism"]} for role in ("train", "pu_val")},
        test_results=dict(PA={}, OA={}),
        candidate_runs=[
            dict(
                status="succeeded",
                attempts=1,
                epoch_checkpoints=[dict(epoch_position=i) for i in range(1, 22)],
            )
        ],
        selection=dict(
            PA=dict(checkpoint=dict(path=str(weight), sha256=script.service.sha(weight)))
        ),
    )
    path.write_text(json.dumps(manifest))
    return job, campaign, manifest, path, weight


def test_basic_accept_requires_real_selected_weights_and_full_budget(tmp_path):
    job, campaign, _, path, weight = accepted_fixture(tmp_path)
    assert script.accept(tmp_path, job, campaign)["manifest_sha256"] == script.service.sha(path)
    weight.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        script.accept(tmp_path, job, campaign)


@pytest.mark.parametrize(
    "field,value", [("seed", 8), ("protocol_sha256", "changed"), ("failures", ["failed"])]
)
def test_param_wrong_result_identity_rejected(tmp_path, field, value):
    job, campaign, manifest, path, _ = accepted_fixture(tmp_path)
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity/failure"):
        script.accept(tmp_path, job, campaign)


def test_edge_incomplete_trajectory_wrong_mechanism_or_retry_is_not_success(tmp_path):
    job, campaign, manifest, path, _ = accepted_fixture(tmp_path)
    for kind in ("epochs", "mechanism", "retry"):
        changed = copy.deepcopy(manifest)
        if kind == "epochs":
            changed["candidate_runs"][0]["epoch_checkpoints"].pop()
        elif kind == "retry":
            changed["candidate_runs"][0]["attempts"] = 2
        else:
            changed["generation"]["train"]["mechanism"] = "sar_lbe_a"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            script.accept(tmp_path, job, campaign)


def test_edge_changed_input_stat_or_digest_is_rejected(tmp_path):
    path = tmp_path / "train.npz"
    path.write_bytes(b"input")
    record = dict(
        files={
            str(path): dict(
                size=5, mtime_ns=path.stat().st_mtime_ns, sha256=script.service.sha(path)
            )
        }
    )
    script.verify_inputs(record, full_hash=True)
    record["files"][str(path)]["sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        script.verify_inputs(record, full_hash=True)
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="changed"):
        script.verify_inputs(record)
