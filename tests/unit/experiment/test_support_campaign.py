"""Correction namespaces and full-budget sentinel release, without real GPUs."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from pu_toolbox.experiment import support_campaign as campaign
from pu_toolbox.experiment.queue_scheduling import dispatch_priority
from pu_toolbox.experiment.survey_protocol import digest, load_protocol

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / "source"
    package = source / "pu_toolbox/experiment"
    package.mkdir(parents=True)
    for name in (
        "support_checkpoints.py",
        "survey_protocol_v3.json",
        "survey_recipe_registry_v1.json",
    ):
        shutil.copyfile(ROOT / "pu_toolbox/experiment" / name, package / name)
    (source / "uv.lock").write_text("test lock")
    protocol = load_protocol(package / "survey_protocol_v3.json")
    support = tmp_path / "support/spambase"
    for seed in protocol["seeds"]:
        split = support / f"split_{seed}"
        split.mkdir(parents=True)
        (split / "support_manifest.json").write_text(
            json.dumps(dict(dataset="spambase", seed=seed, class_prior=dict(population=0.4)))
        )
    unit = dict(dataset="spambase", profile="lagam", training_path="native_2d")
    runs = [
        dict(
            dataset="spambase",
            method="lagam",
            training_path="native_2d",
            mechanism="scar",
            seed=s,
            c_token=c,
            view="os-compatible",
        )
        for s in protocol["seeds"]
        for c in protocol["c_tokens"]["scar"]
    ]
    plan = dict(
        schema_version="survey-support-repair-plan-1",
        launched=False,
        source=dict(
            path=str(source),
            code_commit="f" * 40,
            support_recorder_sha256=campaign.file_hash(package / "support_checkpoints.py"),
        ),
        protocol_sha256=digest(protocol),
        result_roots=[str(tmp_path / "old")],
        commands=[
            dict(
                unit=unit,
                results_root=str(tmp_path / "new/spambase__lagam__native_2d"),
                pilot_argv=["--support-root", str(support.parent)],
            )
        ],
        fits=[dict(**unit, status="rerun_required", sources=[]) for _ in runs],
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    (tmp_path / "spambase__lagam__native_2d.plan.json").write_text(
        json.dumps(dict(source_protocol_sha256=digest(protocol), runs=runs))
    )
    monkeypatch.setattr(
        campaign.subprocess,
        "check_output",
        lambda argv, **kw: "f" * 40 if "rev-parse" in argv else b"",
    )
    return path, plan


def test_basic_determ_verified_mount_is_read_only_and_sentinel_first(prepared):
    path, _ = prepared
    meta, jobs = campaign.load_support_campaign(path)
    assert (meta, jobs) == campaign.load_support_campaign(path)
    assert len(jobs) == 15 and sum(j["status"] == "held" for j in jobs) == 14
    assert sum(j["campaign_canary"] for j in jobs) == 1
    assert sorted(jobs, key=dispatch_priority)[0]["campaign_canary"]
    assert not Path(meta["root"]).exists()


@pytest.mark.parametrize("problem", ["root", "protocol", "commit", "recorder", "grid"])
def test_param_preflight_failure_never_creates_output(prepared, problem):
    path, plan = prepared
    if problem == "root":
        Path(plan["commands"][0]["results_root"]).parent.mkdir()
    elif problem == "grid":
        preflight = path.parent / "spambase__lagam__native_2d.plan.json"
        content = json.loads(preflight.read_text())
        content["runs"].pop()
        preflight.write_text(json.dumps(content))
    else:
        field = {
            "protocol": "protocol_sha256",
            "commit": "code_commit",
            "recorder": "support_recorder_sha256",
        }[problem]
        (plan if problem == "protocol" else plan["source"])[field] = "changed"
        path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        campaign.load_support_campaign(path)


def test_basic_explicit_namespace_preserves_baseline_and_mount_is_idempotent(prepared):
    meta, jobs = campaign.load_support_campaign(prepared[0])
    old = copy.deepcopy(jobs[0])
    old.pop("campaign_id")
    old.update(label="baseline", root=old["root"] + "_old")
    merged = campaign.mount_campaign([old], meta, jobs)
    assert len(merged) == 16 and merged[0] == old
    assert campaign.mount_campaign(merged, meta, jobs) is merged
    duplicate = copy.deepcopy(jobs[0])
    duplicate.update(root=duplicate["root"] + "_dup", label="duplicate")
    with pytest.raises(ValueError, match="fit identity"):
        campaign.mount_campaign([duplicate], dict(campaign_id="another"), jobs)


@pytest.mark.parametrize(
    "status,verified,released,blocked",
    [
        ("running", False, 0, 0),
        ("completed", False, 0, 0),
        ("completed", True, 14, 0),
        ("failed", False, 0, 14),
        ("blocked", False, 0, 14),
    ],
)
def test_edge_canary_fail_closed_release(prepared, status, verified, released, blocked):
    _, jobs = campaign.load_support_campaign(prepared[0])
    sentinel = next(j for j in jobs if j["campaign_canary"])
    sentinel.update(status=status, gate_passed=verified)
    campaign.release_canary_followups(jobs)
    followups = [j for j in jobs if not j["campaign_canary"]]
    assert sum(j["status"] == "pending" for j in followups) == released
    assert sum(j["status"] == "blocked" for j in followups) == blocked


def test_edge_later_failure_quarantines_unstarted_siblings_but_not_running_workers(prepared):
    _, jobs = campaign.load_support_campaign(prepared[0])
    jobs[0].update(status="completed", gate_passed=True)
    campaign.release_canary_followups(jobs)
    jobs[1]["status"] = "failed"
    jobs[2]["status"] = "running"
    campaign.release_canary_followups(jobs)
    assert jobs[2]["status"] == "running"
    assert sum(j["status"] == "blocked" for j in jobs) == 12


def test_basic_empty_campaign_tables_are_separate_and_do_not_touch_evidence(prepared):
    metadata, jobs = campaign.load_support_campaign(prepared[0])
    root = Path(metadata["root"])
    root.mkdir()
    campaign.write_campaign_tables(root, jobs)
    assert (root / "per_run.csv").read_text(encoding="utf-8-sig").startswith("method,dataset,")
    assert (root / "survey_results.xlsx").is_file()
    assert not list(root.rglob("batch_ref.json"))
