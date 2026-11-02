"""Per-fit applicability, immutable replacement identities and per-c canaries."""

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from pu_toolbox.experiment.survey_protocol import digest, load_protocol, resolve_protocol_path
from pu_toolbox.experiment.survey_recipe_registry import canonical_digest, load_registry
from scripts import prepare_kldce_recovery as script
from scripts import run_survey_backlog_campaign as backlog

pytestmark = pytest.mark.unit


def parent(tmp_path, monkeypatch):
    root = tmp_path / "parent"
    root.mkdir()
    protocol = load_protocol(resolve_protocol_path("survey-v3.1"))
    registry = load_registry(
        backlog.ROOT / "pu_toolbox/experiment/survey_recipe_registry_v1_1.json"
    )
    jobs, inputs = [], {}
    for seed in range(2):
        directory = tmp_path / f"split_{seed}"
        directory.mkdir()
        for role in ("train", "pu_val", "clean_val", "test"):
            np.savez(
                directory / f"{role}.npz",
                X=np.zeros((100, 3), dtype="float32"),
                y=np.arange(100) % 2,
            )
        (directory / "split_manifest.json").write_text(
            json.dumps(dict(dataset="spambase", seed=seed, class_prior=dict(population=0.5)))
        )
        inputs[str(directory)] = backlog.input_evidence(
            directory, dataset="spambase", seed=seed, support=False
        )
        for token in ("0.1", "0.3", "0.5"):
            jobs.append(
                dict(
                    label=f"kldce-spambase-s{seed}-c{token}",
                    family="kldce-spambase-scar",
                    dataset="spambase",
                    method="kldce",
                    path="native_2d",
                    mechanism="scar",
                    seed=seed,
                    c=token,
                    canary=seed == 0 and token == "0.1",
                    status="pending",
                    data_dir=str(directory),
                    support=False,
                    lane="cpu",
                    checkpoint_epochs=None,
                )
            )
    campaign = dict(
        protocol_version="survey-v3.1",
        protocol_sha256=digest(protocol),
        registry_sha256=canonical_digest(registry),
        jobs=jobs,
        inputs=inputs,
    )
    (root / "campaign.json").write_text(json.dumps(campaign))
    (root / "status.json").write_text(
        json.dumps(
            dict(
                jobs=[
                    dict(j, status="failed" if j["canary"] else "held_family_failure") for j in jobs
                ]
            )
        )
    )
    monkeypatch.setattr(backlog, "clean_commit", lambda: "fixed")
    return root


def test_basic_determ_preflight_preserves_denominator_and_recovers_only_feasible_c(
    tmp_path, monkeypatch
):
    root = parent(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    campaign = script.prepare(tmp_path / "recovery", root)
    assert dict(Counter(j["status"] for j in campaign["jobs"])) == dict(
        blocked_preflight=4, pending=1, waiting_canary=1
    )
    assert (
        campaign["total_fits"] == 6
        and campaign["reconciliation"]["additional_scientific_fits"] == 0
    )
    assert campaign["gpu_allowlist"] == [] and campaign["max_cpu_jobs"] == 1
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    assert script.prepare(tmp_path / "other-plan", root) == campaign
    backlog.verify_reconciliation(campaign)
    canary = next(j for j in campaign["jobs"] if j["canary"])
    assert canary["c"] == "0.5"
    canary["status"] = "completed"
    backlog.release(campaign["jobs"])
    assert dict(Counter(j["status"] for j in campaign["jobs"])) == dict(
        blocked_preflight=4, completed=1, pending=1
    )
    with pytest.raises(ValueError, match="exists"):
        script.prepare(tmp_path / "recovery", root)


@pytest.mark.parametrize("c,feasible", [("0.1", False), ("0.3", False), ("0.5", True)])
def test_param_screen_matches_actual_trainer_without_model_or_feature_fit(
    tmp_path, monkeypatch, c, feasible
):
    from pu_toolbox.estimators.risk.kldce import KLDCEClassifier
    from pu_toolbox.experiment.kernel_candidate import KLDCECandidateTrainer
    from pu_toolbox.experiment.strategies import SCARGenerator

    root = parent(tmp_path, monkeypatch)
    campaign = script.prepare(tmp_path / "out", root)
    job = next(j for j in campaign["jobs"] if j["c"] == c and j["seed"] == 0)
    labels, shape, _ = script.train_metadata(Path(job["data_dir"]))
    marked, meta = SCARGenerator().generate(None, labels, float(c), 0)
    trainer = KLDCECandidateTrainer(
        prior_policy="population_override", kernel_payload_cap_bytes=256 * 2**30
    )
    trainer.validate_generation(meta, meta)
    model = KLDCEClassifier(flip_probability=1 - meta["c_realized"])
    assert job["preflight"]["feasibility"]["feasible"] is feasible
    if feasible:
        actual = trainer.preflight(
            model, np.zeros(shape, dtype="float32"), marked, class_prior=0.5, os_or_ts="os"
        )
        assert (
            actual["qp_equality_box_feasibility"]["rhs"] == job["preflight"]["feasibility"]["rhs"]
        )
        assert (
            actual["qp_named_array_payload_bytes"] + actual["covariance_payload_lower_bound_bytes"]
            == job["preflight"]["named_payload_bytes"]
        )
    else:
        with pytest.raises(ValueError, match="infeasible"):
            trainer.preflight(
                model, np.zeros(shape, dtype="float32"), marked, class_prior=0.5, os_or_ts="os"
            )


@pytest.mark.parametrize("status", ["running", "completed", "pending"])
def test_edge_recovery_refuses_duplicate_active_or_completed_work(tmp_path, monkeypatch, status):
    root = parent(tmp_path, monkeypatch)
    state = backlog.read(root / "status.json")
    state["jobs"][0]["status"] = status
    (root / "status.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="duplicate"):
        script.prepare(tmp_path / "out", root)
    assert not (tmp_path / "out").exists()


def test_edge_changed_identity_and_blocked_worker_are_refused(tmp_path, monkeypatch):
    root = parent(tmp_path, monkeypatch)
    out = tmp_path / "out"
    campaign = script.prepare(out, root)
    with pytest.raises(ValueError, match="blocked fit"):
        backlog.worker(out, campaign["jobs"][0]["label"])
    changed = copy.deepcopy(campaign)
    changed["jobs"][0]["c"] = "0.5"
    with pytest.raises(ValueError, match="identity changed"):
        backlog.verify_reconciliation(changed)
    assert not (out / "runs").exists()


def test_edge_resource_cap_keeps_mathematically_feasible_fit_blocked(tmp_path, monkeypatch):
    root = parent(tmp_path, monkeypatch)
    job = backlog.read(root / "campaign.json")["jobs"][2]
    budget = dict(max_acs_iter=50, max_inner_iter=2000, kernel_payload_cap_bytes=1)
    result = script.screen(job, script.train_metadata(Path(job["data_dir"])), 0.5, budget)
    assert result["feasibility"]["feasible"]
    assert result["blockers"] == ["named_QP_and_covariance_exceed_frozen_cap"]
    assert not result["model_or_kernel_constructed"]


def test_edge_original_campaign_without_version_still_requires_matching_frozen_hashes(
    tmp_path, monkeypatch
):
    root = parent(tmp_path, monkeypatch)
    original = backlog.read(root / "campaign.json")
    original.pop("protocol_version")
    (root / "campaign.json").write_text(json.dumps(original))
    fixed = script.prepare(tmp_path / "out", root)
    assert fixed["protocol_version"] == "survey-v3.1"
    backlog.verify_reconciliation(fixed)
    original["protocol_sha256"] = "changed"
    (root / "campaign.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="protocol/registry changed"):
        script.prepare(tmp_path / "bad", root)
