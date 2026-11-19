"""Check campaign safety gates without training or claiming a device."""

from __future__ import annotations

import csv
import json
import sys
from argparse import Namespace

import pytest

from pu_toolbox.utils.serialization import file_hash, strict_canonical_hash
from scripts import run_paired_resnet_campaign as campaign

pytestmark = pytest.mark.unit

COMMIT = "c" * 40
GPU = "GPU-f4af1aef-0099-034d-81bc-7c4fa44b2a95"
MODEL = "NVIDIA RTX A6000"


def _state():
    return {job.key: {"status": "pending", "attempts": []} for job in campaign.jobs()}


def _metrics(value):
    return {metric: value for metric in campaign.METRICS}


def _set_pair(state, dataset, seed, method="robust_pu"):
    for arm, value in zip(campaign.ARMS, (0.7, 0.8), strict=True):
        job = campaign.Job(dataset, method, seed, arm)
        state[job.key].update(
            status="completed",
            accepted={
                "pair_identity": f"pair-{seed}",
                "environment_identity": "env",
                "metrics": {"PA": _metrics(value), "OA": _metrics(value)},
                "elapsed_seconds": 1.0,
                "peak_gpu_bytes": 128,
                "peak_rss_bytes": 32,
            },
        )


def _write_receipt(output, job):
    output.mkdir()
    model = output / "weights"
    model.write_bytes(b"persistent checkpoint")
    references = {"path": str(model), "sha256": file_hash(model), "persistent": True}
    pa = "PA+clean-support" if job.method == "self_pu_full_meta" else "PA"
    metrics = {pa: _metrics(0.72), "OA": _metrics(0.75)}
    pair = {
        "study": campaign.STUDY,
        "dataset": job.dataset,
        "method": job.method,
        "seed": job.seed,
        "mode": "full",
        "c": 0.3,
        "sampling": "os",
        "source_commit": COMMIT,
    }
    env = {"cuda_available": True, "gpu_devices": [{"index": 0, "name": MODEL}]}
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "seed": job.seed,
                "failures": [],
                "paired_study": pair,
                "paired_arm": job.arm,
                "environment": env,
                "checkpoint_replay_verified": True,
                "selection": {key: {"checkpoint": references} for key in metrics},
                "test_results": metrics,
            }
        ),
        encoding="utf-8",
    )
    receipt = {
        **job.identity(),
        "schema_version": 1,
        "mode": "full",
        "study": campaign.STUDY,
        "source_commit": COMMIT,
        "pair_identity": strict_canonical_hash(pair),
        "environment_identity": strict_canonical_hash(env),
        "environment": env,
        "metrics": metrics,
        "manifest_sha256": file_hash(manifest),
        "checkpoints": [references],
        "elapsed_seconds": 12.0,
        "peak_gpu_bytes": 128,
        "peak_rss_bytes": 32,
    }
    (output / "accepted.json").write_text(json.dumps(receipt), encoding="utf-8")
    return receipt


def test_basic_dry_run_has_exact_grid_without_touching_host(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(campaign, "gpu_snapshot", lambda: pytest.fail("read GPU"))
    monkeypatch.setattr(campaign, "source_commit", lambda: pytest.fail("read git"))
    root = tmp_path / "untouched"
    assert campaign.main(["--dry-run", "--root", str(root), "--data-root", str(tmp_path)]) == 0
    grid = json.loads(capsys.readouterr().out)
    assert len(grid) == len({tuple(row.values()) for row in grid}) == 120
    assert [row["arm"] for row in grid[:2]] == ["frozen", "trainable"]
    assert not root.exists()


def test_guard_gpu_busy_compute_even_when_used_and_utilization_look_idle(monkeypatch):
    def output(args):
        if "--query-compute-apps=gpu_uuid,pid" in args:
            return f"{GPU}, 2135"
        return f"1, {GPU}, NVIDIA RTX A6000, 1, 0"

    monkeypatch.setattr(campaign, "_output", output)
    [device] = campaign.gpu_snapshot()
    assert device.compute_pids == (2135,)
    assert not device.idle


def test_invalid_nvml_snapshot_is_not_available(monkeypatch):
    answer = iter([f"1, {GPU}, NVIDIA RTX A6000, 1, 0", "unknown-card, 323"])
    monkeypatch.setattr(campaign, "_output", lambda args: next(answer))
    with pytest.raises(ValueError, match="inconsistent physical"):
        campaign.gpu_snapshot()


def test_canary_gate_uses_matched_pair_dependencies_not_accuracy():
    state = _state()
    enabled = {"robust_pu", "split_pu"}
    job = campaign.Job("cifar10", "robust_pu", 1, "frozen")
    assert campaign.waiting_reason(job, state, enabled, 0) == (
        "seed-zero paired canary not accepted"
    )
    _set_pair(state, "cifar10", 0)
    canary_t = campaign.Job("cifar10", "robust_pu", 0, "trainable")
    state[canary_t.key]["accepted"]["metrics"] = {"PA": _metrics(0.1), "OA": _metrics(0.1)}
    assert campaign.waiting_reason(job, state, enabled, 0) is None
    state[canary_t.key]["accepted"]["environment_identity"] = "different"
    assert campaign.waiting_reason(job, state, enabled, 0).startswith("unmatched paired receipt")
    other = campaign.Job("cifar10", "split_pu", 0, "frozen")
    assert campaign.waiting_reason(other, state, enabled, 0) is None


def test_fashion_waits_until_enabled_cifar_family_is_complete_or_isolated():
    state = _state()
    for seed in range(5):
        _set_pair(state, "cifar10", seed)
    enabled = {"robust_pu", "dist_pu"}
    job = campaign.Job("fashion_mnist", "robust_pu", 0, "trainable")
    assert campaign.waiting_reason(job, state, enabled, 0) == (
        "waiting for enabled CIFAR families to complete or be isolated"
    )
    failed = campaign.Job("cifar10", "dist_pu", 0, "trainable")
    state[failed.key]["status"] = "failed"
    assert campaign.waiting_reason(job, state, enabled, 0) is None
    assert "failed fit" in campaign.waiting_reason(
        campaign.Job("fashion_mnist", "dist_pu", 0, "trainable"), state, enabled, 0
    )


def test_full_meta_receipt_requires_correct_selection_and_replays_artifact_identity(tmp_path):
    job = campaign.Job("cifar10", "self_pu_full_meta", 3, "frozen")
    root = tmp_path / "result"
    expected = _write_receipt(root, job)
    assert campaign.verify_receipt(job, root, COMMIT, MODEL) == expected
    assert set(expected["metrics"]) == {"PA+clean-support", "OA"}
    (root / "weights").write_bytes(b"unexpected checkpoint")
    with pytest.raises(ValueError, match="selected checkpoint digest mismatch"):
        campaign.verify_receipt(job, root, COMMIT, MODEL)


def test_rejects_unrelated_checkpoint_even_when_digest_matches(tmp_path):
    job = campaign.Job("cifar10", "robust_pu", 0, "trainable")
    root = tmp_path / "result"
    receipt = _write_receipt(root, job)
    unrelated = root / "extra_weights"
    unrelated.write_bytes(b"persistent but not selected")
    receipt["checkpoints"] = [{"path": str(unrelated), "sha256": file_hash(unrelated)}]
    (root / "accepted.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="do not cover distinct PA/OA"):
        campaign.verify_receipt(job, root, COMMIT, MODEL)


@pytest.mark.skipif(sys.platform != "linux", reason="local host campaign uses flock")
def test_retry_once_even_if_controller_died_before_attempt_folder(tmp_path):
    job = campaign.Job("cifar10", "robust_pu", 0, "frozen")
    row = _state()[job.key]
    row.update(
        status="running",
        attempts=[{"root": str(tmp_path / "unlaunched"), "gpu": GPU, "launched_at": 0}],
    )
    campaign._reconcile(job, row, {}, 240)
    assert row["status"] == "pending"
    assert row["not_before"] == 240 + campaign.RETRY_BACKOFF_SECONDS
    root = tmp_path / "second"
    root.mkdir()
    row["status"] = "running"
    row["attempts"].append({"root": str(root), "gpu": GPU, "launched_at": 420})
    (root / "exit.json").write_text('{"exit_code":1}', encoding="utf-8")
    campaign._reconcile(job, row, {}, 421)
    assert row["status"] == "failed"
    assert len(row["attempts"]) == 2


@pytest.mark.skipif(sys.platform != "linux", reason="local host campaign uses flock")
def test_restart_does_not_treat_a_live_worker_as_failed(tmp_path):
    job = campaign.Job("cifar10", "robust_pu", 0, "frozen")
    row = _state()[job.key]
    row.update(status="running", attempts=[{"root": str(tmp_path), "gpu": GPU, "launched_at": 0}])
    with campaign._lock(tmp_path / "worker.lock"):
        campaign._reconcile(job, row, {}, 300)
    assert row["status"] == "running"


def test_five_seed_summary_never_mixes_hardware_or_missing_pairs(tmp_path):
    state = _state()
    for seed in range(4):
        _set_pair(state, "cifar10", seed)
    report = campaign.write_summary(tmp_path, state)
    assert report["paired_seed_count"] == 4
    assert report["five_seed_metric_rows"] == 0
    _set_pair(state, "cifar10", 4)
    report = campaign.write_summary(tmp_path, state)
    assert report["five_seed_metric_rows"] == 10
    with (tmp_path / "paired_five_seed.csv").open(encoding="utf-8-sig") as stream:
        result = next(csv.DictReader(stream))
    assert result["wins"] == "5"
    assert float(result["delta_mean"]) == pytest.approx(0.1)
    for arm in campaign.ARMS:
        job = campaign.Job("cifar10", "robust_pu", 4, arm)
        state[job.key]["accepted"]["environment_identity"] = "another-env"
    assert campaign.write_summary(tmp_path, state)["five_seed_metric_rows"] == 0


def test_campaign_advances_all_enabled_work_and_keeps_unapproved_methods_held(
    monkeypatch, tmp_path
):
    config = Namespace(
        root=tmp_path,
        methods="robust_pu",
        ready_file=None,
        gpu_model=MODEL,
        gpus=(1, 2),
        max_parallel=2,
        poll_seconds=2,
    )
    monkeypatch.setattr(campaign, "source_commit", lambda: COMMIT)
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        campaign,
        "gpu_snapshot",
        lambda: [campaign.GPU(i, f"GPU-{i}", MODEL, 1, 0, ()) for i in config.gpus],
    )
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda _: Namespace(free=3 * campaign.MIN_DISK_BYTES),
    )
    dispatched = []

    def successful_launch(job, row, _device, _config, _state_path, _state):
        dispatched.append(job)
        row.update(
            status="completed",
            accepted={
                "pair_identity": f"{job.dataset}/{job.method}/{job.seed}",
                "environment_identity": "fixed-env",
                "metrics": {"PA": _metrics(0.7), "OA": _metrics(0.8)},
                "elapsed_seconds": 1.0,
                "peak_gpu_bytes": 128,
                "peak_rss_bytes": 32,
            },
        )

    monkeypatch.setattr(campaign, "_launch", successful_launch)
    assert campaign.campaign_loop(config, {"source_commit": COMMIT}) == 2
    assert len(dispatched) == len({job.key for job in dispatched}) == 20
    assert all(job.method == "robust_pu" for job in dispatched)
    assert all(job.dataset == "cifar10" for job in dispatched[:10])
    assert all(job.dataset == "fashion_mnist" for job in dispatched[10:])
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"] == {"completed": 20, "held": 100}
    assert summary["complete"] is False
