"""Candidate launch controls do not waive protocol gates or steal busy GPUs."""

import json
from pathlib import Path

import pytest

from scripts import run_candidate_readiness_campaign as campaign

pytestmark = pytest.mark.unit


def test_basic_determ_foreign_context_and_memory_exclude_gpu(monkeypatch):
    outputs = iter(["0, GPU-a, 49000\n5, GPU-b, 49000\n6, GPU-c, 30000\n", "GPU-b\n"] * 2)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **k: next(outputs))
    assert campaign.available_gpus([0, 5, 6]) == [0]
    assert campaign.available_gpus([0, 5, 6]) == [0]


@pytest.mark.parametrize("indices", [[], [0, 0], [0, 5, 6], [1]])
def test_param_invalid_gpu_scope_refused(tmp_path, indices):
    with pytest.raises(ValueError):
        campaign.run(tmp_path, indices)


def test_edge_exclusive_output_preserves_existing_content(tmp_path):
    path = tmp_path / "status.json"
    campaign.write(path, {"old": True}, exclusive=True)
    with pytest.raises(FileExistsError):
        campaign.write(path, {"new": True}, exclusive=True)
    assert json.loads(path.read_text()) == {"old": True}


def test_basic_explicit_worker_command_never_changes_source_budget(tmp_path):
    metadata = dict(input_directory="/full/split", input_evidence={"population_prior": 0.4})
    job = dict(label="gradpu", method="gradpu", view="ts")
    command = campaign.command(metadata, job, tmp_path)
    assert command[command.index("--view") + 1] == "ts"
    assert command[command.index("--class-prior") + 1] == "0.4"
    assert command[command.index("--c") + 1] == "0.1"
    assert command[command.index("--data-dir") + 1] == "/full/split"
    assert command[command.index("--parameters") + 1] == str(tmp_path / "gradpu.parameters.json")
    assert "--execute" in command


def test_edge_subset_receipt_rejected_before_loading_training(tmp_path):
    receipt = dict(dataset="cifar10", seed=0, role_sizes={"train": 24})
    (tmp_path / "split_manifest.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="no subset fallback"):
        campaign.validate_split(tmp_path)


def test_basic_prepare_explicit_full_budgets(tmp_path, monkeypatch):
    import sys

    from scripts import prepare_candidate_image_resource_plan as planner

    monkeypatch.setitem(sys.modules, "prepare_candidate_image_resource_plan", planner)
    monkeypatch.setattr(campaign, "validate_split", lambda p: {"population_prior": 0.4})
    monkeypatch.setattr(
        campaign.subprocess,
        "check_output",
        lambda argv, **kwargs: b"" if "status" in argv else "test-commit\n",
    )
    output = tmp_path / "campaign"
    plan = campaign.prepare(Path("/full/split"), output)
    jobs = {j["method"]: j for j in plan["jobs"]}
    assert len(plan["jobs"]) == 6
    assert jobs["gradpu"]["parameters"]["max_epochs"] == 200
    assert jobs["pan"]["parameters"]["max_epochs"] == 100
    assert jobs["vpu"]["parameters"]["max_epochs"] == 100
    assert jobs["p3mix"]["parameters"]["batching_policy"] == "merged_shuffle"
    assert jobs["cvir"]["parameters"]["warm_start_epochs"] == 1
    assert jobs["cvir"]["parameters"]["max_epochs"] == 20
    assert not plan["formal_admission"] and not plan["frozen_budget_approved"]
    assert all(j["parameters"]["device"] == "cuda" for j in plan["jobs"])


@pytest.mark.parametrize("fail,expected_launches", [(False, 6), (True, 2)])
def test_param_service_finishes_or_holds_after_worker_failure(
    tmp_path, monkeypatch, fail, expected_launches
):
    import sys

    from scripts import prepare_candidate_image_resource_plan as planner

    monkeypatch.setitem(sys.modules, "prepare_candidate_image_resource_plan", planner)
    monkeypatch.setattr(campaign, "validate_split", lambda p: {"population_prior": 0.4})
    monkeypatch.setattr(
        campaign.subprocess,
        "check_output",
        lambda argv, **kw: b"" if "status" in argv else "test-commit\n",
    )
    output = tmp_path / "campaign"
    campaign.prepare(Path("/full/split"), output)
    monkeypatch.setattr(campaign, "available_gpus", lambda indices: list(indices))
    monkeypatch.setattr(campaign.signal, "signal", lambda *args: None)
    monkeypatch.setattr(campaign, "lock_controller", lambda path: path.open("a"))
    monkeypatch.setattr(campaign.time, "sleep", lambda value: None)
    launched = []

    class Worker:
        def __init__(self, argv, **kwargs):
            self.pid = len(launched) + 100
            self.failed = fail and not launched
            launched.append(argv)
            assert kwargs["start_new_session"] is True
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] in ("0", "5")
            root = Path(argv[argv.index("--out-dir") + 1])
            root.mkdir()
            campaign.write(root / "summary.json", {"accepted_engineering_run": not self.failed})

        def poll(self):
            return 1 if self.failed else 0

    monkeypatch.setattr(campaign.subprocess, "Popen", Worker)
    campaign.run(output, [0, 5])
    state = json.loads((output / "status.json").read_text())
    assert len(launched) == expected_launches
    assert not state["formal_admission"]
    assert sum(j["status"] == "held_after_failure_or_warning" for j in state["jobs"]) == (
        4 if fail else 0
    )
