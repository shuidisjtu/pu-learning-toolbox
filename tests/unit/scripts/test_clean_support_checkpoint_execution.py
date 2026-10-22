"""CLI-to-artifact integration for explicit CPU support checkpoint experiments."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_clean_support_experiment as script  # noqa: E402

from pu_toolbox.experiment.support_data import (  # noqa: E402
    prepare_support_dataset,
    save_support_dataset,
)

pytestmark = pytest.mark.unit


def product(tmp_path):
    rng = np.random.RandomState(0)
    features, truth = rng.normal(size=(400, 5)), np.tile([0, 1], 200)
    bundle, support, metadata = prepare_support_dataset(
        features,
        truth,
        dataset="spambase",
        seed=0,
        support_size=40,
        support_seed=7,
        group_policy="not_applicable",
        source="synthetic checkpoint CLI fixture",
    )
    path = tmp_path / "product"
    save_support_dataset(bundle, support, metadata, path)
    return path


def parameters(method):
    common = {
        "device": "cpu",
        "hidden_dim": 4,
        "batch_size": 32,
        "max_epochs": 3,
        "warmup_epochs": 0,
    }
    return common | (
        {"self_paced_start": 0, "self_paced_end": 1, "distill_start": 2}
        if method == "self_pu"
        else {"num_clusters": 2}
    )


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
def test_basic_actual_cpu_meta_checkpoints_both_selected_models_and_replay(method, tmp_path):
    plan = script.build_plan(
        product(tmp_path),
        method=method,
        c_token="0.5",
        parameters=parameters(method),
        selection_scope="meta_checkpoints",
        checkpoint_bytes_per_component=65536,
    )
    assert plan["protocols"] == ["PA+clean-support", "OA"]
    assert plan["reclaim_unselected_checkpoints"] is True
    assert "pu_toolbox/experiment/checkpoints.py" in plan["source_files_sha256"]
    output = tmp_path / "results"
    summary = script.execute(plan, output)
    assert summary["accepted_engineering_run"] and summary["selected_model_replay_verified"]
    assert set(summary["test"]) == {"PA+clean-support", "OA"}
    assert summary["extra_supervision"]["pa_eligible"] is False
    assert summary["extra_supervision"]["pa_clean_support_eligible"] is True
    assert len(list(output.glob("selected_*.pkl"))) == 2
    for name in summary["selection"]:
        with (output / f"selected_{name}.pkl").open("rb") as stream:
            saved = pickle.load(stream)
        assert saved["selection"]["checkpoint"]["persistent"]
        assert not saved["selection"]["checkpoint"]["reclaimed"]
        assert saved["model"].selection_threshold == saved["selection"]["threshold"]


@pytest.mark.parametrize(
    "scope,size",
    [("meta_checkpoints", None), ("meta_checkpoints", 0), ("terminal", 65536), ("unknown", 65536)],
)
def test_param_ambiguous_scope_or_missing_storage_budget_rejected(scope, size, tmp_path):
    with pytest.raises(ValueError):
        script.build_plan(
            product(tmp_path),
            method="lagam",
            c_token="0.5",
            parameters=parameters("lagam"),
            selection_scope=scope,
            checkpoint_bytes_per_component=size,
        )


def test_edge_changed_checkpoint_budget_after_plan_cannot_train(tmp_path, monkeypatch):
    plan = script.build_plan(
        product(tmp_path),
        method="lagam",
        c_token="0.5",
        parameters=parameters("lagam"),
        selection_scope="meta_checkpoints",
        checkpoint_bytes_per_component=65536,
    )
    plan["checkpoint_bytes_per_component"] += 1

    def never(*args, **kwargs):
        raise AssertionError("changed plan must be rejected before training")

    monkeypatch.setattr(script.ExperimentRunner, "fit", never)
    with pytest.raises(ValueError, match="changed after planning"):
        script.execute(plan, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_determ_cli_dry_run_exposes_explicit_scope_without_training(tmp_path, monkeypatch, capsys):
    root = product(tmp_path)
    params = tmp_path / "params.json"
    params.write_text(json.dumps(parameters("lagam")))

    def never(*args, **kwargs):
        raise AssertionError("dry-run must not fit")

    monkeypatch.setattr(script.ExperimentRunner, "fit", never)
    assert (
        script.main(
            [
                "--artifact-dir",
                str(root),
                "--method",
                "lagam",
                "--c",
                "0.5",
                "--parameters",
                str(params),
                "--selection-scope",
                "meta_checkpoints",
                "--checkpoint-bytes-per-component",
                "65536",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["selection_scope"] == "meta_checkpoints"
    assert payload["protocols"] == ["PA+clean-support", "OA"]
