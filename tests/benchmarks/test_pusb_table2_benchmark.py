"""Tests for the claim-safe PUSB Table 2 benchmark runner."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.assigned_methods import pusb_table2_benchmark, pusb_table2_parallel
from benchmarks.assigned_methods.pusb_table2_aggregate import aggregate_shards
from benchmarks.assigned_methods.pusb_table2_benchmark import (
    COMPATIBILITY_POLICY,
    STRICT_POLICY,
    build_dataset_plan,
    load_config,
    run_benchmark,
    summarize_plan,
)
from benchmarks.assigned_methods.pusb_table2_parallel import _shard_command, run_parallel
from benchmarks.assigned_methods.pusb_table2_report import (
    build_statistical_summary,
    validate_strict_trials,
    write_report,
)

pytestmark = [pytest.mark.unit, pytest.mark.paper]


def _config(policy=STRICT_POLICY):
    fidelity = {
        STRICT_POLICY: "paper_protocol_strict_feasible_subset",
        COMPATIBILITY_POLICY: "official_released_compatibility",
    }[policy]
    return {
        "schema_version": 1,
        "protocol": "pusb_table2_benchmark",
        "fidelity_level": fidelity,
        "paper_claim": False,
        "sampling_policy": policy,
        "datasets": ["mushrooms"],
        "experiment": {
            "initial_seed": 10,
            "repetitions": 2,
            "class_priors": [0.5],
            "unlabeled_sizes": [10],
            "positive_size": 5,
            "test_size": 10,
            "holdout_size": 20,
            "selection_probability_power": 2,
        },
        "model": {"variant": "test", "parameters": {}},
        "density_ratio": {"enabled": False},
    }


def _single_trial_plan():
    return pd.DataFrame(
        [
            {
                "dataset": "mushrooms",
                "seed": 10,
                "repetition": 0,
                "unlabeled_size": 10,
                "class_prior": 0.5,
                "strictly_feasible": True,
                "actual_unlabeled_size": 10,
                "actual_test_size": 10,
                "failure_reasons": "",
                "cell_all_repetitions_feasible": True,
                "selected_for_execution": True,
                "sampling_policy": STRICT_POLICY,
                "fidelity_level": "paper_protocol_strict_feasible_subset",
            }
        ]
    )


def test_basic_load_config_accepts_matching_strict_policy(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    assert load_config(path)["sampling_policy"] == STRICT_POLICY


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("paper_claim", True, "paper_claim"),
        ("sampling_policy", "silent", "sampling_policy"),
        ("fidelity_level", "paper_protocol", "fidelity_level"),
    ],
)
def test_param_load_config_rejects_claim_unsafe_combinations(tmp_path, field, value, message):
    config = _config()
    config[field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_edge_strict_policy_excludes_whole_cell_when_one_split_is_undersized():
    y = np.r_[np.ones(4, dtype=int), np.zeros(36, dtype=int)]
    config = _config()
    config["experiment"]["class_priors"] = [0.8]
    plan = build_dataset_plan("mushrooms", y, config)

    assert not plan["cell_all_repetitions_feasible"].any()
    assert not plan["selected_for_execution"].any()
    assert summarize_plan(plan)["excluded_cells"] == 1


def test_basic_compatibility_policy_selects_and_labels_undersized_trials():
    y = np.r_[np.ones(4, dtype=int), np.zeros(36, dtype=int)]
    config = _config(COMPATIBILITY_POLICY)
    config["experiment"]["class_priors"] = [0.8]
    plan = build_dataset_plan("mushrooms", y, config)
    summary = summarize_plan(plan)

    assert plan["selected_for_execution"].all()
    assert summary["selected_trials"] == 2
    assert summary["undersized_selected_trials"] > 0


def test_determ_plan_preserves_cross_cell_continuous_seed_order():
    config = _config()
    config["experiment"]["unlabeled_sizes"] = [10, 20]
    y = np.tile([0, 1], 50)
    plan = build_dataset_plan("mushrooms", y, config)
    assert plan["seed"].tolist() == [10, 11, 12, 13]


def test_basic_plan_only_writes_claim_safe_artifacts(tmp_path, monkeypatch):
    plan = _single_trial_plan()
    monkeypatch.setattr(
        pusb_table2_benchmark,
        "build_benchmark_plan",
        lambda *args, **kwargs: (plan, {"mushrooms": {"shape": [40, 2]}}),
    )

    trials = run_benchmark(
        _config(), data_root=tmp_path, output_dir=tmp_path / "output", plan_only=True
    )

    assert trials.empty
    for name in ("trial_plan.csv", "excluded_cells.csv", "plan_manifest.json"):
        assert (tmp_path / "output" / name).is_file()
    manifest = json.loads((tmp_path / "output" / "plan_manifest.json").read_text())
    assert manifest["status"] == "planned_not_executed"
    assert manifest["paper_claim"] is False


def test_determ_execution_checkpoints_and_resume_skips_completed_trial(tmp_path, monkeypatch):
    plan = _single_trial_plan()
    monkeypatch.setattr(
        pusb_table2_benchmark,
        "build_benchmark_plan",
        lambda *args, **kwargs: (plan, {"mushrooms": {"shape": [40, 2]}}),
    )
    monkeypatch.setattr(
        pusb_table2_benchmark,
        "load_table2_dataset",
        lambda *args, **kwargs: (np.ones((40, 2)), np.tile([0, 1], 20), {}),
    )
    calls = []

    def fake_run_trials(config, x, y):
        calls.append(config["experiment"]["seeds"][0])
        return pd.DataFrame(
            [
                {
                    "protocol": config["protocol"],
                    "fidelity_level": config["fidelity_level"],
                    "paper_claim": False,
                    "seed": 10,
                    "class_prior": 0.5,
                    "unlabeled_size": 10,
                    "actual_unlabeled_size": 10,
                    "requested_test_size": 10,
                    "test_size": 10,
                    "quantile_accuracy": 0.8,
                    "quantile_balanced_accuracy": 0.8,
                    "roc_auc": 0.9,
                    "elapsed_seconds": 0.1,
                }
            ]
        )

    monkeypatch.setattr(pusb_table2_benchmark, "run_trials", fake_run_trials)
    output = tmp_path / "output"
    first = run_benchmark(_config(), data_root=tmp_path, output_dir=output)
    second = run_benchmark(_config(), data_root=tmp_path, output_dir=output, resume=True)

    assert len(first) == len(second) == 1
    assert calls == [10]
    assert json.loads((output / "run_manifest.json").read_text())["status"] == "completed"


def test_determ_shards_partition_selected_trials_without_overlap(tmp_path, monkeypatch):
    plan = pd.concat([_single_trial_plan() for _ in range(4)], ignore_index=True)
    plan["seed"] = [10, 11, 12, 13]
    monkeypatch.setattr(
        pusb_table2_benchmark,
        "build_benchmark_plan",
        lambda *args, **kwargs: (plan, {"mushrooms": {"shape": [40, 2]}}),
    )
    output = tmp_path / "output"
    run_benchmark(
        _config(),
        data_root=tmp_path,
        output_dir=output,
        plan_only=True,
        shard_count=2,
        shard_index=1,
    )
    written = pd.read_csv(output / "trial_plan.csv")
    assert written.loc[written["selected_for_shard"], "seed"].tolist() == [11, 13]
    manifest = json.loads((output / "plan_manifest.json").read_text())
    assert manifest["shard_selected_trials"] == 2


def test_edge_resume_rejects_different_shard_scope(tmp_path, monkeypatch):
    plan = _single_trial_plan()
    monkeypatch.setattr(
        pusb_table2_benchmark,
        "build_benchmark_plan",
        lambda *args, **kwargs: (plan, {"mushrooms": {"shape": [40, 2]}}),
    )
    output = tmp_path / "output"
    run_benchmark(
        _config(),
        data_root=tmp_path,
        output_dir=output,
        plan_only=True,
        shard_count=2,
        shard_index=0,
    )
    with pytest.raises(ValueError, match="shard scope differs"):
        run_benchmark(
            _config(),
            data_root=tmp_path,
            output_dir=output,
            resume=True,
            plan_only=True,
            shard_count=2,
            shard_index=1,
        )


def test_basic_aggregate_shards_requires_exact_plan_keys(tmp_path):
    plan = pd.concat([_single_trial_plan() for _ in range(2)], ignore_index=True)
    plan["seed"] = [10, 11]
    plan_path = tmp_path / "plan.csv"
    plan.to_csv(plan_path, index=False)
    shard_root = tmp_path / "shards"
    config_hash = hashlib.sha256(b"{}").hexdigest()
    for index, seed in enumerate((10, 11)):
        shard = shard_root / f"shard-{index:02d}"
        shard.mkdir(parents=True)
        trials = pd.DataFrame(
            [
                {
                    "dataset": "mushrooms",
                    "seed": seed,
                    "class_prior": 0.5,
                    "unlabeled_size": 10,
                    "quantile_accuracy": 0.8,
                    "roc_auc": 0.9,
                    "elapsed_seconds": 0.1,
                }
            ]
        )
        trials.to_csv(shard / "trials.csv", index=False)
        (shard / "resolved_config.json").write_text("{}", encoding="utf-8")
        manifest = {
            "status": "completed",
            "execution_scope": {"shard_count": 2, "shard_index": index},
            "paper_claim": False,
            "n_completed_trials": 1,
            "config_sha256": config_hash,
            "fidelity_level": "paper_protocol_strict_feasible_subset",
        }
        (shard / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    trials = aggregate_shards(
        shard_root,
        plan_path=plan_path,
        output_dir=tmp_path / "aggregate",
        shard_count=2,
    )

    assert len(trials) == 2
    manifest = json.loads((tmp_path / "aggregate" / "aggregation_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["n_trials"] == manifest["n_expected_trials"] == 2


def test_basic_shard_command_adds_resume_only_for_existing_scope(tmp_path):
    arguments = {
        "config": tmp_path / "config.json",
        "data_root": tmp_path / "data",
        "shard_dir": tmp_path / "shard",
        "shard_count": 4,
        "shard_index": 2,
    }
    arguments["shard_dir"].mkdir()
    assert "--resume" not in _shard_command(**arguments)
    (arguments["shard_dir"] / "resolved_config.json").write_text("{}", encoding="utf-8")
    command = _shard_command(**arguments)
    assert command[-1] == "--resume"
    assert command[command.index("--shard-index") + 1] == "2"


def test_determ_parallel_runner_retries_only_failed_shards_then_aggregates(tmp_path, monkeypatch):
    calls = []
    aggregated = []

    def fake_run_shard(**kwargs):
        index = kwargs["shard_index"]
        calls.append(index)
        return index, int(index == 1 and calls.count(1) == 1)

    monkeypatch.setattr(pusb_table2_parallel, "_run_shard", fake_run_shard)
    monkeypatch.setattr(
        pusb_table2_parallel,
        "aggregate_shards",
        lambda *args, **kwargs: aggregated.append(kwargs["shard_count"]),
    )
    run_parallel(
        config=tmp_path / "config.json",
        data_root=tmp_path / "data",
        shard_root=tmp_path / "shards",
        plan_path=tmp_path / "plan.csv",
        aggregate_output=tmp_path / "aggregate",
        shard_count=3,
        workers=2,
        retries=1,
    )

    assert sorted(calls) == [0, 1, 1, 2]
    assert aggregated == [3]


def test_basic_statistical_report_uses_paired_differences():
    trials = _report_trials()
    summary = build_statistical_summary(trials)
    assert len(summary) == 1
    assert summary.loc[0, "quantile_accuracy_mean"] == pytest.approx(0.75)
    assert summary.loc[0, "paired_accuracy_difference_mean"] == pytest.approx(0.1)
    assert summary.loc[0, "paired_accuracy_difference_win_rate"] == 1.0


def test_edge_strict_report_rejects_undersized_samples():
    trials = _report_trials()
    trials.loc[0, "actual_unlabeled_size"] = 9
    with pytest.raises(ValueError, match="undersized unlabeled"):
        validate_strict_trials(trials, expected_trials=2, expected_repetitions=2)


def test_determ_report_writes_csv_json_and_markdown(tmp_path):
    trials_path = tmp_path / "trials.csv"
    _report_trials().to_csv(trials_path, index=False)
    summary, report = write_report(
        trials_path,
        output_dir=tmp_path / "report",
        expected_trials=2,
        expected_repetitions=2,
    )
    assert len(summary) == 1
    assert report["paper_claim"] is False
    for name in ("statistical_summary.csv", "benchmark_report.json", "REPORT.md"):
        assert (tmp_path / "report" / name).is_file()


def _report_trials():
    return pd.DataFrame(
        {
            "dataset": ["mushrooms", "mushrooms"],
            "seed": [10, 11],
            "class_prior": [0.5, 0.5],
            "unlabeled_size": [10, 10],
            "actual_unlabeled_size": [10, 10],
            "requested_test_size": [10, 10],
            "test_size": [10, 10],
            "paper_claim": [False, False],
            "sampling_policy": [STRICT_POLICY, STRICT_POLICY],
            "strictly_feasible_split": [True, True],
            "cell_all_repetitions_feasible": [True, True],
            "quantile_accuracy": [0.7, 0.8],
            "quantile_balanced_accuracy": [0.7, 0.8],
            "roc_auc": [0.8, 0.9],
            "density_ratio_accuracy": [0.6, 0.7],
            "density_ratio_balanced_accuracy": [0.6, 0.7],
            "density_ratio_roc_auc": [0.7, 0.8],
            "cv_all_converged": [True, True],
            "optimizer_success": [True, True],
            "elapsed_seconds": [1.0, 1.1],
            "density_ratio_elapsed_seconds": [0.5, 0.6],
            "sigma": [1.0, 1.0],
            "reg_lambda": [0.1, 0.1],
        }
    )
