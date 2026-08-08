"""Tests for the claim-safe PUSB Table 2 benchmark runner."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.assigned_methods import pusb_table2_benchmark
from benchmarks.assigned_methods.pusb_table2_benchmark import (
    COMPATIBILITY_POLICY,
    STRICT_POLICY,
    build_dataset_plan,
    load_config,
    run_benchmark,
    summarize_plan,
)

pytestmark = pytest.mark.unit


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


@pytest.mark.unit
def test_basic_load_config_accepts_matching_strict_policy(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    assert load_config(path)["sampling_policy"] == STRICT_POLICY


@pytest.mark.unit
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


@pytest.mark.unit
def test_edge_strict_policy_excludes_whole_cell_when_one_split_is_undersized():
    y = np.r_[np.ones(4, dtype=int), np.zeros(36, dtype=int)]
    config = _config()
    config["experiment"]["class_priors"] = [0.8]
    plan = build_dataset_plan("mushrooms", y, config)

    assert not plan["cell_all_repetitions_feasible"].any()
    assert not plan["selected_for_execution"].any()
    assert summarize_plan(plan)["excluded_cells"] == 1


@pytest.mark.unit
def test_basic_compatibility_policy_selects_and_labels_undersized_trials():
    y = np.r_[np.ones(4, dtype=int), np.zeros(36, dtype=int)]
    config = _config(COMPATIBILITY_POLICY)
    config["experiment"]["class_priors"] = [0.8]
    plan = build_dataset_plan("mushrooms", y, config)
    summary = summarize_plan(plan)

    assert plan["selected_for_execution"].all()
    assert summary["selected_trials"] == 2
    assert summary["undersized_selected_trials"] > 0


@pytest.mark.unit
def test_determ_plan_preserves_cross_cell_continuous_seed_order():
    config = _config()
    config["experiment"]["unlabeled_sizes"] = [10, 20]
    y = np.tile([0, 1], 50)
    plan = build_dataset_plan("mushrooms", y, config)
    assert plan["seed"].tolist() == [10, 11, 12, 13]


@pytest.mark.unit
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


@pytest.mark.unit
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
