"""Tests for the assigned-method benchmark runner."""

# Test variables mirror the public estimator API.
# ruff: noqa: N806

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.assigned_methods.runner import (
    _case_control_data,
    _sar_data,
    load_config,
    run_benchmark,
    run_trials,
    summarize_trials,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def benchmark_config():
    return {
        "schema_version": 1,
        "protocol": "clean_room",
        "seeds": [3, 7],
        "data": {
            "n_positive": 20,
            "n_unlabeled": 50,
            "n_train": 80,
            "n_test": 60,
            "n_features": 3,
            "class_prior": 0.3,
            "separation": 2.0,
            "label_frequency": 0.4,
            "sar_strength": 1.0,
            "sar_mechanism": "linear",
        },
        "methods": {
            "class_prior_estimation": {
                "variant": "test_penl1",
                "parameters": {"n_centers": 20},
            },
            "pusb": {
                "variant": "test_logistic",
                "parameters": {},
            },
        },
    }


@pytest.mark.unit
def test_basic_load_config_runs(tmp_path, benchmark_config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(benchmark_config), encoding="utf-8")
    assert load_config(path) == benchmark_config


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("protocol", "paper_like", "clean_room"),
        ("seeds", [], "seeds"),
    ],
)
@pytest.mark.unit
def test_invalid_config_raises_error(tmp_path, benchmark_config, field, value, message):
    benchmark_config[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(benchmark_config), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.unit
def test_unknown_method_validation_error(tmp_path, benchmark_config):
    benchmark_config["methods"]["unknown"] = {"variant": "bad", "parameters": {}}
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(benchmark_config), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_config(path)


@pytest.mark.unit
def test_case_control_output_shapes_and_positive_counts(benchmark_config):
    X_train, y_pu, X_test, y_test = _case_control_data(
        np.random.default_rng(4),
        benchmark_config["data"],
    )
    assert X_train.shape == (70, 3)
    assert X_test.shape == (60, 3)
    assert y_pu.sum() == 20
    assert set(np.unique(y_test)) == {0, 1}


@pytest.mark.unit
def test_sar_extreme_seed_reproducibility(benchmark_config):
    first = _sar_data(11, benchmark_config["data"], "linear")
    second = _sar_data(11, benchmark_config["data"], "linear")
    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left, right)
    assert first[1].sum() > 0


@pytest.mark.unit
def test_multiseed_trials_are_deterministic(benchmark_config):
    first = run_trials(benchmark_config)
    second = run_trials(benchmark_config)
    metric_columns = sorted(set(first.columns) - {"elapsed_seconds"})
    assert first[metric_columns].equals(second[metric_columns])
    assert len(first) == 4


@pytest.mark.unit
def test_scar_sar_mechanism_expansion_outputs_paired_rows(benchmark_config):
    benchmark_config["seeds"] = [5]
    benchmark_config["methods"] = {
        "pusb": {
            "variant": "test_logistic",
            "parameters": {},
        }
    }
    benchmark_config["data"]["sar_mechanisms"] = ["scar", "linear", "nonlinear"]
    trials = run_trials(benchmark_config)
    assert len(trials) == 3
    assert set(trials["labeling_mechanism"]) == {"scar", "linear", "nonlinear"}
    assert trials["seed"].nunique() == 1
    assert np.isfinite(trials["posterior_spearman"]).all()
    assert trials["pairwise_ranking_accuracy"].between(0, 1).all()


@pytest.mark.unit
def test_summary_counts_all_seed_outputs(benchmark_config):
    summary = summarize_trials(run_trials(benchmark_config))
    assert not summary.empty
    assert set(summary["method"]) == {"class_prior_estimation", "pusb"}
    assert set(summary["n"]) == {2}


@pytest.mark.unit
def test_benchmark_writes_all_output_artifacts(tmp_path, benchmark_config):
    trials, summary = run_benchmark(benchmark_config, tmp_path)
    assert len(trials) == 4
    assert not summary.empty
    for name in ("trials.csv", "summary.csv", "resolved_config.json", "run_manifest.json"):
        assert (tmp_path / name).is_file()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["paper_claim"] is False
    assert manifest["n_trials"] == 4
    assert len(manifest["runner_sha256"]) == 64
    assert manifest["git_worktree_dirty"] in {True, False, None}


@pytest.mark.unit
def test_dist_pu_optional_backend_basic_fit(benchmark_config):
    pytest.importorskip("torch")
    benchmark_config["seeds"] = [0]
    benchmark_config["methods"] = {
        "dist_pu": {
            "variant": "test_mlp",
            "parameters": {
                "hidden_dim": 4,
                "epochs": 1,
                "mixup_weight": 0.0,
                "device": "cpu",
            },
        }
    }
    trials = run_trials(benchmark_config)
    assert len(trials) == 1
    assert np.isfinite(trials.loc[0, "roc_auc"])


@pytest.mark.unit
def test_official_lock_and_all_config_outputs_are_valid_json():
    config_root = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "assigned_methods" / "configs"
    )
    documents = {
        path: json.loads(path.read_text(encoding="utf-8")) for path in config_root.rglob("*.json")
    }
    assert len(documents) == 10
    source_lock = documents[config_root / "official_sources.lock.json"]
    assert source_lock["sources"]["class_prior_estimation"]["paper_doi"] == (
        "10.1007/s10994-016-5604-6"
    )
    assert source_lock["sources"]["recpe"]["commit"]
    assert source_lock["sources"]["lbe"]["sha256"]
