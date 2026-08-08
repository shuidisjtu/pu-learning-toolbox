"""Tests for the official-data PUSB compatibility runner."""

# Test fixtures follow the runner's X/y convention.
# ruff: noqa: N803, N806

import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.assigned_methods import pusb_official_data
from benchmarks.assigned_methods.pusb_official_data import (
    construct_official_split,
    load_config,
    run_benchmark,
    run_trials,
)


def _classification_data(seed=4):
    rng = np.random.RandomState(seed)
    positive = rng.normal(1.0, 0.7, size=(360, 4))
    negative = rng.normal(-0.5, 0.9, size=(540, 4))
    return np.vstack((positive, negative)), np.r_[np.ones(360), np.zeros(540)].astype(int)


def _config():
    return {
        "schema_version": 1,
        "protocol": "pusb_official_data",
        "fidelity_level": "smoke",
        "paper_claim": False,
        "dataset": {"path": "ijcnn1", "sha256": "test-hash"},
        "experiment": {
            "seeds": [12],
            "class_priors": [0.4],
            "unlabeled_sizes": [40],
            "positive_size": 20,
            "test_size": 40,
            "holdout_size": 200,
            "selection_probability_power": 2,
        },
        "model": {
            "variant": "test_rbf",
            "parameters": {
                "n_basis": 6,
                "cv": 2,
                "sigma_grid": [0.5],
                "reg_grid": [0.1],
                "max_iter": 50,
            },
        },
        "limitations": ["test"],
    }


def _fake_density_ratio_fitter(x, y, **parameters):
    assert len(x) > 0 and len(y) > 0
    assert parameters["alpha"] == 0
    np.random.random()
    return SimpleNamespace(
        kernel_info=SimpleNamespace(sigma=0.5),
        lambda_=0.1,
        compute_density_ratio=lambda coordinates: np.asarray(coordinates)[:, 0],
    )


@pytest.mark.unit
def test_basic_official_split_has_locked_group_sizes_and_prior():
    X, y = _classification_data()
    split = construct_official_split(
        X,
        y,
        seed=12,
        class_prior=0.4,
        unlabeled_size=40,
        positive_size=20,
        test_size=40,
        holdout_size=200,
        selection_probability_power=2,
    )

    assert split["X_pu"].shape == (60, 4)
    assert split["y_pu"].sum() == 20
    assert split["X_test"].shape == (40, 4)
    assert split["y_test"].mean() == 0.4
    assert np.all((split["selection_probability"] >= 0) & (split["selection_probability"] <= 1))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("protocol", "clean_room", "protocol"),
        ("paper_claim", True, "paper_claim"),
        ("fidelity_level", "paper", "fidelity_level"),
    ],
)
def test_param_invalid_official_data_config_is_rejected(tmp_path, field, value, message):
    config = _config()
    config[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.unit
def test_edge_official_split_rejects_impossible_holdout():
    X, y = _classification_data()
    with pytest.raises(ValueError, match="holdout_size"):
        construct_official_split(
            X,
            y,
            seed=12,
            class_prior=0.4,
            unlabeled_size=40,
            holdout_size=len(X),
        )


@pytest.mark.unit
def test_determ_official_data_trial_is_reproducible():
    X, y = _classification_data()
    first = run_trials(_config(), X, y)
    second = run_trials(_config(), X, y)
    columns = sorted(set(first.columns) - {"elapsed_seconds"})

    assert first[columns].equals(second[columns])
    assert first.loc[0, "paper_claim"] == np.False_
    assert 0.0 <= first.loc[0, "roc_auc"] <= 1.0


@pytest.mark.unit
def test_determ_density_ratio_comparator_is_routed_and_restores_numpy_rng():
    X, y = _classification_data()
    config = _config()
    config["density_ratio"] = {
        "enabled": True,
        "parameters": {
            "kernel_num": 5,
            "sigma_range": [0.5],
            "lambda_range": [0.1],
        },
    }
    np.random.seed(99)
    state_before = np.random.get_state()

    trials = run_trials(config, X, y, density_ratio_fitter=_fake_density_ratio_fitter)
    state_after = np.random.get_state()

    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    assert trials.loc[0, "density_ratio_sigma"] == 0.5
    assert trials.loc[0, "density_ratio_reg_lambda"] == 0.1
    assert 0.0 <= trials.loc[0, "density_ratio_accuracy"] <= 1.0
    assert 0.0 <= trials.loc[0, "density_ratio_roc_auc"] <= 1.0


@pytest.mark.unit
def test_basic_official_data_benchmark_writes_provenance_artifacts(tmp_path, monkeypatch):
    X, y = _classification_data()
    monkeypatch.setattr(
        pusb_official_data,
        "load_ijcnn1",
        lambda path, expected_sha256=None: (X, y, "verified-test-hash"),
    )

    trials = run_benchmark(_config(), data_root=tmp_path, output_dir=tmp_path / "results")

    assert len(trials) == 1
    for name in ("trials.csv", "summary.csv", "run_manifest.json", "resolved_config.json"):
        assert (tmp_path / "results" / name).is_file()
    manifest = json.loads((tmp_path / "results" / "run_manifest.json").read_text())
    assert manifest["paper_claim"] is False
    assert manifest["dataset"]["sha256"] == "verified-test-hash"
    assert manifest["n_trials"] == 1


@pytest.mark.unit
def test_determ_resume_skips_completed_trial_without_duplication(tmp_path, monkeypatch):
    X, y = _classification_data()
    monkeypatch.setattr(
        pusb_official_data,
        "load_ijcnn1",
        lambda path, expected_sha256=None: (X, y, "verified-test-hash"),
    )
    output = tmp_path / "results"
    first = run_benchmark(_config(), data_root=tmp_path, output_dir=output)
    second = run_benchmark(_config(), data_root=tmp_path, output_dir=output, resume=True)

    assert len(first) == len(second) == 1
    assert len(pusb_official_data.pd.read_csv(output / "trials.csv")) == 1


@pytest.mark.unit
def test_edge_resume_rejects_changed_config(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    (output / "resolved_config.json").write_text(json.dumps(_config()), encoding="utf-8")
    changed = _config()
    changed["experiment"]["seeds"] = [99]

    with pytest.raises(ValueError, match="resume config differs"):
        run_benchmark(changed, data_root=tmp_path, output_dir=output, resume=True)
