# ruff: noqa: N803, N806

"""Tests for clean-validation model selection in the deep-PU runner."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.deep_pu import official_data
from benchmarks.deep_pu.official_data import (
    _grid_candidates,
    load_official_data_config,
    make_pu_split,
    run_official_data_benchmark,
)


def _loader(config, root, *, download):
    del config, root, download
    train_labels = np.tile([0, 1], 20)
    test_labels = np.tile([0, 1], 10)
    X_train = np.c_[train_labels, np.arange(40) / 40].astype(np.float32)
    X_test = np.c_[test_labels, np.arange(20) / 20].astype(np.float32)
    return X_train, train_labels, X_test, test_labels


def _config():
    return {
        "schema_version": 1,
        "protocol": "official_data",
        "fidelity_level": "smoke",
        "paper_claim": False,
        "seeds": [5],
        "dataset": {
            "name": "cifar10",
            "positive_classes": [1],
            "n_labeled_positive": 6,
            "n_unlabeled": 20,
            "n_test": 12,
            "clean_validation_fraction": 0.2,
            "class_prior": 0.5,
            "representation": "flattened_pixels",
        },
        "methods": {
            "weighted_contrastive_pu": {
                "variant": "selection-test",
                "parameters": {},
                "model_selection": {
                    "strategy": "clean_validation_grid",
                    "metric": "accuracy",
                    "parameter_grid": {
                        "contrastive_weight": [0.1, 0.9],
                        "distribution_weight": [0.1, 0.9],
                    },
                    "refit": True,
                },
            }
        },
        "runtime": {"device": "cpu"},
        "claim_policy": "test only",
        "known_gaps": ["test"],
    }


class _FakeWConPU:
    def __init__(self, parameters, class_prior, fit_sizes):
        self.parameters = parameters
        self.class_prior_ = class_prior
        self.fit_sizes = fit_sizes

    def fit(self, X, y_pu):
        self.fit_sizes.append((len(X), len(y_pu)))
        return self

    def decision_function(self, X):
        preferred = (
            self.parameters.get("contrastive_weight") == 0.9
            and self.parameters.get("distribution_weight") == 0.1
        )
        orientation = 1.0 if preferred else -1.0
        return orientation * (np.asarray(X)[:, 0] - 0.5)


@pytest.mark.unit
def test_basic_clean_validation_split_is_disjoint_and_supervised():
    dataset = make_pu_split(
        *_loader({}, ".", download=False),
        positive_classes=[1],
        n_labeled_positive=6,
        n_unlabeled=20,
        n_test=12,
        seed=5,
        clean_validation_fraction=0.2,
    )
    assert dataset.X_validation.shape == (8, 2)
    assert dataset.y_validation_true.shape == (8,)
    assert dataset.y_validation_pu is None
    assert dataset.manifest["validation_kind"] == "clean"
    assert dataset.manifest["train_validation_overlap"] == 0


@pytest.mark.unit
def test_determ_grid_selects_refits_persists_and_resumes(tmp_path, monkeypatch):
    config = _config()
    fit_sizes = []

    def fake_builder(method, method_config, *, class_prior, seed):
        del method, seed
        return _FakeWConPU(method_config["parameters"], class_prior, fit_sizes)

    monkeypatch.setattr(official_data, "_build_estimator", fake_builder)
    output = tmp_path / "output"
    trials, _ = run_official_data_benchmark(
        config,
        output,
        data_root=tmp_path / "data",
        loader=_loader,
    )
    selection = json.loads(trials.loc[0, "selected_parameters"])
    assert selection == {"contrastive_weight": 0.9, "distribution_weight": 0.1}
    assert trials.loc[0, "selection_score"] == pytest.approx(1.0)
    assert trials.loc[0, "selection_candidates"] == 4
    assert len(fit_sizes) == 5
    assert set(fit_sizes) == {(26, 26)}
    assert len(pd.read_csv(output / "model_selection.csv")) == 4

    run_official_data_benchmark(
        config,
        output,
        data_root=tmp_path / "data",
        resume=True,
        loader=_loader,
    )
    assert len(fit_sizes) == 5


@pytest.mark.unit
@pytest.mark.parametrize("metric", ["accuracy", "balanced_accuracy", "roc_auc"])
def test_param_supported_selection_metrics_load(tmp_path, metric):
    config = _config()
    config["methods"]["weighted_contrastive_pu"]["model_selection"]["metric"] = metric
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert (
        load_official_data_config(path)["methods"]["weighted_contrastive_pu"]["model_selection"][
            "metric"
        ]
        == metric
    )


@pytest.mark.unit
def test_edge_selection_requires_clean_validation_and_valid_grid(tmp_path):
    config = _config()
    config["dataset"].pop("clean_validation_fraction")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="clean_validation_fraction"):
        load_official_data_config(path)

    config["dataset"]["clean_validation_fraction"] = 0.2
    config["methods"]["weighted_contrastive_pu"]["model_selection"]["parameter_grid"] = {
        "contrastive_weight": []
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty value list"):
        load_official_data_config(path)

    assert _grid_candidates({"a": [1, 2], "b": [3, 4]}) == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]
