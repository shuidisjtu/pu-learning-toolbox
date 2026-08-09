# ruff: noqa: N806

"""Tests for the unified deep-PU benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks.deep_pu.runner import (
    GaussianConditionalGenerator,
    load_config,
    make_case_control_data,
    run_benchmark,
    run_trials,
    summarize_trials,
)

pytestmark = [pytest.mark.unit, pytest.mark.paper]


@pytest.fixture
def deep_config():
    return {
        "schema_version": 1,
        "protocol": "clean_room",
        "seeds": [2],
        "data": {
            "n_positive": 12,
            "n_unlabeled": 28,
            "n_test": 40,
            "n_features": 4,
            "class_prior": 0.4,
            "separation": 1.5,
        },
        "methods": {
            "infomax_pu": {
                "variant": "test_purl",
                "parameters": {
                    "representation_dim": 3,
                    "hidden_dim": 6,
                    "representation_epochs": 2,
                    "classifier_epochs": 2,
                },
            },
            "weighted_contrastive_pu": {
                "variant": "test_wconpu",
                "parameters": {
                    "hidden_dim": 6,
                    "embedding_dim": 3,
                    "queue_size": 16,
                    "batch_size": 16,
                    "max_epochs": 2,
                },
            },
            "dgpu": {
                "variant": "test_gaussian",
                "generator": {"noise_scale": 0.1},
                "parameters": {
                    "hidden_dim": 6,
                    "rounds": 1,
                    "initialization_epochs": 2,
                    "annotation_epochs": 1,
                    "generated_samples": 10,
                    "batch_size": 16,
                },
            },
        },
        "limitations": ["test configuration"],
    }


class TestDeepPUBenchmarkRunner:
    def test_basic_load_config(self, tmp_path, deep_config):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(deep_config), encoding="utf-8")
        assert load_config(path) == deep_config

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("schema_version", 2, "schema_version"),
            ("protocol", "paper_like", "clean_room"),
            ("seeds", [], "seeds"),
        ],
    )
    def test_param_invalid_top_level_config(self, tmp_path, deep_config, field, value, message):
        deep_config[field] = value
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(deep_config), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_config(path)

    def test_param_unknown_method_is_rejected(self, tmp_path, deep_config):
        deep_config["methods"]["unknown"] = {"variant": "bad", "parameters": {}}
        path = tmp_path / "unknown.json"
        path.write_text(json.dumps(deep_config), encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported"):
            load_config(path)

    def test_deterministic_case_control_data(self, deep_config):
        first = make_case_control_data(7, deep_config["data"])
        second = make_case_control_data(7, deep_config["data"])
        for left, right in zip(first, second, strict=True):
            np.testing.assert_allclose(left, right)
        assert first[0].shape == (40, 4)
        assert first[1].sum() == 12
        assert set(np.unique(first[3])) == {0, 1}

    def test_basic_gaussian_generator_protocol(self):
        X = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]])
        y = np.array([1, 1, 0, 0])
        generator = GaussianConditionalGenerator(noise_scale=0.01).fit(X, y)

        first = generator.sample(3, class_label=1, random_state=5)
        second = generator.sample(3, class_label=1, random_state=5)

        assert generator.fit_calls_ == 1
        assert first.shape == (3, 2)
        np.testing.assert_allclose(first, second)

    def test_edge_generator_requires_fit_and_valid_parameters(self):
        with pytest.raises(ValueError, match="positive"):
            GaussianConditionalGenerator(noise_scale=0.0)
        generator = GaussianConditionalGenerator()
        with pytest.raises(ValueError, match="must be fitted"):
            generator.sample(1, class_label=1)

    def test_basic_all_three_methods_emit_finite_trials(self, deep_config):
        trials = run_trials(deep_config)

        assert len(trials) == 3
        assert set(trials["method"]) == {
            "infomax_pu",
            "weighted_contrastive_pu",
            "dgpu",
        }
        assert not trials["paper_claim"].any()
        assert np.isfinite(trials["roc_auc"]).all()
        assert trials["bayes_posterior_spearman"].between(-1, 1).all()
        assert trials.loc[trials["method"] == "dgpu", "generator_fit_calls"].iloc[0] == 1

    def test_deterministic_selected_method_metrics(self, deep_config):
        first = run_trials(deep_config, selected_methods=["infomax_pu"])
        second = run_trials(deep_config, selected_methods=["infomax_pu"])
        columns = sorted(set(first.columns) - {"elapsed_seconds"})
        assert first[columns].equals(second[columns])

    def test_basic_summary_counts_each_seed(self, deep_config):
        summary = summarize_trials(run_trials(deep_config))
        assert set(summary["method"]) == set(deep_config["methods"])
        assert set(summary["n"]) == {1}

    def test_basic_output_artifacts_and_claim_boundary(self, tmp_path, deep_config):
        trials, summary = run_benchmark(
            deep_config,
            tmp_path,
            selected_methods=["infomax_pu"],
        )
        assert len(trials) == 1
        assert not summary.empty
        for name in ("trials.csv", "summary.csv", "resolved_config.json", "run_manifest.json"):
            assert (tmp_path / name).is_file()
        manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["paper_claim"] is False
        assert manifest["n_trials"] == 1
        assert len(manifest["runner_sha256"]) == 64

    def test_edge_official_configs_are_locked_and_not_executed(self):
        root = Path(__file__).resolve().parents[2] / "benchmarks" / "deep_pu" / "configs"
        source_lock = json.loads((root / "official_sources.lock.json").read_text(encoding="utf-8"))
        official = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "official").glob("*.json"))
        ]

        assert set(source_lock["sources"]) == {
            "infomax_pu",
            "weighted_contrastive_pu",
            "dgpu",
        }
        assert len(official) == 3
        assert all(item["protocol"] == "paper_like" for item in official)
        assert all(item["status"] == "locked_not_executed" for item in official)
        assert all("claim_policy" in item for item in official)
