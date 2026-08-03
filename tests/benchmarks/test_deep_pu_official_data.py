# ruff: noqa: N803, N806

"""Tests for official-data loading, PU splitting, preflight, and resume."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks.deep_pu.official_data import (
    build_dataset,
    environment_preflight,
    load_official_data_config,
    make_pu_split,
    run_official_data_benchmark,
)
from benchmarks.deep_pu.preflight_paper import audit_locked_configs


@pytest.fixture
def official_config():
    return {
        "schema_version": 1,
        "protocol": "official_data",
        "fidelity_level": "smoke",
        "paper_claim": False,
        "seeds": [3],
        "dataset": {
            "name": "fashion_mnist",
            "positive_classes": [1],
            "n_labeled_positive": 6,
            "n_unlabeled": 12,
            "n_test": 12,
            "class_prior": 0.5,
            "representation": "flattened_pixels",
        },
        "methods": {
            "infomax_pu": {
                "variant": "test",
                "parameters": {
                    "representation_dim": 2,
                    "hidden_dim": 4,
                    "representation_epochs": 1,
                    "classifier_epochs": 1,
                },
            }
        },
        "runtime": {"device": "cpu"},
        "claim_policy": "test only",
        "known_gaps": ["test"],
    }


def fake_loader(config, root, *, download):
    del config, root, download
    rng = np.random.default_rng(10)
    train_labels = np.tile([0, 1], 20)
    test_labels = np.tile([0, 1], 10)
    return (
        rng.normal(size=(40, 4)).astype(np.float32),
        train_labels,
        rng.normal(size=(20, 4)).astype(np.float32),
        test_labels,
    )


@pytest.mark.unit
class TestOfficialDataBenchmark:
    def test_basic_load_config(self, tmp_path, official_config):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(official_config), encoding="utf-8")
        assert load_official_data_config(path) == official_config

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("protocol", "paper_like", "protocol"),
            ("paper_claim", True, "paper_claim"),
            ("fidelity_level", "paper_result", "fidelity_level"),
            ("seeds", [], "seeds"),
        ],
    )
    def test_param_rejects_invalid_claim_and_protocol(
        self, tmp_path, official_config, field, value, message
    ):
        official_config[field] = value
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(official_config), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_official_data_config(path)

    def test_determ_split_is_reproducible_and_disjoint(self):
        arrays = fake_loader({}, ".", download=False)
        first = make_pu_split(
            *arrays,
            positive_classes=[1],
            n_labeled_positive=6,
            n_unlabeled=12,
            n_test=12,
            seed=7,
        )
        second = make_pu_split(
            *arrays,
            positive_classes=[1],
            n_labeled_positive=6,
            n_unlabeled=12,
            n_test=12,
            seed=7,
        )
        np.testing.assert_allclose(first.X_train, second.X_train)
        assert first.manifest == second.manifest
        assert first.manifest["labeled_unlabeled_overlap"] == 0
        assert len(first.manifest["split_indices_sha256"]) == 64

    def test_edge_split_rejects_impossible_sizes(self):
        arrays = fake_loader({}, ".", download=False)
        with pytest.raises(ValueError, match="not enough"):
            make_pu_split(
                *arrays,
                positive_classes=[1],
                n_labeled_positive=30,
                n_unlabeled=5,
                n_test=10,
                seed=0,
            )

    def test_basic_build_dataset_uses_loader(self, tmp_path, official_config):
        dataset = build_dataset(
            official_config["dataset"],
            tmp_path,
            seed=3,
            download=False,
            loader=fake_loader,
        )
        assert dataset.X_train.shape == (18, 4)
        assert dataset.y_pu.sum() == 6
        assert set(dataset.y_test) == {0, 1}

    def test_edge_preflight_reports_backend_and_cuda_blockers(self, official_config):
        official_config["methods"] = {"dgpu": {"variant": "test", "parameters": {}}}
        official_config["runtime"]["device"] = "cuda"
        report = environment_preflight(official_config, data_root=".", download=False)
        assert not report["ready"]
        assert any("EDM" in item for item in report["blockers"])
        if not report["cuda_available"]:
            assert any("CUDA" in item for item in report["blockers"])

    def test_basic_run_writes_claim_safe_artifacts(self, tmp_path, official_config):
        trials, summary = run_official_data_benchmark(
            official_config,
            tmp_path / "output",
            data_root=tmp_path / "data",
            loader=fake_loader,
        )
        assert len(trials) == 1
        assert not summary.empty
        manifest = json.loads(
            (tmp_path / "output" / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["paper_claim"] is False
        assert manifest["dataset_splits"]["3"]["labeled_unlabeled_overlap"] == 0
        assert (tmp_path / "output" / "preflight.json").is_file()

    def test_determ_resume_does_not_duplicate_trials(self, tmp_path, official_config):
        output = tmp_path / "output"
        run_official_data_benchmark(
            official_config, output, data_root=tmp_path / "data", loader=fake_loader
        )
        trials, _ = run_official_data_benchmark(
            official_config,
            output,
            data_root=tmp_path / "data",
            resume=True,
            loader=fake_loader,
        )
        assert len(trials) == 1

        official_config["dataset"]["n_test"] = 14
        with pytest.raises(ValueError, match="differs"):
            run_official_data_benchmark(
                official_config,
                output,
                data_root=tmp_path / "data",
                resume=True,
                loader=fake_loader,
            )

    def test_basic_locked_config_audit_separates_blockers(self, tmp_path):
        config = {
            "method": "dgpu",
            "status": "locked_not_executed",
            "execution_requirements": {
                "cuda_required": False,
                "external_backends": ["conditional_edm"],
                "restricted_datasets": ["CelebA"],
                "implementation_gaps": ["visual adapter missing"],
            },
        }
        (tmp_path / "dgpu.json").write_text(json.dumps(config), encoding="utf-8")
        report = audit_locked_configs(tmp_path)
        blockers = report["methods"][0]["blockers"]
        assert not report["all_ready"]
        assert len(blockers) == 3
        assert any("EDM" in item for item in blockers)
