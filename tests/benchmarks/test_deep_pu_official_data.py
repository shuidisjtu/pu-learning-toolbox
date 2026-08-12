# ruff: noqa: N803, N806

"""Tests for official-data loading, PU splitting, preflight, and resume."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks.deep_pu.official_data import (
    _dataset_artifacts,
    _git_worktree_dirty,
    build_dataset,
    environment_preflight,
    load_official_data_config,
    make_pu_split,
    run_official_data_benchmark,
)
from benchmarks.deep_pu.preflight_paper import audit_locked_configs
from benchmarks.deep_pu.runner import _build_estimator
from pu_toolbox.prior import KernelMeanPriorEstimator

pytestmark = [pytest.mark.unit, pytest.mark.paper]


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


class TestOfficialDataBenchmark:
    def test_basic_load_config(self, tmp_path, official_config):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(official_config), encoding="utf-8")
        assert load_official_data_config(path) == official_config

    def test_edge_load_rejects_invalid_prior_and_validation_modes(self, tmp_path, official_config):
        method = official_config["methods"]["infomax_pu"]
        method["class_prior_mode"] = "guess"
        path = tmp_path / "invalid-mode.json"
        path.write_text(json.dumps(official_config), encoding="utf-8")
        with pytest.raises(ValueError, match="class_prior_mode"):
            load_official_data_config(path)

        method["class_prior_mode"] = "estimate"
        method["parameters"]["prior_estimator"] = {"name": "unknown"}
        path.write_text(json.dumps(official_config), encoding="utf-8")
        with pytest.raises(ValueError, match="kernel_mean"):
            load_official_data_config(path)

        method["parameters"].pop("prior_estimator")
        method["fit"] = {"use_validation_for_early_stopping": True}
        path.write_text(json.dumps(official_config), encoding="utf-8")
        with pytest.raises(ValueError, match="validation forwarding"):
            load_official_data_config(path)

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
            validation_positive=3,
            validation_unlabeled=4,
            unlabeled_class_prior=0.25,
        )
        second = make_pu_split(
            *arrays,
            positive_classes=[1],
            n_labeled_positive=6,
            n_unlabeled=12,
            n_test=12,
            seed=7,
            validation_positive=3,
            validation_unlabeled=4,
            unlabeled_class_prior=0.25,
        )
        np.testing.assert_allclose(first.X_train, second.X_train)
        assert first.manifest == second.manifest
        assert first.manifest["labeled_unlabeled_overlap"] == 0
        assert first.manifest["train_validation_overlap"] == 0
        assert first.manifest["hidden_unlabeled_positive_rate"] == pytest.approx(0.25)
        assert first.manifest["target_unlabeled_class_prior"] == pytest.approx(0.25)
        assert first.X_validation.shape == (7, 4)
        assert first.y_validation_pu.sum() == 3
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

    def test_basic_run_writes_claim_safe_artifacts(self, tmp_path, official_config, monkeypatch):
        official_config["dataset"].update({"validation_positive": 3, "validation_unlabeled": 4})
        official_config["methods"]["infomax_pu"]["fit"] = {
            "use_validation_for_early_stopping": True
        }
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
        assert trials.loc[0, "class_prior_mode"] == "known"
        assert trials.loc[0, "class_prior_absolute_error"] == pytest.approx(0.0)

        project_root = Path(__file__).resolve().parents[2]
        tracked_output = project_root / "benchmarks" / "deep_pu" / "results" / "test-output"
        commands = []

        def fake_git_value(root, command):
            assert root == project_root
            commands.append(command)
            return ""

        monkeypatch.setattr("benchmarks.deep_pu.official_data._git_value", fake_git_value)
        assert _git_worktree_dirty(project_root, tracked_output) is False
        assert ":(exclude)benchmarks/deep_pu/results/test-output/**" in commands[0]

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

    def test_basic_infomax_protocol_config_locks_paper_network(self):
        root = Path(__file__).resolve().parents[2] / "benchmarks" / "deep_pu" / "configs"
        config = load_official_data_config(root / "official_data_infomax_fashion_protocol.json")
        parameters = config["methods"]["infomax_pu"]["parameters"]
        assert len(config["seeds"]) == 20
        assert config["dataset"]["validation_positive"] == 50
        assert config["dataset"]["validation_unlabeled"] == 200
        assert config["methods"]["infomax_pu"]["class_prior_mode"] == "estimate"
        assert parameters["classifier_hidden_dims"] == [300, 300, 300]
        assert parameters["representation_gradient_noise"] == pytest.approx(0.01)
        assert parameters["prior_estimator"]["name"] == "kernel_mean"
        assert parameters["prior_estimator"]["parameters"]["variant"] == "km1"
        report = environment_preflight(config, data_root=".", download=False)
        assert report["ready"]
        assert not any("backbone" in item for item in report["warnings"])
        prior_configs = {
            0.3: "official_data_infomax_fashion_protocol_pi03.json",
            0.5: "official_data_infomax_fashion_protocol.json",
            0.7: "official_data_infomax_fashion_protocol_pi07.json",
        }
        for prior, filename in prior_configs.items():
            prior_config = load_official_data_config(root / filename)
            assert prior_config["dataset"]["class_prior"] == pytest.approx(prior)
            assert prior_config["dataset"]["unlabeled_class_prior"] == pytest.approx(prior)

    def test_basic_wconpu_protocol_config_locks_visual_pipeline(self):
        root = Path(__file__).resolve().parents[2] / "benchmarks" / "deep_pu" / "configs"
        config = load_official_data_config(root / "official_data_wconpu_cifar10_protocol.json")
        method = config["methods"]["weighted_contrastive_pu"]
        vision = method["parameters"]["vision"]
        assert config["dataset"]["positive_classes"] == [0, 1, 8, 9]
        assert config["dataset"]["representation"] == "image_tensor"
        assert config["dataset"]["clean_validation_fraction"] == pytest.approx(0.1)
        assert config["dataset"]["n_unlabeled"] == 44000
        assert vision["backbone"]["name"] == "cnn13"
        assert vision["weak_augmentation"]["name"] == "simaugment"
        assert vision["strong_augmentation"]["name"] == "randaugment"
        assert method["parameters"]["scheduler"] == "cosine_annealing"
        assert method["model_selection"]["metric"] == "accuracy"
        assert len(method["model_selection"]["parameter_grid"]["contrastive_weight"]) == 4
        assert len(method["model_selection"]["parameter_grid"]["distribution_weight"]) == 4
        report = environment_preflight(config, data_root=".", download=False)
        if not report["cuda_available"]:
            assert any("CUDA" in item for item in report["blockers"])

    def test_basic_image_split_preserves_nchw_shape(self):
        rng = np.random.default_rng(2)
        arrays = (
            rng.random((30, 3, 8, 8), dtype=np.float32),
            np.tile([0, 1], 15),
            rng.random((20, 3, 8, 8), dtype=np.float32),
            np.tile([0, 1], 10),
        )
        dataset = make_pu_split(
            *arrays,
            positive_classes=[1],
            n_labeled_positive=5,
            n_unlabeled=10,
            n_test=10,
            seed=4,
        )
        assert dataset.X_train.shape == (15, 3, 8, 8)
        assert dataset.X_test.shape == (10, 3, 8, 8)

    @pytest.mark.parametrize(
        ("dataset_name", "filename"),
        [("svhn", "train_32x32.mat"), ("stl10", "stl10_binary.tar.gz")],
    )
    def test_param_visual_archives_are_hashed(self, tmp_path, dataset_name, filename):
        (tmp_path / filename).write_bytes(b"paper-protocol-fixture")
        artifacts = _dataset_artifacts(tmp_path, dataset_name)
        assert artifacts[0]["path"] == filename
        assert len(artifacts[0]["sha256"]) == 64

    def test_basic_runner_builds_structured_kernel_mean_prior(self):
        estimator = _build_estimator(
            "infomax_pu",
            {
                "parameters": {
                    "prior_estimator": {
                        "name": "kernel_mean",
                        "parameters": {"variant": "km2", "max_qp_iter": 10},
                    }
                }
            },
            class_prior=None,
            seed=7,
        )
        assert estimator.class_prior is None
        assert isinstance(estimator.prior_estimator, KernelMeanPriorEstimator)
        assert estimator.prior_estimator.variant == "km2"
