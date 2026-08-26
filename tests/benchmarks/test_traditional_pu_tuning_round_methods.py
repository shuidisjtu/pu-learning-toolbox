"""Per-method config-generation tests for the tuning-round registry.

One smoke test per registered method (nnPU/upu/LLSVM/Elkan-Noto) plus the
non-tunable-parameter rejections.  Core kldce generation logic lives in
test_traditional_pu_tuning_round.py; shared builders in tuning_helpers.py.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.traditional_pu.tuning_round import generate_round_configs
from tests.benchmarks.tuning_helpers import (
    ELKAN_NOTO_BASE_PARAMS,
    LLSVM_BASE_PARAMS,
    NNPU_BASE_PARAMS,
    UPU_BASE_PARAMS,
    base_config_path,
)

pytestmark = pytest.mark.unit


class TestMethodRegistration:
    def test_basic_nnpu_generate_keeps_non_tunable_nulls(self, tmp_path):
        base = base_config_path(
            tmp_path,
            methods={"nnpu": dict(NNPU_BASE_PARAMS)},
            timeouts={"nnpu": 120},
        )
        paths = generate_round_configs(
            base, tmp_path / "round", [("c1", {"beta": 0.5, "batch_size": 64})], method="nnpu"
        )
        produced = json.loads(paths[0].read_text(encoding="utf-8"))
        assert set(produced["methods"]) == {"nnpu"}
        params = produced["methods"]["nnpu"]
        assert params["beta"] == 0.5 and params["batch_size"] == 64
        assert params["gamma"] == 1.0  # untouched default kept
        # JSON-inexpressible params stay at their base nulls, not overridden.
        assert params["model"] is None and params["optimizer"] is None
        assert params["device"] is None
        assert "locks_source_defaults" not in produced

    def test_edge_nnpu_rejects_non_tunable_overrides(self, tmp_path):
        base = base_config_path(tmp_path, methods={"nnpu": dict(NNPU_BASE_PARAMS)})
        with pytest.raises(ValueError, match="not tunable"):
            generate_round_configs(
                base, tmp_path / "round", [("c1", {"optimizer": "adam"})], method="nnpu"
            )
        with pytest.raises(ValueError, match="not tunable"):
            generate_round_configs(
                base, tmp_path / "round", [("c1", {"device": "cuda"})], method="nnpu"
            )

    def test_edge_nnpu_missing_non_tunable_keys_pass_coverage_check(self, tmp_path):
        # model/optimizer/device are excluded from the constructor-defaults
        # coverage check — a base missing them still generates (their
        # constructor defaults of None are the correct values).
        methods = {
            "nnpu": {
                k: v
                for k, v in NNPU_BASE_PARAMS.items()
                if k not in ("model", "optimizer", "device")
            }
        }
        base = base_config_path(tmp_path, methods=methods)
        paths = generate_round_configs(base, tmp_path / "round", [("c1", {})], method="nnpu")
        produced = json.loads(paths[0].read_text(encoding="utf-8"))
        assert "model" not in produced["methods"]["nnpu"]

    def test_basic_upu_generate_with_rbf_overrides(self, tmp_path):
        base = base_config_path(
            tmp_path,
            methods={"upu": dict(UPU_BASE_PARAMS)},
            timeouts={"upu": 120},
        )
        paths = generate_round_configs(
            base,
            tmp_path / "round",
            [("c1", {"basis": "rbf", "kernel_width": 0.5})],
            method="upu",
        )
        produced = json.loads(paths[0].read_text(encoding="utf-8"))
        assert set(produced["methods"]) == {"upu"}
        params = produced["methods"]["upu"]
        assert params["basis"] == "rbf" and params["kernel_width"] == 0.5
        assert params["loss"] == "double_hinge"  # untouched default kept
        assert "locks_source_defaults" not in produced

    def test_basic_llsvm_generate_with_sgd_overrides(self, tmp_path):
        base = base_config_path(
            tmp_path,
            methods={"llsvm": dict(LLSVM_BASE_PARAMS)},
            timeouts={"llsvm": 120},
        )
        paths = generate_round_configs(
            base,
            tmp_path / "round",
            [("c1", {"learning_rate": 1e-5, "reg_lambda": 0.1})],
            method="llsvm",
        )
        produced = json.loads(paths[0].read_text(encoding="utf-8"))
        assert set(produced["methods"]) == {"llsvm"}
        params = produced["methods"]["llsvm"]
        assert params["learning_rate"] == 1e-5 and params["reg_lambda"] == 0.1
        assert params["gamma"] == 10.0  # untouched default kept
        assert "locks_source_defaults" not in produced

    def test_basic_elkan_noto_generate_with_calibration_overrides(self, tmp_path):
        base = base_config_path(
            tmp_path,
            methods={"elkan_noto": dict(ELKAN_NOTO_BASE_PARAMS)},
            timeouts={"elkan_noto": 120},
        )
        paths = generate_round_configs(
            base,
            tmp_path / "round",
            [("c1", {"calibration_method": "isotonic", "n_cv_folds": 5})],
            method="elkan_noto",
        )
        produced = json.loads(paths[0].read_text(encoding="utf-8"))
        assert set(produced["methods"]) == {"elkan_noto"}
        params = produced["methods"]["elkan_noto"]
        assert params["calibration_method"] == "isotonic" and params["n_cv_folds"] == 5
        assert params["mode"] == "probability_correction"  # untouched default kept
        assert params["base_estimator"] is None  # JSON-inexpressible, kept at null
        assert "locks_source_defaults" not in produced

    def test_edge_elkan_noto_rejects_base_estimator_override(self, tmp_path):
        base = base_config_path(tmp_path, methods={"elkan_noto": dict(ELKAN_NOTO_BASE_PARAMS)})
        with pytest.raises(ValueError, match="not tunable"):
            generate_round_configs(
                base,
                tmp_path / "round",
                [("c1", {"base_estimator": "logreg"})],
                method="elkan_noto",
            )

    def test_param_elkan_noto_merges_overrides_and_keeps_defaults(self, tmp_path):
        base = base_config_path(tmp_path, methods={"elkan_noto": dict(ELKAN_NOTO_BASE_PARAMS)})
        paths = generate_round_configs(
            base,
            tmp_path / "round",
            [("c1", {"n_cv_folds": 10, "eps": 1e-6})],
            method="elkan_noto",
        )
        produced = json.loads(paths[0].read_text(encoding="utf-8"))
        params = produced["methods"]["elkan_noto"]
        assert params["n_cv_folds"] == 10 and params["eps"] == 1e-6
        assert params["calibration_method"] == "sigmoid"  # untouched default kept
        assert params["mode"] == "probability_correction"

    def test_determ_elkan_noto_same_input_same_output(self, tmp_path):
        base = base_config_path(tmp_path, methods={"elkan_noto": dict(ELKAN_NOTO_BASE_PARAMS)})
        first = generate_round_configs(
            base, tmp_path / "a", [("c1", {"eps": 1e-9})], method="elkan_noto"
        )
        second = generate_round_configs(
            base, tmp_path / "b", [("c1", {"eps": 1e-9})], method="elkan_noto"
        )
        assert json.loads(first[0].read_text(encoding="utf-8")) == json.loads(
            second[0].read_text(encoding="utf-8")
        )
