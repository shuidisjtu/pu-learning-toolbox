"""Shipped traditional-PU config tests: loadability and contract §2 grid cells."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.traditional_pu.runner import (
    PNU_RATIOS,
    _iter_scenario_specs,
    load_config,
    run_trials,
)

pytestmark = pytest.mark.unit

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "traditional_pu" / "configs"
SEVEN_METHODS = CONFIGS_DIR / "seven_methods_pu_baseline_v1.json"
PNU_BASELINE = CONFIGS_DIR / "pnu_baseline_v1.json"

# README grid: SCAR 6 methods × 3 priors × 2 scales; SAR only the 4
# PU-family methods (LDCE/KLDCE gated out, contract §2.3); PNU runs its own
# tri-label config (contract §2.2) and never appears in the seven-methods
# grid.  PNU: 3 ratios × 2 scales.
SEVEN_METHODS_CELLS = 6 * 3 * 2 * 2 - 2 * 3 * 2  # 72 - 12 SAR LDCE/KLDCE = 60
PNU_CELLS = 3 * 2  # 6

SEVEN_METHODS_NAMES = {"elkan_noto", "upu", "nnpu", "ldce", "kldce", "llsvm"}


def _mini_config(methods: dict) -> dict:
    """Minimal runner config (not via load_config; direct dict for speed)."""
    return {
        "schema_version": 1,
        "protocol": "traditional_pu_baseline",
        "seed_set_development": [0],
        "seed_set_confirmation": [10],
        "data": {
            "n_samples_small": 50,
            "n_samples_mid": 50,
            "n_features": 5,
            "class_priors": [0.5],
            "separation": 2.0,
            "label_frequency": 0.5,
            "sar_strength": 1.0,
            "sar_mechanism": "linear",
            "pn_ratios": ["1:1:4"],
        },
        "timeouts": {"upu": 60, "llsvm": 60},
        "methods": methods,
        "limitations": [],
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class TestShippedConfigs:
    def test_basic_both_shipped_configs_load(self):
        for path in (SEVEN_METHODS, PNU_BASELINE):
            cfg = load_config(path)
            assert cfg["protocol"] == "traditional_pu_baseline"
            assert cfg["schema_version"] == 1
            assert cfg["methods"]

    def test_basic_seven_methods_grid_has_no_pnu_cells(self):
        cfg = load_config(SEVEN_METHODS)
        cells = list(_iter_scenario_specs(cfg))
        assert set(cfg["methods"]) == SEVEN_METHODS_NAMES
        assert all(name != "pnu" for name, _, _ in cells)
        assert all(not scenario.startswith("pnu-") for _, scenario, _ in cells)
        assert "pnu" not in cfg["timeouts"]

    def test_basic_pnu_config_grid_only_pnu_cells(self):
        cfg = load_config(PNU_BASELINE)
        cells = list(_iter_scenario_specs(cfg))
        assert cells
        assert all(name == "pnu" for name, _, _ in cells)
        assert all(spec["kind"] == "pnu" for _, _, spec in cells)
        assert all(scenario.startswith("pnu-") for _, scenario, _ in cells)

    def test_basic_grid_cell_counts_match_readme(self):
        # seven-methods: 6 methods × 3 priors × 2 scales × 2 kinds (SCAR+SAR)
        # minus the 12 SAR LDCE/KLDCE cells = 60; no PNU cells at all.
        seven = list(_iter_scenario_specs(load_config(SEVEN_METHODS)))
        assert len(seven) == SEVEN_METHODS_CELLS == 60
        scar = [spec for _, _, spec in seven if spec["kind"] == "scar"]
        sar = [spec for _, _, spec in seven if spec["kind"] == "sar"]
        assert len(scar) == 6 * 3 * 2
        assert len(sar) == 4 * 3 * 2
        assert all(spec["kind"] != "pnu" for _, _, spec in seven)
        # pnu config: 1 method × 3 ratios × 2 scales = 6
        pnu = list(_iter_scenario_specs(load_config(PNU_BASELINE)))
        assert len(pnu) == PNU_CELLS == 6
        assert {spec["ratio"] for _, _, spec in pnu} == set(PNU_RATIOS)

    def test_basic_each_method_has_timeout_guard(self):
        for path in (SEVEN_METHODS, PNU_BASELINE):
            cfg = load_config(path)
            assert set(cfg["methods"]) <= set(cfg["timeouts"])

    def test_determ_repeated_loads_identical(self):
        a = load_config(SEVEN_METHODS)
        b = load_config(SEVEN_METHODS)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


class TestLoadConfigValidation:
    def test_param_seed_sets_must_not_overlap(self, tmp_path):
        cfg = _mini_config({"upu": {}})
        cfg["seed_set_development"] = [0, 1]
        cfg["seed_set_confirmation"] = [1, 2]
        with pytest.raises(ValueError, match="must not overlap"):
            load_config(_write_config(tmp_path, cfg))

    def test_param_unknown_pn_ratio_rejected(self, tmp_path):
        cfg = _mini_config({"pnu": {}})
        cfg["data"]["pn_ratios"] = ["9:9:9"]
        with pytest.raises(ValueError, match="unknown ratios"):
            load_config(_write_config(tmp_path, cfg))

    def test_edge_pnu_without_pn_ratios_rejected(self, tmp_path):
        cfg = _mini_config({"pnu": {}})
        del cfg["data"]["pn_ratios"]
        with pytest.raises(ValueError, match="pn_ratios"):
            load_config(_write_config(tmp_path, cfg))

    def test_edge_empty_methods_rejected(self, tmp_path):
        cfg = _mini_config({})
        with pytest.raises(ValueError, match="methods"):
            load_config(_write_config(tmp_path, cfg))


class TestResumeFingerprint:
    def test_param_resume_rejects_changed_config(self, tmp_path):
        out = tmp_path / "resume"
        cfg_a = _mini_config({"upu": {}})
        run_trials(cfg_a, results_dir=out, resume=False, progress=False)
        assert (out / ".config_sha256").exists()
        cfg_b = _mini_config({"llsvm": {}})
        with pytest.raises(ValueError, match="resume config mismatch"):
            run_trials(cfg_b, results_dir=out, resume=True, progress=False)
