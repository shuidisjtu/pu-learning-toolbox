"""Runner orchestration tests: state machine, resume, artifacts, failure isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.traditional_pu.runner import (
    METRIC_COLUMNS,
    load_config,
    run_trials,
    write_artifacts,
)

pytestmark = pytest.mark.unit

MINI_CONFIG = {
    "schema_version": 1,
    "protocol": "traditional_pu_baseline",
    "description": "mini test config",
    "seed_set_development": [0, 1],
    "seed_set_confirmation": [2, 3],
    "data": {
        "n_samples_small": 50,
        "n_samples_mid": 50,
        "n_features": 5,
        "class_priors": [0.5],
        "separation": 2.0,
        "label_frequency": 0.5,
        "sar_strength": 1.0,
        "pn_ratios": ["1:1:4"],
    },
    "timeouts": {
        "upu": 60,
        "nnpu": 60,
        "pnu": 60,
        "elkan_noto": 60,
        "ldce": 60,
        "kldce": 60,
        "llsvm": 120,
    },
    "methods": {"upu": {}, "llsvm": {}},
    "limitations": [],
}


def _write_config(tmp_path: Path, **overrides) -> Path:
    """Write a config derived from MINI_CONFIG with ``data``/top-level overrides."""
    cfg = dict(MINI_CONFIG)
    if "data" in overrides:
        cfg["data"] = {**cfg["data"], **overrides.pop("data")}
    cfg.update(overrides)
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


class TestRunnerMini:
    def test_basic_run_and_artifacts(self, tmp_path):
        config = load_config(_write_config(tmp_path))
        out = tmp_path / "out"
        trials, summary = run_trials(
            config, results_dir=out, seed_set="development", resume=False, progress=False
        )
        # 2 methods × 1 prior × 2 scales × 2 kinds (scar+sar) × 2 seeds = 16
        assert len(trials) == 16
        keys = list(zip(trials["algorithm"], trials["scenario"], trials["seed"], strict=True))
        assert len(set(keys)) == len(keys)  # every (algorithm, scenario, seed) unique
        assert (trials["status"] == "success").all()
        # no predict_proba (upu/llsvm) ⇒ calibration metrics NaN + reason, not a failure
        upu = trials[trials["algorithm"] == "upu"]
        assert upu["brier_score"].isna().all()
        assert upu["brier_score_unavailable_reason"].eq("requires predict_proba").all()
        assert not summary.empty
        write_artifacts(trials, summary, config, out)
        for name in (
            "resolved_config.json",
            "run_manifest.json",
            "trials.csv",
            "summary.csv",
            "report.md",
        ):
            assert (out / name).exists()
        manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["protocol"] == "traditional_pu_baseline"
        assert manifest["paper_claim"] is False
        assert manifest["n_trials"] == 16

    def test_basic_resume_skips_done_cells(self, tmp_path):
        config = load_config(_write_config(tmp_path))
        out = tmp_path / "out2"
        trials1, _ = run_trials(
            config, results_dir=out, seed_set="development", resume=False, progress=False
        )
        trials2, _ = run_trials(
            config, results_dir=out, seed_set="development", resume=True, progress=False
        )
        # resume re-uses persisted cells: same row count, no duplicated keys
        assert len(trials2) == len(trials1) == 16
        keys1 = set(zip(trials1["algorithm"], trials1["scenario"], trials1["seed"], strict=True))
        keys2 = set(zip(trials2["algorithm"], trials2["scenario"], trials2["seed"], strict=True))
        assert keys1 == keys2


class TestLoadConfig:
    def test_param_invalid_schema_version_raises(self, tmp_path):
        path = _write_config(tmp_path, schema_version=2)
        with pytest.raises(ValueError, match="schema_version"):
            load_config(path)

    def test_param_invalid_protocol_raises(self, tmp_path):
        path = _write_config(tmp_path, protocol="paper_like")
        with pytest.raises(ValueError, match="traditional_pu_baseline"):
            load_config(path)

    def test_param_empty_seeds_raises(self, tmp_path):
        path = _write_config(tmp_path, seed_set_development=[])
        with pytest.raises(ValueError, match="seed_set_development"):
            load_config(path)

    def test_param_unknown_method_raises(self, tmp_path):
        path = _write_config(tmp_path, methods={"upu": {}, "bogus": {}})
        with pytest.raises(ValueError, match="unknown method"):
            load_config(path)


class TestIllConditioned:
    def test_edge_ill_conditioned_unit_flagged(self, tmp_path):
        # label_frequency=0.0 ⇒ real_h=1.0 ⇒ |1-2πh| = 0 ⇒ ill-conditioned (checked
        # before any data generation / fit, so degenerate data never matters here).
        config = load_config(
            _write_config(
                tmp_path,
                methods={"ldce": {}},
                data={"label_frequency": 0.0},
            )
        )
        trials, _ = run_trials(
            config,
            results_dir=tmp_path / "out3",
            seed_set="development",
            resume=False,
            progress=False,
        )
        scar = trials[trials["scenario"].str.startswith("scar-pi0.5")]
        assert len(scar) == 4  # 2 scales × 2 seeds
        assert (scar["status"] == "failed").all()
        assert scar["failure_reason"].str.contains("ill_conditioned").all()


class TestDeterminism:
    def test_determ_repeat_runs_identical_metrics(self, tmp_path):
        config = load_config(_write_config(tmp_path))
        a, _ = run_trials(
            config,
            results_dir=tmp_path / "ra",
            seed_set="development",
            resume=False,
            progress=False,
        )
        b, _ = run_trials(
            config,
            results_dir=tmp_path / "rb",
            seed_set="development",
            resume=False,
            progress=False,
        )
        cols = METRIC_COLUMNS + ["status"]
        a = a.sort_values(["algorithm", "scenario", "seed"]).reset_index(drop=True)
        b = b.sort_values(["algorithm", "scenario", "seed"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[cols], b[cols])
