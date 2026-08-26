"""Runner orchestration tests: state machine, resume, artifacts, failure isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import benchmarks.traditional_pu.runner as runner
from benchmarks.traditional_pu.leakage_audit import (
    AUDIT_RULE_VERSION,
    run_leakage_preflight,
)
from benchmarks.traditional_pu.run import _write_timeout_profile
from benchmarks.traditional_pu.runner import (
    METRIC_COLUMNS,
    _iter_scenario_specs,
    _run_one_trial,
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
        # Leakage audit (design §3.4): without a preflight report the artifact
        # set records audit_only — no audit proof, no performance claims.
        assert manifest["data_leakage_audit"]["status"] == "audit_only"

        # With a real preflight report the manifest embeds it verbatim…
        audit = run_leakage_preflight(config, out, seed_set="development")
        write_artifacts(trials, summary, config, out, leakage_audit=audit)
        manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["data_leakage_audit"]["status"] == "pass"
        assert manifest["data_leakage_audit"]["rule_version"] == AUDIT_RULE_VERSION

        # …and a report from a different config must never vouch for this one.
        other = load_config(_write_config(tmp_path, data={"label_frequency": 0.2}))
        stale_audit = run_leakage_preflight(other, out, seed_set="development")
        with pytest.raises(ValueError, match="stale"):
            write_artifacts(trials, summary, config, out, leakage_audit=stale_audit)

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


class TestMetricFailureIsolation:
    def test_param_non_proba_metric_valueerror_fails_trial(self, tmp_path, monkeypatch):
        # A ValueError from a non-calibration metric (e.g. pu_auc_roc on
        # degenerate labels) is a genuine trial failure — it must fail the
        # row, not masquerade as an unavailable metric inside a success row.
        config = load_config(_write_config(tmp_path))
        original = runner._metric_value

        def patched(name, *args, **kwargs):
            if name == "pu_auc_roc":
                raise ValueError("degenerate labels")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(runner, "_metric_value", patched)
        spec = next(s for _, _, s in _iter_scenario_specs(config) if s["kind"] == "scar")
        row = _run_one_trial("upu", spec, 0, config)
        assert row["status"] == "failed"
        assert "ValueError('degenerate labels')" in row["failure_reason"]

    def test_param_proba_metric_valueerror_marks_unavailable(self, tmp_path, monkeypatch):
        # Calibration-metric ValueErrors keep the documented availability
        # semantics: NaN + "invalid probability output (not in [0,1])",
        # and the row stays a success.
        config = load_config(_write_config(tmp_path, methods={"elkan_noto": {}}))
        original = runner._metric_value

        def patched(name, *args, **kwargs):
            if name == "brier_score":
                raise ValueError("proba out of range")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(runner, "_metric_value", patched)
        spec = next(s for _, _, s in _iter_scenario_specs(config) if s["kind"] == "scar")
        row = _run_one_trial("elkan_noto", spec, 0, config)
        assert row["status"] == "success"
        assert np.isnan(row["brier_score"])
        assert row["brier_score_unavailable_reason"] == "invalid probability output (not in [0,1])"


class TestPnuProtocol:
    def test_basic_pnu_protocol_cell_counts(self, tmp_path):
        # PNU runs only its own tri-label cells: 1 ratio × 2 scales × 2 seeds.
        config = load_config(_write_config(tmp_path, methods={"pnu": {}}))
        trials, _ = run_trials(
            config,
            results_dir=tmp_path / "outpnu",
            seed_set="development",
            resume=False,
            progress=False,
        )
        assert len(trials) == 4
        assert (trials["algorithm"] == "pnu").all()
        assert trials["scenario"].str.startswith("pnu-").all()
        assert (trials["status"] == "success").all()

    def test_param_pnu_grid_emits_no_scar_sar_cells(self, tmp_path):
        # Contract §2.2: PNU must never run under the SCAR/SAR binary-PU cells.
        config = load_config(_write_config(tmp_path, methods={"pnu": {}}))
        cells = list(_iter_scenario_specs(config))
        assert len(cells) == 2  # 1 ratio × 2 scales
        assert all(name == "pnu" for name, _, _ in cells)
        assert all(not s.startswith("scar-") and not s.startswith("sar-") for _, s, _ in cells)

    def test_edge_pnu_tri_label_metrics_unavailable(self, tmp_path):
        # Binary-PU-label metrics are NaN + reason in PNU cells; the oracle
        # metrics still compute on the {0, 1} ground truth.
        config = load_config(_write_config(tmp_path, methods={"pnu": {}}))
        trials, _ = run_trials(
            config,
            results_dir=tmp_path / "outpnu2",
            seed_set="development",
            resume=False,
            progress=False,
        )
        row = trials.iloc[0]
        assert np.isnan(row["pu_recall"])
        assert (
            row["pu_recall_unavailable_reason"] == "requires binary PU labels (PNU tri-label cell)"
        )
        assert np.isnan(row["pu_zero_one_risk"])
        assert (
            row["pu_zero_one_risk_unavailable_reason"]
            == "requires binary PU labels (PNU tri-label cell)"
        )
        assert np.isnan(row["brier_score"])
        assert row["brier_score_unavailable_reason"] == "requires predict_proba"
        assert not np.isnan(row["pu_auc_roc"])


class TestSarGrid:
    def test_param_sar_grid_skips_ldce_kldce(self, tmp_path):
        # Contract §2.3: SAR is a propensity-mechanism diagnostic for the
        # PU-family methods; the flip-model methods LDCE/KLDCE run on SCAR only.
        config = load_config(_write_config(tmp_path, methods={"ldce": {}, "llsvm": {}}))
        trials, _ = run_trials(
            config,
            results_dir=tmp_path / "outsar",
            seed_set="development",
            resume=False,
            progress=False,
        )
        ldce = trials[trials["algorithm"] == "ldce"]
        assert len(ldce) == 4  # 2 scales × 2 seeds, SCAR only
        assert not ldce["scenario"].str.startswith("sar-").any()
        assert ldce["scenario"].str.startswith("scar-").all()
        # llsvm keeps its SAR diagnostic line
        assert trials["scenario"].str.startswith("sar-").any()
        # grid-level: kldce is gated out of SAR too
        kldce_cfg = load_config(_write_config(tmp_path, methods={"kldce": {}}))
        cells = list(_iter_scenario_specs(kldce_cfg))
        assert all(not s.startswith("sar-") for _, s, _ in cells)
        assert any(s.startswith("scar-") for _, s, _ in cells)


class TestTimeoutProfile:
    def test_param_profile_falls_back_to_pnu_cell_for_pnu_only_config(self, tmp_path):
        # --timeout-profile logs one minimal trial per algorithm; pnu-only
        # configs have no scar-small cells (contract §2.2), so it must fall
        # back to the first PNU cell instead of StopIteration.
        config = load_config(_write_config(tmp_path, methods={"pnu": {}}))
        out_csv = tmp_path / "timeout_profile.csv"
        _write_timeout_profile(config, str(out_csv))
        df = pd.read_csv(out_csv)
        assert list(df["algorithm"]) == ["pnu"]
        assert list(df["n_samples"]) == [50]  # first PNU cell = small scale
        assert list(df["n_features"]) == [5]
        assert not df["elapsed_seconds"].isna().any()
