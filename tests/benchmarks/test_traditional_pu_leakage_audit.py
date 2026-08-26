"""Leakage audit gate tests (data_leakage_audit_design §5/§7, phase A).

Negative tests only: a gate that merely passes clean data proves nothing —
every check must be shown to block a deliberately planted leak.  The
preprocessing audit (design §5 item 4) belongs to phase B and is not
implemented or tested here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import benchmarks.traditional_pu.run as run_module
from benchmarks.traditional_pu.leakage_audit import (
    AUDIT_RULE_VERSION,
    LeakageAuditError,
    check_duplicate_samples,
    check_feature_blacklist,
    check_trial_columns,
    guard_fit_labels,
    run_leakage_preflight,
)

pytestmark = pytest.mark.unit


class TestFeatureBlacklist:
    def test_basic_feature_blacklist_rejects_y_true_column(self):
        """Design §5.1: a feature column literally named y_true must be blocked."""
        hits = check_feature_blacklist(["x1", "y_true", "x2"])
        assert [h["column"] for h in hits] == ["y_true"]
        assert all("pattern" in h and "reason" in h for h in hits)

    def test_basic_feature_blacklist_rejects_propensity_columns(self):
        """Design §5.2: propensity / selection probability columns must be blocked."""
        hits = check_feature_blacklist(["propensity", "selection_probability", "height"])
        assert {h["column"] for h in hits} == {"propensity", "selection_probability"}

    def test_param_feature_blacklist_case_insensitive_and_extra_patterns(self):
        """Design §2.2: matching is case-insensitive; callers may add extra terms."""
        hits = check_feature_blacklist(["Y_TRUE", "Score_Proxy"], extra_patterns=["score_proxy"])
        assert [h["column"] for h in hits] == ["Y_TRUE", "Score_Proxy"]

    def test_edge_feature_blacklist_fullname_match_ignores_similar_columns(self):
        """Full-name matching (not substring): look-alike columns must pass."""
        hits = check_feature_blacklist(["split_ratio", "prioritized", "fold_id"])
        assert hits == []


class TestDuplicateSamples:
    def test_basic_duplicate_samples_detects_cross_split_overlap(self):
        """Design §5.3: a sample duplicated across train/test must be detected."""
        shared = [1.0, 2.0]
        train = np.array([[0.0, 0.0], shared, [3.0, 4.0]])
        test = np.array([[5.0, 5.0], shared])
        report = check_duplicate_samples(train, test)
        assert report["n_overlap"] == 1
        assert report["train_indices"] == [1]
        assert report["test_indices"] == [1]

    def test_param_duplicate_samples_clean_and_empty_splits_pass(self):
        """Disjoint splits and empty arrays must report zero overlap."""
        train = np.array([[0.0, 0.0], [1.0, 1.0]])
        test = np.array([[2.0, 2.0], [3.0, 3.0]])
        assert check_duplicate_samples(train, test)["n_overlap"] == 0
        empty = np.empty((0, 2))
        assert check_duplicate_samples(train, empty)["n_overlap"] == 0
        assert check_duplicate_samples(empty, test)["n_overlap"] == 0

    def test_edge_duplicate_samples_shape_mismatch_raises(self):
        """A shape mismatch is a programming error — fail loudly, not silently."""
        train = np.array([[0.0, 0.0]])
        test = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="feature"):
            check_duplicate_samples(train, test)


class TestFitLabelGuard:
    def test_basic_guard_fit_labels_rejects_same_object_and_view_alias(self):
        """Design §5.5: passing y_true itself (or a view of it) to fit must be blocked."""
        y_true = np.array([1, 0, 1, 0])
        with pytest.raises(LeakageAuditError, match="leakage"):
            guard_fit_labels(y_true, y_true)
        with pytest.raises(LeakageAuditError, match="leakage"):
            guard_fit_labels(y_true[:], y_true)

    def test_edge_guard_fit_labels_equal_values_distinct_objects_pass(self):
        """label_frequency=1.0 legitimately makes y_pu == y_true in value — allowed."""
        y_true = np.array([1, 0, 1, 0])
        y_pu = np.array([1, 0, 1, 0])  # equal values, independent allocation
        guard_fit_labels(y_pu, y_true)  # must not raise


def _mini_config():
    """Minimal validated-config shape (mirrors protocol MINI_CONFIG, upu only)."""
    return {
        "schema_version": 1,
        "protocol": "traditional_pu_baseline",
        "seed_set_development": [0, 1],
        "seed_set_confirmation": [2, 3],
        "data": {
            "n_samples_small": 50,
            "n_samples_mid": 50,
            "n_features": 5,
            "class_priors": [0.5],
            "separation": 2.0,
            "label_frequency": 0.5,
        },
        "methods": {"upu": {}},
        "timeouts": {"upu": 60},
        "limitations": [],
    }


class TestPreflight:
    def test_basic_preflight_pass_writes_audit_json(self, tmp_path):
        """Design §3.1/§6: a clean synthetic preflight passes and persists the report."""
        config = _mini_config()
        audit = run_leakage_preflight(config, tmp_path / "results", seed_set="development")
        assert audit["status"] == "pass"
        path = tmp_path / "results" / "data_leakage_audit.json"
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert persisted["status"] == "pass"
        assert persisted["rule_version"] == AUDIT_RULE_VERSION
        assert persisted["seed_set"] == "development"
        assert persisted["seeds"] == [0, 1]
        assert persisted["generation_parameters"] == config["data"]
        assert persisted["scenario_hash"]
        checks = persisted["checks"]
        assert checks["feature_blacklist"]["status"] == "not_applicable"
        assert checks["feature_blacklist"]["reason"]
        assert checks["duplicate_samples"]["status"] == "not_applicable"
        assert checks["duplicate_samples"]["reason"]
        assert checks["y_true_path"]["enforcement"] == "runner_gate"
        assert checks["trial_label_columns"]["enforcement"] == "runner_gate"

    def test_param_preflight_blocked_persists_report_then_raises(self, tmp_path):
        """Design §4: a blocked preflight persists the reproducible cause, then raises."""
        config = _mini_config()
        results = tmp_path / "results"
        with pytest.raises(LeakageAuditError):
            run_leakage_preflight(
                config,
                results,
                seed_set="development",
                feature_columns=["label", "x1"],
            )
        persisted = json.loads((results / "data_leakage_audit.json").read_text(encoding="utf-8"))
        assert persisted["status"] == "blocked"
        assert [h["column"] for h in persisted["hits"]] == ["label"]

    def test_determ_preflight_scenario_hash_stable_and_seed_set_sensitive(self, tmp_path):
        """Same input hashes identically across runs; a different seed set changes it."""
        config = _mini_config()
        a = run_leakage_preflight(config, tmp_path / "a", seed_set="development")
        b = run_leakage_preflight(config, tmp_path / "b", seed_set="development")
        assert a["scenario_hash"] == b["scenario_hash"]
        c = run_leakage_preflight(config, tmp_path / "c", seed_set="confirmation")
        assert c["scenario_hash"] != a["scenario_hash"]


class TestCliGate:
    def test_basic_cli_main_runs_preflight_and_includes_audit_in_manifest(self, tmp_path):
        """CLI end-to-end: preflight runs, audit json exists, manifest records pass."""
        config_path = tmp_path / "mini.json"
        config_path.write_text(json.dumps(_mini_config()), encoding="utf-8")
        results = tmp_path / "results"
        code = run_module.main(
            ["--config", str(config_path), "--results-dir", str(results), "--no-resume"]
        )
        assert code == 0
        persisted = json.loads((results / "data_leakage_audit.json").read_text(encoding="utf-8"))
        assert persisted["status"] == "pass"
        manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["data_leakage_audit"]["status"] == "pass"

    def test_param_cli_main_preflight_block_returns_one(self, tmp_path, monkeypatch):
        """A blocked preflight exits 1 and never enters the grid."""
        config_path = tmp_path / "mini.json"
        config_path.write_text(json.dumps(_mini_config()), encoding="utf-8")

        def fake_preflight(
            config,
            results_dir,
            *,
            seed_set="development",
            feature_columns=None,
            split_pairs=None,
        ):
            raise LeakageAuditError("planted leak")

        monkeypatch.setattr(run_module, "run_leakage_preflight", fake_preflight)
        code = run_module.main(["--config", str(config_path), "--results-dir", str(tmp_path / "r")])
        assert code == 1


class TestTrialColumnGate:
    def test_param_trial_column_gate_rejects_y_columns_allows_clean(self):
        """Design §5.6: any y_* raw-label column in a trial row must be blocked."""
        hits = check_trial_columns(["algorithm", "y_true", "Y_PN", "seed"])
        assert hits == ["y_true", "Y_PN"]
        assert check_trial_columns(["algorithm", "scenario", "seed"]) == []
