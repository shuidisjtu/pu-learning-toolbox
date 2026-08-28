"""Tests for the baseline-config drift gate (source defaults vs pinned config)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_baseline_configs as c  # noqa: E402


def _locked_config() -> dict:
    """The real locked v4 config, deep-copied for mutation per test."""
    path = c.CONFIGS_DIR / "seven_methods_pu_baseline_v4.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_basic_constructor_defaults_exclude_runner_injected_keys():
    """random_state/class_prior are runner-injected and must not be pinned."""
    for method in c.ESTIMATOR_CLASSES:
        defaults = c._constructor_defaults(method)
        assert "random_state" not in defaults
    assert "class_prior" not in c._constructor_defaults("nnpu")


@pytest.mark.unit
def test_basic_locked_config_pins_constructor_defaults_exactly():
    """The shipped locked config (v4) matches live constructor defaults."""
    config = _locked_config()
    assert c.check_config("seven_methods_pu_baseline_v4.json", config) == []


@pytest.mark.unit
def test_drifted_default_value_reported():
    """A pinned value that differs from the constructor default is an issue."""
    config = _locked_config()
    config["methods"]["ldce"]["max_iter"] = 100
    issues = c.check_config("cfg.json", config)
    assert len(issues) == 1
    assert "ldce.max_iter" in issues[0]
    assert "10000" in issues[0]


@pytest.mark.unit
def test_unpinned_default_key_reported():
    """A constructor default missing from the config is an unpinned issue."""
    config = _locked_config()
    del config["methods"]["kldce"]["inner_tol"]
    issues = c.check_config("cfg.json", config)
    assert len(issues) == 1
    assert "unpinned" in issues[0]
    assert "inner_tol" in issues[0]


@pytest.mark.unit
def test_unknown_pinned_key_reported():
    """A pinned key absent from the constructor signature is an issue."""
    config = _locked_config()
    config["methods"]["upu"]["retired_param"] = 42
    issues = c.check_config("cfg.json", config)
    assert len(issues) == 1
    assert "retired_param" in issues[0]


@pytest.mark.unit
def test_param_pinned_copy_mutation_isolated():
    """check_config never mutates the config it inspects."""
    config = _locked_config()
    snapshot = copy.deepcopy(config)
    c.check_config("cfg.json", config)
    assert config == snapshot


@pytest.mark.unit
def test_no_locked_config_fails_main(tmp_path, monkeypatch):
    """A config set with no locks_source_defaults flag fails the gate."""
    monkeypatch.setattr(c, "CONFIGS_DIR", tmp_path)
    assert c.main() == 1


@pytest.mark.unit
def test_edge_empty_methods_config_is_clean():
    """A locked config with no methods maps to no issues (empty boundary)."""
    assert c.check_config("empty.json", {"methods": {}}) == []


@pytest.mark.unit
def test_determ_repeated_checks_are_identical():
    """check_config is deterministic: repeated calls give identical issues."""
    config = _locked_config()
    config["methods"]["nnpu"]["max_epochs"] = 999
    assert c.check_config("cfg.json", config) == c.check_config("cfg.json", config)
