# ruff: noqa: N803, N806

"""Tests for repeated PU shift monitoring."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pu_toolbox.diagnostics import PUShiftMonitor

pytestmark = pytest.mark.unit


def _data(seed=4):
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(80, 3))
    labels = np.zeros(80, dtype=int)
    labels[:16] = 1
    return rng, source, labels


def test_basic_monitor_records_ordered_windows_and_deltas():
    rng, source, labels = _data()
    monitor = PUShiftMonitor(source, labels, cv=2)
    first, _ = monitor.update(
        source + rng.normal(0.1, 0.05, source.shape),
        window_id="w1",
        y_window_pu=labels,
        timestamp="2026-01-01T00:00:00Z",
    )
    second, _ = monitor.update(
        source + rng.normal(0.4, 0.05, source.shape),
        window_id="w2",
        y_window_pu=labels,
        timestamp="2026-02-01T00:00:00Z",
    )
    assert first.auc_change is None
    assert second.auc_change == pytest.approx(second.domain_auc - first.domain_auc)
    assert [item.window_id for item in monitor.history] == ["w1", "w2"]


def test_edge_missing_target_labels_raise_adaptation_alert():
    _, source, labels = _data()
    window, _ = PUShiftMonitor(source, labels, cv=2).update(source, window_id="missing")
    assert window.alert_level == "critical"
    assert "adaptation_not_ready" in window.alert_codes


def test_param_duplicate_id_and_bad_timestamp_are_rejected():
    _, source, labels = _data()
    monitor = PUShiftMonitor(source, labels, cv=2)
    monitor.update(source, window_id="same", y_window_pu=labels)
    with pytest.raises(ValueError, match="already exists"):
        monitor.update(source, window_id="same", y_window_pu=labels)
    with pytest.raises(ValueError, match="ISO-8601"):
        monitor.update(source, window_id="new", y_window_pu=labels, timestamp="yesterday")


def test_determ_history_round_trip_and_configuration_guard(tmp_path):
    _, source, labels = _data()
    monitor = PUShiftMonitor(source, labels, cv=2)
    monitor.update(source, window_id="w1", y_window_pu=labels, timestamp="2026-01-01T00:00:00Z")
    path = monitor.save_history(tmp_path / "history.json")
    restored = PUShiftMonitor(source, labels, cv=2).load_history(path)
    assert restored.to_dict() == monitor.to_dict()
    assert json.loads(path.read_text())["n_windows"] == 1
    with pytest.raises(ValueError, match="configuration"):
        PUShiftMonitor(source, labels, cv=3).load_history(path)


def test_edge_invalid_threshold_is_rejected():
    _, source, labels = _data()
    with pytest.raises(ValueError, match="auc_jump_threshold"):
        PUShiftMonitor(source, labels, auc_jump_threshold=-0.1)
