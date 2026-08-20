# ruff: noqa: N803, N806

"""Tests for shift-monitor and active-review CLI commands."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from pu_toolbox.cli import build_parser, main

pytestmark = pytest.mark.unit


class _ProbabilityModel:
    def predict_proba(self, X):
        probability = 1.0 / (1.0 + np.exp(-np.asarray(X)[:, 0]))
        return np.column_stack([1.0 - probability, probability])


def _monitor_arguments(tmp_path, *, out_name="out", window_id="w1", history=None):
    rng = np.random.default_rng(4)
    source = rng.normal(size=(60, 3))
    target = source + rng.normal(0.2, 0.05, source.shape)
    labels = np.zeros(60, dtype=int)
    labels[:15] = 1
    paths = {}
    for name, values in {"source": source, "target": target}.items():
        paths[name] = tmp_path / f"{name}.csv"
        pd.DataFrame(values, columns=["a", "b", "c"]).to_csv(paths[name], index=False)
    paths["labels"] = tmp_path / "labels.csv"
    pd.DataFrame({"label": labels}).to_csv(paths["labels"], index=False)
    values = [
        "shift-monitor",
        "--reference-data",
        str(paths["source"]),
        "--reference-labels",
        str(paths["labels"]),
        "--window-data",
        str(paths["target"]),
        "--window-labels",
        str(paths["labels"]),
        "--window-id",
        window_id,
        "--timestamp",
        "2026-08-21T00:00:00Z",
        "--out-dir",
        str(tmp_path / out_name),
        "--cv",
        "2",
        "--quiet",
    ]
    if history is not None:
        values.extend(["--history", str(history)])
    return values


def _review_arguments(tmp_path, *, out_name="review"):
    data = np.array([[-2.0, 0], [-0.1, 0], [0.1, 0], [2.0, 0]])
    labels = np.array([0, 0, 0, 1])
    data_path = tmp_path / "review_data.csv"
    labels_path = tmp_path / "review_labels.csv"
    model_path = tmp_path / "model.pkl"
    pd.DataFrame(data, columns=["a", "b"]).to_csv(data_path, index=False)
    pd.DataFrame({"label": labels}).to_csv(labels_path, index=False)
    model_path.write_bytes(pickle.dumps(_ProbabilityModel()))
    return [
        "review",
        "--model",
        str(model_path),
        "--data",
        str(data_path),
        "--labels",
        str(labels_path),
        "--out-dir",
        str(tmp_path / out_name),
        "--query-budget",
        "2",
        "--quiet",
    ]


def test_basic_shift_monitor_writes_history_and_window_artifacts(tmp_path):
    parsed = build_parser().parse_args(_monitor_arguments(tmp_path))
    parsed.func(parsed)
    out = tmp_path / "out"
    payload = json.loads((out / "shift_history.json").read_text())
    assert payload["n_windows"] == 1
    assert (out / "window_summary.json").is_file()
    assert (out / "window_shift.json").is_file()
    assert (out / "source_importance_weights.csv").is_file()


def test_param_monitor_appends_compatible_previous_history(tmp_path):
    first = build_parser().parse_args(_monitor_arguments(tmp_path))
    first.func(first)
    history = tmp_path / "out" / "shift_history.json"
    second = build_parser().parse_args(
        _monitor_arguments(tmp_path, out_name="out2", window_id="w2", history=history)
    )
    second.func(second)
    payload = json.loads((tmp_path / "out2" / "shift_history.json").read_text())
    assert [row["window_id"] for row in payload["windows"]] == ["w1", "w2"]


def test_basic_review_writes_selective_predictions_and_query_rows(tmp_path):
    parsed = build_parser().parse_args(_review_arguments(tmp_path))
    parsed.func(parsed)
    out = tmp_path / "review"
    payload = json.loads((out / "uncertainty.json").read_text())
    assert payload["summary"]["n_queries"] == 2
    rows = pd.read_csv(out / "uncertainty_rows.csv")
    assert rows["selected_for_review"].sum() == 2


def test_edge_shift_weighted_review_requires_weights(tmp_path, capsys):
    arguments = _review_arguments(tmp_path) + ["--query-strategy", "shift_weighted"]
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 1
    assert "importance_weight" in capsys.readouterr().err


def test_determ_review_artifacts_are_identical(tmp_path):
    first = _review_arguments(tmp_path, out_name="review_a")
    second = _review_arguments(tmp_path, out_name="review_b")
    build_parser().parse_args(first).func(build_parser().parse_args(first))
    build_parser().parse_args(second).func(build_parser().parse_args(second))
    assert (tmp_path / "review_a" / "uncertainty.json").read_bytes() == (
        tmp_path / "review_b" / "uncertainty.json"
    ).read_bytes()
