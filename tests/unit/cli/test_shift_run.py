# ruff: noqa: N803, N806

"""Tests for the paired ``shift-run`` command."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pu_toolbox.cli import build_parser, main

pytestmark = pytest.mark.unit


def _arguments(tmp_path):
    rng = np.random.default_rng(8)
    source = rng.normal(size=(60, 3))
    target = source + rng.normal(0.2, 0.05, source.shape)
    labels = np.zeros(60, dtype=int)
    labels[:15] = 1
    truth = np.zeros(60, dtype=int)
    truth[:30] = 1
    paths = {}
    for name, values in {"source": source, "target": target}.items():
        paths[name] = tmp_path / f"{name}.csv"
        pd.DataFrame(values, columns=["a", "b", "c"]).to_csv(paths[name], index=False)
    for name, values in {"source_labels": labels, "target_labels": labels, "truth": truth}.items():
        paths[name] = tmp_path / f"{name}.csv"
        pd.DataFrame({"label": values}).to_csv(paths[name], index=False)
    return [
        "shift-run",
        "--source-data",
        str(paths["source"]),
        "--source-labels",
        str(paths["source_labels"]),
        "--target-data",
        str(paths["target"]),
        "--target-labels",
        str(paths["target_labels"]),
        "--target-true-labels",
        str(paths["truth"]),
        "--class-prior",
        "0.5",
        "--target-class-prior",
        "0.5",
        "--out-dir",
        str(tmp_path / "out"),
        "--cv",
        "2",
        "--quiet",
    ]


def test_basic_shift_run_writes_comparison_artifacts(tmp_path):
    args = _arguments(tmp_path)
    parsed = build_parser().parse_args(args)
    parsed.func(parsed)
    out = tmp_path / "out"
    payload = json.loads((out / "shift_comparison.json").read_text())
    assert payload["analysis_type"] == "shift_adaptation_comparison"
    assert (out / "shift_comparison.md").is_file()
    assert (out / "shift_report.json").is_file()
    assert (out / "source_importance_weights.csv").is_file()
    assert (out / "target_predictions.csv").is_file()


def test_edge_target_labels_are_required(tmp_path, capsys):
    args = _arguments(tmp_path)
    index = args.index("--target-labels")
    del args[index : index + 2]
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(args)
    assert exc.value.code == 2
    assert "--target-labels" in capsys.readouterr().err


def test_param_invalid_minimum_is_user_error(tmp_path, capsys):
    args = _arguments(tmp_path) + ["--min-improvement", "-1"]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 1
    assert "min_improvement" in capsys.readouterr().err
