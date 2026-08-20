# ruff: noqa: N803, N806

"""Tests for the ``shift-audit`` command."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pu_toolbox.cli import build_parser, main

pytestmark = pytest.mark.unit


def _write_domains(tmp_path, *, seed=42, target_features=3):
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(60, 3))
    target = rng.normal(loc=0.3, size=(50, target_features))
    source_labels = np.zeros(60, dtype=int)
    target_labels = np.zeros(50, dtype=int)
    source_labels[:12] = 1
    target_labels[:10] = 1
    paths = {
        "source": tmp_path / "source.csv",
        "source_labels": tmp_path / "source_labels.csv",
        "target": tmp_path / "target.csv",
        "target_labels": tmp_path / "target_labels.csv",
    }
    pd.DataFrame(source, columns=["f0", "f1", "f2"]).to_csv(paths["source"], index=False)
    pd.DataFrame({"label": source_labels}).to_csv(paths["source_labels"], index=False)
    pd.DataFrame(target, columns=[f"f{i}" for i in range(target_features)]).to_csv(
        paths["target"], index=False
    )
    pd.DataFrame({"label": target_labels}).to_csv(paths["target_labels"], index=False)
    return paths


def _arguments(paths, out_dir, *, target_labels=True):
    values = [
        "shift-audit",
        "--source-data",
        str(paths["source"]),
        "--source-labels",
        str(paths["source_labels"]),
        "--target-data",
        str(paths["target"]),
        "--out-dir",
        str(out_dir),
        "--cv",
        "2",
    ]
    if target_labels:
        values.extend(["--target-labels", str(paths["target_labels"])])
    return values


def test_basic_command_writes_three_stable_artifacts(tmp_path):
    paths = _write_domains(tmp_path)
    out_dir = tmp_path / "out"
    args = build_parser().parse_args(_arguments(paths, out_dir))
    args.func(args)

    payload = json.loads((out_dir / "shift_report.json").read_text(encoding="utf-8"))
    assert payload["analysis_type"] == "marginal_distribution_shift_audit"
    assert payload["adaptation_ready"] is True
    assert (out_dir / "shift_report.md").is_file()
    assert (out_dir / "source_importance_weights.csv").is_file()


def test_param_target_labels_are_optional_for_audit(tmp_path):
    paths = _write_domains(tmp_path)
    out_dir = tmp_path / "out"
    args = build_parser().parse_args(_arguments(paths, out_dir, target_labels=False))
    args.func(args)
    payload = json.loads((out_dir / "shift_report.json").read_text(encoding="utf-8"))
    assert payload["adaptation_ready"] is False
    assert any(issue["code"] == "target_pu_missing" for issue in payload["issues"])


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--alpha", "0", "alpha"),
        ("--probability-clip", "0.5", "probability_clip"),
        ("--cv", "1", "cv"),
    ],
)
def test_param_invalid_numeric_option_reports_user_error(tmp_path, capsys, option, value, message):
    paths = _write_domains(tmp_path)
    arguments = _arguments(paths, tmp_path / "out")
    arguments.extend([option, value])
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 1
    assert message in capsys.readouterr().err


def test_edge_feature_count_mismatch_reports_user_error(tmp_path, capsys):
    paths = _write_domains(tmp_path, target_features=4)
    with pytest.raises(SystemExit) as exc:
        main(_arguments(paths, tmp_path / "out"))
    assert exc.value.code == 1
    assert "same number" in capsys.readouterr().err


def test_edge_missing_source_file_reports_user_error(tmp_path, capsys):
    paths = _write_domains(tmp_path)
    paths["source"] = tmp_path / "missing.csv"
    with pytest.raises(SystemExit) as exc:
        main(_arguments(paths, tmp_path / "out"))
    assert exc.value.code == 1
    assert "file not found" in capsys.readouterr().err


def test_determ_same_inputs_produce_identical_json(tmp_path):
    paths = _write_domains(tmp_path)
    outputs = []
    for name in ("a", "b"):
        out_dir = tmp_path / name
        args = build_parser().parse_args(_arguments(paths, out_dir))
        args.func(args)
        outputs.append((out_dir / "shift_report.json").read_bytes())
    assert outputs[0] == outputs[1]
