# ruff: noqa: N802, N803, N806, E501

"""Tests for the sensitivity subcommand (pu-workflow step 4)."""

from __future__ import annotations

import json

import pytest

from pu_toolbox.cli import build_parser, main
from pu_toolbox.preprocessing.pu_labeling import make_scar_dataset


def _write_inputs(tmp_path, n=60, seed=42):
    X, y_pu, _ = make_scar_dataset(n=n, c=0.5, n_features=5, separation=4.0, random_state=seed)
    X_path = tmp_path / "X.csv"
    y_path = tmp_path / "y_pu.csv"
    X_path.write_text(
        "f0,f1,f2,f3,f4\n" + "\n".join(",".join(map(str, r)) for r in X) + "\n",
        encoding="utf-8",
    )
    y_path.write_text("label\n" + "\n".join(map(str, y_pu)) + "\n", encoding="utf-8")
    return X_path, y_path


@pytest.mark.unit
def test_basic_sensitivity_writes_nine_points(tmp_path):
    X_path, y_path = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "sensitivity",
            "--data",
            str(X_path),
            "--labels",
            str(y_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    args.func(args)
    payload = json.loads((out_dir / "sensitivity.json").read_text(encoding="utf-8"))
    assert payload["analysis_type"] == "fixed_output_assumption_sensitivity"
    assert len(payload["points"]) == 9


@pytest.mark.unit
def test_param_custom_grid_size(tmp_path):
    X_path, y_path = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "sensitivity",
            "--data",
            str(X_path),
            "--labels",
            str(y_path),
            "--class-priors",
            "0.2,0.4,0.6,0.8",
            "--out-dir",
            str(out_dir),
        ]
    )
    args.func(args)
    payload = json.loads((out_dir / "sensitivity.json").read_text(encoding="utf-8"))
    assert len(payload["points"]) == 4


@pytest.mark.unit
def test_param_prior_requiring_classifier_reports_error(tmp_path, capsys):
    X_path, y_path = _write_inputs(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "sensitivity",
                "--data",
                str(X_path),
                "--labels",
                str(y_path),
                "--classifier",
                "upu",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 1
    assert "requires a class prior" in capsys.readouterr().err


@pytest.mark.unit
def test_edge_invalid_grid_reports_error(tmp_path, capsys):
    X_path, y_path = _write_inputs(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "sensitivity",
                "--data",
                str(X_path),
                "--labels",
                str(y_path),
                "--class-priors",
                "0.1,abc",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 1
    assert "cannot parse class-priors grid" in capsys.readouterr().err


@pytest.mark.unit
def test_edge_unknown_classifier_reports_error(tmp_path, capsys):
    X_path, y_path = _write_inputs(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "sensitivity",
                "--data",
                str(X_path),
                "--labels",
                str(y_path),
                "--classifier",
                "not_a_method",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.unit
def test_determ_same_input_same_output(tmp_path):
    X_path, y_path = _write_inputs(tmp_path)
    for name in ("a", "b"):
        args = build_parser().parse_args(
            [
                "sensitivity",
                "--data",
                str(X_path),
                "--labels",
                str(y_path),
                "--out-dir",
                str(tmp_path / name),
            ]
        )
        args.func(args)
    a = (tmp_path / "a" / "sensitivity.json").read_bytes()
    b = (tmp_path / "b" / "sensitivity.json").read_bytes()
    assert a == b
