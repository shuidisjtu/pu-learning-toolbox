# ruff: noqa: N802, N803, N806, E501

"""Tests for the profile subcommand (pu-workflow step 1)."""

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
        "feature_0,feature_1,feature_2,feature_3,feature_4\n"
        + "\n".join(",".join(f"{v:.6f}" for v in row) for row in X)
        + "\n",
        encoding="utf-8",
    )
    y_path.write_text("label\n" + "\n".join(str(v) for v in y_pu) + "\n", encoding="utf-8")
    return X_path, y_path


@pytest.mark.unit
def test_basic_profile_writes_strict_json(tmp_path):
    X_path, y_path = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    args = build_parser().parse_args(
        ["profile", "--data", str(X_path), "--labels", str(y_path), "--out-dir", str(out_dir)]
    )
    args.func(args)
    payload = json.loads((out_dir / "profile.json").read_text(encoding="utf-8"))
    assert payload["selection_diagnostic"]["status"] in {
        "plausible",
        "at_risk",
        "inconclusive",
    }
    assert isinstance(payload["issues"], list)
    assert "summary" in payload


@pytest.mark.unit
def test_param_true_labels_accepted(tmp_path):
    X, y_pu, y_true = make_scar_dataset(n=60, c=0.5, n_features=5, separation=4.0, random_state=42)
    X_path = tmp_path / "X.csv"
    y_path = tmp_path / "y_pu.csv"
    yt_path = tmp_path / "y_true.csv"
    X_path.write_text("f0,f1,f2,f3,f4\n" + "\n".join(",".join(map(str, r)) for r in X) + "\n")
    y_path.write_text("label\n" + "\n".join(map(str, y_pu)) + "\n")
    yt_path.write_text("label\n" + "\n".join(map(str, y_true)) + "\n")
    out_dir = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "profile",
            "--data",
            str(X_path),
            "--labels",
            str(y_path),
            "--true-labels",
            str(yt_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    args.func(args)
    assert (out_dir / "profile.json").exists()


@pytest.mark.unit
def test_param_missing_labels_file_reports_error(tmp_path, capsys):
    X_path, y_path = _write_inputs(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "profile",
                "--data",
                str(X_path),
                "--labels",
                str(tmp_path / "none.csv"),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.unit
def test_edge_out_dir_is_existing_file_reports_error(tmp_path, capsys):
    X_path, y_path = _write_inputs(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "profile",
                "--data",
                str(X_path),
                "--labels",
                str(y_path),
                "--out-dir",
                str(blocker),
            ]
        )
    assert exc.value.code == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.unit
def test_edge_headerless_csv_reports_error(tmp_path, capsys):
    X_path = tmp_path / "noheader.csv"
    X_path.write_text("1.0,2.0\n3.0,4.0\n5.0,6.0\n", encoding="utf-8")
    y_path = tmp_path / "y.csv"
    y_path.write_text("label\n1\n0\n0\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "profile",
                "--data",
                str(X_path),
                "--labels",
                str(y_path),
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
                "profile",
                "--data",
                str(X_path),
                "--labels",
                str(y_path),
                "--out-dir",
                str(tmp_path / name),
            ]
        )
        args.func(args)
    a = (tmp_path / "a" / "profile.json").read_bytes()
    b = (tmp_path / "b" / "profile.json").read_bytes()
    assert a == b
