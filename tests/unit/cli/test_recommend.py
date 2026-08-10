# ruff: noqa: N802, N803, N806, E501

"""Tests for the recommend subcommand (pu-workflow step 2)."""

from __future__ import annotations

import json

import pytest

from pu_toolbox.cli import build_parser, main
from pu_toolbox.preprocessing.data_profiler import profile_pu_data
from pu_toolbox.preprocessing.pu_labeling import make_scar_dataset


def _write_profile(tmp_path):
    X, y_pu, _ = make_scar_dataset(n=60, c=0.5, n_features=5, separation=4.0, random_state=42)
    profile = profile_pu_data(X, y_pu)
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(profile.to_dict(), ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return path


def _write_data(tmp_path):
    X, y_pu, _ = make_scar_dataset(n=60, c=0.5, n_features=5, separation=4.0, random_state=42)
    X_path = tmp_path / "X.csv"
    y_path = tmp_path / "y_pu.csv"
    X_path.write_text(
        "f0,f1,f2,f3,f4\n" + "\n".join(",".join(map(str, r)) for r in X) + "\n",
        encoding="utf-8",
    )
    y_path.write_text("label\n" + "\n".join(map(str, y_pu)) + "\n", encoding="utf-8")
    return X_path, y_path


@pytest.mark.unit
def test_basic_recommend_writes_candidates(tmp_path):
    profile_path = _write_profile(tmp_path)
    out_dir = tmp_path / "out"
    args = build_parser().parse_args(
        ["recommend", "--profile", str(profile_path), "--out-dir", str(out_dir)]
    )
    args.func(args)
    payload = json.loads((out_dir / "recommendation.json").read_text(encoding="utf-8"))
    assert isinstance(payload["candidates"], list)
    assert len(payload["candidates"]) > 0
    assert payload["candidates"][0]["rank"] == 1


@pytest.mark.unit
def test_param_explicit_class_prior_and_gpu_flag(tmp_path):
    profile_path = _write_profile(tmp_path)
    out_dir = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "recommend",
            "--profile",
            str(profile_path),
            "--class-prior",
            "0.3",
            "--has-gpu",
            "--out-dir",
            str(out_dir),
        ]
    )
    args.func(args)
    payload = json.loads((out_dir / "recommendation.json").read_text(encoding="utf-8"))
    assert payload["provenance"]["has_gpu"] is True
    assert len(payload["candidates"]) > 0


@pytest.mark.unit
def test_param_prior_estimator_requires_data(tmp_path, capsys):
    profile_path = _write_profile(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "recommend",
                "--profile",
                str(profile_path),
                "--prior-estimator",
                "recpe",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 1
    assert "requires --data and --labels" in capsys.readouterr().err


@pytest.mark.unit
def test_edge_unknown_prior_estimator_reports_error(tmp_path, capsys):
    profile_path = _write_profile(tmp_path)
    X_path, y_path = _write_data(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "recommend",
                "--profile",
                str(profile_path),
                "--prior-estimator",
                "not_a_prior",
                "--data",
                str(X_path),
                "--labels",
                str(y_path),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 1
    assert "Unknown prior estimator" in capsys.readouterr().err


@pytest.mark.unit
def test_edge_missing_profile_reports_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "recommend",
                "--profile",
                str(tmp_path / "none.json"),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.unit
def test_determ_same_input_same_output(tmp_path):
    profile_path = _write_profile(tmp_path)
    for name in ("a", "b"):
        args = build_parser().parse_args(
            ["recommend", "--profile", str(profile_path), "--out-dir", str(tmp_path / name)]
        )
        args.func(args)
    a = (tmp_path / "a" / "recommendation.json").read_bytes()
    b = (tmp_path / "b" / "recommendation.json").read_bytes()
    assert a == b
