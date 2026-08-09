# ruff: noqa: N802, N803, N806, E501

"""Tests for the make-demo-data subcommand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pu_toolbox.cli import build_parser, main


@pytest.mark.unit
def test_demo_writes_three_files(tmp_path):
    """X/y_pu/y_true CSVs are written with 2n rows."""
    args = build_parser().parse_args(
        ["make-demo-data", "--out-dir", str(tmp_path), "--n", "50", "--seed", "1"]
    )
    args.func(args)
    for name in ("X.csv", "y_pu.csv", "y_true.csv"):
        assert (tmp_path / name).exists()
    assert len(pd.read_csv(tmp_path / "X.csv")) == 100


@pytest.mark.unit
def test_param_too_few_labeled_positives_reports_error(tmp_path, capsys):
    """n=3 at c=0.5 labels only 2 positives -- fewer than the default 5-fold
    CV demands -- so the guard must refuse instead of writing CSVs that
    `run` cannot consume (regression: the old --n>=3 guard passed but the
    demo still failed under the default 5-fold CV)."""
    with pytest.raises(SystemExit) as exc:
        main(["make-demo-data", "--out-dir", str(tmp_path / "d"), "--n", "3"])
    assert exc.value.code == 1
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "d" / "X.csv").exists()


@pytest.mark.unit
def test_edge_n_below_minimum_reports_error(tmp_path, capsys):
    """--n < 3 fails with a clear error instead of writing unusable CSVs."""
    with pytest.raises(SystemExit) as exc:
        main(["make-demo-data", "--out-dir", str(tmp_path / "d"), "--n", "1"])
    assert exc.value.code == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.unit
def test_basic_demo_output_consumable_by_run(tmp_path):
    """Demo output feeds straight into `run` (self-contained loop)."""
    demo_dir = tmp_path / "demo"
    out = tmp_path / "out"
    args = build_parser().parse_args(
        ["make-demo-data", "--out-dir", str(demo_dir), "--n", "30", "--seed", "42"]
    )
    args.func(args)
    main(
        [
            "run",
            "--data",
            str(demo_dir / "X.csv"),
            "--labels",
            str(demo_dir / "y_pu.csv"),
            "--out-dir",
            str(out),
            "--cv",
            "3",
            "--seed",
            "42",
            "--classifier",
            "upu",
            "--quiet",
        ]
    )
    assert (out / "report.json").exists()


@pytest.mark.unit
def test_demo_seed_deterministic(tmp_path):
    """Same seed → byte-identical feature CSVs."""
    a, b = tmp_path / "a", tmp_path / "b"
    args = build_parser().parse_args(
        ["make-demo-data", "--out-dir", str(a), "--n", "50", "--seed", "7"]
    )
    args.func(args)
    args = build_parser().parse_args(
        ["make-demo-data", "--out-dir", str(b), "--n", "50", "--seed", "7"]
    )
    args.func(args)
    assert np.array_equal(pd.read_csv(a / "X.csv").to_numpy(), pd.read_csv(b / "X.csv").to_numpy())
