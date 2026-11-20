# ruff: noqa: N802, N803, N806, E501

"""Tests for the make-demo-data subcommand."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pu_toolbox.cli import build_parser, main


@pytest.mark.unit
def test_demo_writes_aligned_observations_truth_and_manifest(tmp_path):
    """The manifest identifies the separate, row-aligned observation and audit artifacts."""
    args = build_parser().parse_args(
        ["make-demo-data", "--out-dir", str(tmp_path), "--n", "50", "--seed", "1"]
    )
    args.func(args)
    for name in ("X.csv", "y_pu.csv", "y_true.csv", "sample_ids.csv", "construction.json"):
        assert (tmp_path / name).exists()
    assert len(pd.read_csv(tmp_path / "X.csv")) == 100
    observed = pd.read_csv(tmp_path / "y_pu.csv")["label"]
    truth = pd.read_csv(tmp_path / "y_true.csv")["label"]
    ids = pd.read_csv(tmp_path / "sample_ids.csv")["sample_id"]
    manifest = json.loads((tmp_path / "construction.json").read_text(encoding="utf-8"))
    assert len(ids) == ids.nunique() == len(truth) == len(observed) == 100
    assert (observed <= truth).all()
    assert manifest["input_sampling"] == "os"
    assert manifest["labeling"]["n_labeled_actual"] == observed.sum() == 25
    assert manifest["cli_seed"] == 1


@pytest.mark.unit
def test_demo_default_separation_is_one():
    """The default separation (1.0) avoids the strong-separation regime where
    pen_l1 systematically under-estimates the prior (0.33 at sep=4.0 vs 0.44
    at sep=1.0 on n=200 SCAR data); the demo's first impression must not be
    a wildly off prior estimate."""
    args = build_parser().parse_args(["make-demo-data", "--out-dir", "d", "--n", "50"])
    assert args.separation == 1.0


@pytest.mark.unit
def test_param_too_few_labeled_positives_reports_error(tmp_path, capsys):
    """The demo refuses to write data with fewer than five labeled positives."""
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
def test_demo_seed_deterministic(tmp_path):
    """Same seed → byte-identical artifacts."""
    a, b = tmp_path / "a", tmp_path / "b"
    args = build_parser().parse_args(
        ["make-demo-data", "--out-dir", str(a), "--n", "50", "--seed", "7"]
    )
    args.func(args)
    args = build_parser().parse_args(
        ["make-demo-data", "--out-dir", str(b), "--n", "50", "--seed", "7"]
    )
    args.func(args)
    for name in ("X.csv", "y_pu.csv", "y_true.csv", "sample_ids.csv", "construction.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


@pytest.mark.unit
def test_determ_demo_keeps_class_order_and_the_sequential_random_stream(tmp_path):
    seed = 3
    args = build_parser().parse_args(
        ["make-demo-data", "--out-dir", str(tmp_path), "--n", "12", "--seed", str(seed)]
    )
    args.func(args)
    rng = np.random.RandomState(seed)
    expected_x = np.vstack((rng.randn(12, 5) + 0.5, rng.randn(12, 5) - 0.5))
    expected_labels = np.zeros(24, dtype=int)
    expected_labels[rng.choice(12, size=6, replace=False)] = 1
    np.testing.assert_allclose(pd.read_csv(tmp_path / "X.csv"), expected_x, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(pd.read_csv(tmp_path / "y_pu.csv")["label"], expected_labels)
