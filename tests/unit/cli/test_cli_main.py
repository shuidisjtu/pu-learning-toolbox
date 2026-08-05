# ruff: noqa: N802, N803, N806, E501

"""Tests for the CLI entry point (parser building, dispatch, exit codes)."""

from __future__ import annotations

import pytest

from pu_toolbox.cli import build_parser, main


@pytest.mark.unit
def test_edge_help_exits_zero(capsys):
    """--help prints usage and exits 0."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "pu-toolbox" in out
    assert "run" in out


@pytest.mark.unit
def test_no_command_exits_two():
    """Missing subcommand is an argparse error (exit 2)."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


@pytest.mark.unit
def test_unknown_command_exits_two():
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code == 2


@pytest.mark.unit
def test_basic_run_subcommand_registered_with_defaults():
    """parse_args resolves the run subcommand with documented defaults."""
    args = build_parser().parse_args(
        ["run", "--data", "x.csv", "--labels", "y.csv", "--out-dir", "out"]
    )
    assert args.command == "run"
    assert args.classifier == "auto"
    assert args.prior_estimator == "recpe"
    assert args.cv == 5
    assert args.seed == 42
    assert args.save_model is False
    assert args.quiet is False


@pytest.mark.unit
def test_param_invalid_cv_value_exits_two():
    """A non-integer --cv is an argparse type error (exit 2)."""
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--data",
                "x.csv",
                "--labels",
                "y.csv",
                "--out-dir",
                "out",
                "--cv",
                "abc",
            ]
        )
    assert exc.value.code == 2
