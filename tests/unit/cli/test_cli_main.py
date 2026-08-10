# ruff: noqa: N802, N803, N806, E501

"""Tests for the CLI entry point (parser building, dispatch, exit codes)."""

from __future__ import annotations

import pytest

from pu_toolbox.cli import build_parser, main
from pu_toolbox.cli.run import _parse_prior_params


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
    assert args.prior_estimator == "pen_l1"
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


@pytest.mark.unit
def test_param_prior_param_parsing_and_coercion():
    """--prior-param KEY=VALUE parses, coerces types, and rejects bad input."""
    params = _parse_prior_params(["sigma=3.0", "n_centers=200", "theta_grid=x"])
    assert params == {"sigma": 3.0, "n_centers": 200, "theta_grid": "x"}
    assert _parse_prior_params(None) == {}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        _parse_prior_params(["sigma"])
    with pytest.raises(ValueError, match="non-empty key"):
        _parse_prior_params(["=3.0"])


@pytest.mark.unit
def test_param_run_parser_accepts_repeated_prior_params():
    """parse_args collects repeated --prior-param into a list."""
    args = build_parser().parse_args(
        [
            "run",
            "--data",
            "x.csv",
            "--labels",
            "y.csv",
            "--out-dir",
            "out",
            "--prior-param",
            "sigma=3.0",
            "--prior-param",
            "reg_lambda=1e-2",
        ]
    )
    assert args.prior_param == ["sigma=3.0", "reg_lambda=1e-2"]


@pytest.mark.unit
def test_deterministic_help_output_stable(capsys):
    """Same argv produces byte-identical stdout and exit code."""
    with pytest.raises(SystemExit) as first_exit:
        main(["--help"])
    first_out = capsys.readouterr().out
    with pytest.raises(SystemExit) as second_exit:
        main(["--help"])
    second_out = capsys.readouterr().out
    assert first_exit.value.code == second_exit.value.code == 0
    assert first_out == second_out
