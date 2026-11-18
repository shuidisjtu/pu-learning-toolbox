# ruff: noqa: N802, N803, N806, E501

"""Tests for the CLI entry point (parser building, dispatch, exit codes)."""

from __future__ import annotations

import pytest

from pu_toolbox.cli import main


@pytest.mark.unit
def test_edge_help_exits_zero(capsys):
    """--help prints usage and exits 0."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "pu-toolbox" in out
    assert "list-methods" in out
    assert "make-demo-data" in out


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


@pytest.mark.unit
@pytest.mark.parametrize("command", ["run", "shift-run", "recommend"])
def test_param_edge_removed_commands_are_not_registered(command):
    with pytest.raises(SystemExit) as exc:
        main([command])
    assert exc.value.code == 2
