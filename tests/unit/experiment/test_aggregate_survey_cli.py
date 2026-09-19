# ruff: noqa: N803, N806, F811
"""The aggregation entry point's command line.

Report assembly is covered by ``test_aggregate_survey_runs.py``; what the CLI
adds is the exit code and what a caller is told, so that is what these pin.
"""

import json

import pytest
from _aggregate_script_helpers import aggregate_script, manifest, write_tree  # noqa: F401

pytestmark = pytest.mark.unit


def test_basic_cli_refuses_a_blocked_pilot_and_explains(aggregate_script, tmp_path, capsys):
    """Blocked is a first-class outcome, not a crash and not a silent pass."""
    root = write_tree(
        tmp_path,
        {"a": manifest(method="upu", formal_eligible=False, blockers=["P2.0c_acceptance"])},
    )
    assert aggregate_script.main([str(root)]) == 1
    printed = capsys.readouterr().out
    assert "P2.0c_acceptance" in printed

    assert aggregate_script.main([str(root), "--diagnostic"]) == 0
    diagnostic = capsys.readouterr().out
    assert "NON-FORMAL" in diagnostic


def test_param_cli_rejects_a_missing_directory(aggregate_script, tmp_path, capsys):
    assert aggregate_script.main([str(tmp_path / "nope")]) == 1
    assert "error:" in capsys.readouterr().err


def test_edge_cli_reports_an_empty_tree_as_an_error(aggregate_script, tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert aggregate_script.main([str(empty)]) == 1
    assert "no manifests" in capsys.readouterr().err


def test_param_script_is_reachable_as_json(aggregate_script, tmp_path, capsys):
    root = write_tree(tmp_path, {"a": manifest(method="upu")})
    assert aggregate_script.main([str(root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["groups"][0]["dataset"] == "spambase"
