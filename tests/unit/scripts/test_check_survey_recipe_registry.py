"""Failure paths of the recipe-registry gate's ``main()``.

R5 of the stage-7 rulings: this gate stays out of CI while the registry is
unmaterialised -- an absent file is *reported* as "nothing was validated"
and exits 0 -- so wiring it in would be an always-passing check.  The
compensation is that its failure paths are pinned here instead: a payload
that is not a registry, or one the schema rejects, must make ``main()``
non-zero.

Separate from ``tests/unit/experiment/test_survey_recipe_registry*.py``,
which cover the registry module's pure validators; this file covers the
script's exit codes and its absent-file branch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_survey_recipe_registry as gate  # noqa: E402

pytestmark = pytest.mark.unit


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    """Run the gate's ``main`` and return (exit code, stdout, stderr)."""
    code = gate.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_basic_absent_registry_declares_nothing_was_validated(tmp_path, capsys):
    code, out, _ = _run(capsys, "--registry", str(tmp_path / "absent.json"))

    assert code == 0
    assert "Nothing was validated" in out


def test_param_malformed_json_fails_main(tmp_path, capsys):
    payload = tmp_path / "registry.json"
    payload.write_text("{not json", encoding="utf-8")

    code, out, _ = _run(capsys, "--registry", str(payload))

    assert code == 1
    assert "Registry check failed" in out


def test_param_payload_without_profiles_fails_main(tmp_path, capsys):
    payload = tmp_path / "registry.json"
    payload.write_text(json.dumps({"recipe_registry_version": "not-a-registry"}), encoding="utf-8")

    code, out, _ = _run(capsys, "--registry", str(payload))

    assert code == 1
    assert "Registry check failed" in out


def test_edge_unknown_schema_version_fails_main(tmp_path, capsys):
    payload = tmp_path / "registry.json"
    payload.write_text(
        json.dumps({"schema_version": "not-a-registry-version", "profiles": {}}),
        encoding="utf-8",
    )

    code, out, _ = _run(capsys, "--registry", str(payload))

    assert code == 1
    assert "unsupported_schema_version" in out


def test_determ_repeated_runs_report_identically(tmp_path, capsys):
    payload = tmp_path / "registry.json"
    payload.write_text(json.dumps({"profiles": {}}), encoding="utf-8")

    first = _run(capsys, "--registry", str(payload))
    second = _run(capsys, "--registry", str(payload))

    assert first == second
