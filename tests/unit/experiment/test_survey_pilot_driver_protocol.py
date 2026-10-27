# tests/unit/experiment/test_survey_pilot_driver_protocol.py

# ruff: noqa: F811, S101

"""Which protocol the pilot driver plans against when it is not told.

The driver defaulted to ``survey-v2.0``, the 645-run pilot's matrix, so a caller
sizing or launching the formal batch got the historical plan unless they
remembered ``--protocol``.  The protocol the project now runs under is
``survey-v3.0``; these pin that the plan says so without being asked, that an
explicit choice still wins, and that a bad protocol is an error rather than a
quiet fall-back to either.
"""

import json

import pytest
from _survey_pilot_helpers import raw_driver, splits_tree  # noqa: F401

pytestmark = pytest.mark.unit

# One method on one dataset: v3 also holds clean-support methods, which need the
# five-role splits this four-role tree does not have, and the point here is only
# which protocol the plan is made under.
_SCOPE = ["--datasets", "spambase", "--methods", "nnpu", "--class-prior", "spambase=0.39"]


def _plan(raw_driver, tmp_path, *extra):
    plan_path = tmp_path / "plan.json"
    code = raw_driver.main(
        [
            "--dry-run",
            "--results",
            str(tmp_path / "none"),
            "--splits",
            str(splits_tree(tmp_path)),
            "--plan-json",
            str(plan_path),
            *_SCOPE,
            *extra,
        ]
    )
    return code, json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None


def test_basic_the_plan_is_made_under_the_current_protocol_by_default(raw_driver, tmp_path):
    code, plan = _plan(raw_driver, tmp_path)

    assert code == 0
    assert plan["protocol_version"] == "survey-v3.0"


def test_param_an_explicit_protocol_is_honoured(raw_driver, tmp_path):
    code, plan = _plan(raw_driver, tmp_path, "--protocol", "survey-v2.0")

    assert code == 0
    assert plan["protocol_version"] == "survey-v2.0"


def test_edge_an_unknown_protocol_file_is_an_error_not_a_silent_default(
    raw_driver, tmp_path, capsys
):
    code, plan = _plan(raw_driver, tmp_path, "--protocol", str(tmp_path / "nope.json"))

    assert code != 0
    assert plan is None
    assert "error" in capsys.readouterr().err.lower()


def test_determ_the_default_plan_equals_the_one_that_names_v3(raw_driver, tmp_path):
    _, default = _plan(raw_driver, tmp_path / "d")
    _, named = _plan(raw_driver, tmp_path / "n", "--protocol", "survey-v3.0")

    assert default["source_protocol_sha256"] == named["source_protocol_sha256"]
    assert default["runs"] == named["runs"]
