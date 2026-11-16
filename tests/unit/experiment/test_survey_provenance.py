# ruff: noqa: F811

"""The report identity block: which checkout and which input tree produced a report.

This file verifies identity assembly and explicit handling of unavailable Git
information:

* the block is **pure** apart from one function, because a determinism check
  compares two builds over one tree;
* a commit that could not be read is recorded as ``unavailable`` rather than left
  blank, because a blank reads as "nothing to record".
"""

import types

import pytest

from pu_toolbox.experiment import survey_provenance
from pu_toolbox.experiment.survey_provenance import (
    CODE_COMMIT_RECORDED,
    CODE_COMMIT_UNAVAILABLE,
    build_provenance,
    input_result_roots,
    resolve_code_commit,
    with_code_commit,
)

pytestmark = pytest.mark.unit


def _config():
    """A two-batch whitelist, deliberately out of name order."""
    return {
        "schema_version": "survey-batch-roots-1",
        "batches": [
            {"name": "B2", "root": "/roots/B2", "expected_manifests": 1},
            {"name": "B1", "root": "/roots/B1", "expected_manifests": 1},
        ],
    }


def test_basic_the_block_records_every_identity_field():
    block = build_provenance(
        protocol_version="survey-v3.0",
        protocol_sha256="a" * 64,
        comparison_version="survey-comparison-v3",
        comparison_sha256="b" * 64,
        result_roots={"B2": "/roots/B2", "B1": "/roots/B1"},
        source_roots={"B1": "/remote/B1"},
        code_commit="c" * 40,
        code_commit_dirty=False,
    )

    assert block["protocol_version"] == "survey-v3.0"
    assert block["comparison_sha256"] == "b" * 64
    assert block["code_commit"] == "c" * 40
    assert block["code_commit_state"] == CODE_COMMIT_RECORDED
    # Sorting fixes the recorded order across repeated builds.
    assert list(block["input_result_roots"]) == ["B1", "B2"]


def test_basic_a_commit_that_was_not_read_is_recorded_as_unavailable():
    block = build_provenance(protocol_version="survey-v3.0", protocol_sha256="a" * 64)

    assert block["code_commit"] is None
    assert block["code_commit_state"] == CODE_COMMIT_UNAVAILABLE


def test_basic_the_input_roots_come_from_the_batch_config():
    assert input_result_roots(_config()) == {"B1": "/roots/B1", "B2": "/roots/B2"}
    # A batch the caller built by hand without a root is omitted, not invented.
    assert input_result_roots({"batches": [{"name": "B1"}, "not a batch"]}) == {}


@pytest.mark.parametrize(
    ("commit", "expected_state"),
    [(None, CODE_COMMIT_UNAVAILABLE), ("d" * 40, CODE_COMMIT_RECORDED)],
)
def test_param_stamping_records_the_commit_and_its_state(commit, expected_state):
    base = build_provenance(protocol_version="survey-v3.0", protocol_sha256="a" * 64)

    stamped = with_code_commit(base, commit, dirty=False)

    assert stamped["code_commit_state"] == expected_state
    # A copy: the block the pure builder produced is left alone, so a caller that
    # stamps twice cannot reach back into the report it already wrote.
    assert base["code_commit"] is None
    assert base["code_commit_state"] == CODE_COMMIT_UNAVAILABLE


@pytest.mark.parametrize(("porcelain", "expected"), [("", False), (" M a.py\n", True)])
def test_param_the_dirty_flag_follows_git_status(monkeypatch, porcelain, expected):
    def fake(argv, **kwargs):
        return types.SimpleNamespace(stdout="abc123\n" if "rev-parse" in argv else porcelain)

    monkeypatch.setattr(survey_provenance.subprocess, "run", fake)

    assert resolve_code_commit() == ("abc123", expected)


def test_edge_a_git_failure_records_no_commit_and_says_so(monkeypatch, capsys):
    def boom(argv, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(survey_provenance.subprocess, "run", boom)

    # Not (None, False): "the question could not be asked" is a different answer
    # from "the work tree is clean", and a report that conflated them would claim
    # an identity it never read.
    assert resolve_code_commit() == (None, None)
    assert "warning:" in capsys.readouterr().err


def test_determ_the_block_is_a_pure_function_of_its_arguments(monkeypatch):
    def boom(argv, **kwargs):
        raise AssertionError("build_provenance must not run a process")

    # Equality alone would not pin this: a git read answers the same twice.  So
    # the process call is made fatal, and the two builds below are what must
    # survive it.  A clock is already excluded by the equality check.
    monkeypatch.setattr(survey_provenance.subprocess, "run", boom)
    kwargs = {
        "protocol_version": "survey-v3.0",
        "protocol_sha256": "a" * 64,
        "result_roots": {"B2": "/roots/B2", "B1": "/roots/B1"},
    }

    assert build_provenance(**kwargs) == build_provenance(**kwargs)
