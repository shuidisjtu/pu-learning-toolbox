# ruff: noqa: F811

"""The report identity block: which checkout and which input tree produced a report.

The P2.2 implementation plan requires every report to name its input result roots,
its code commit, the protocol digest, the comparison-matrix digest and a generation
time.  Three entry points produce reports and one module assembles the block, so
this file pins three things the readers of that block depend on:

* the block is **pure** apart from one function, because a determinism check
  compares two builds over one tree;
* a commit that could not be read is recorded as ``unavailable`` rather than left
  blank, because a blank reads as "nothing to record";
* the **Markdown carries no clock**, because a summary's Markdown and CSV are
  pinned to be byte-identical across two runs of the same input.

The entry-point wiring is pinned here too rather than in the CLI suite: that file
sits at 14 of the 15 test functions the quality gate allows, and its last slot is
worth more than this.  This file is now at that cap itself, so a further
provenance test belongs in a new file rather than appended here.
"""

import importlib
import json
import sys
import types

import pytest
from _survey_summary_helpers import (  # noqa: F401
    SCRIPTS_DIR,
    audit_cli,
    config_for,
    manifest,
    summary_cli,
    write_config,
    write_tree,
)

from pu_toolbox.experiment import survey_provenance
from pu_toolbox.experiment.survey_comparison import (
    comparison_digest,
    load_comparison_protocol,
)
from pu_toolbox.experiment.survey_provenance import (
    CODE_COMMIT_RECORDED,
    CODE_COMMIT_STATES,
    CODE_COMMIT_UNAVAILABLE,
    build_provenance,
    input_result_roots,
    recorded_source_roots,
    render_provenance_lines,
    resolve_code_commit,
    with_code_commit,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
attach = importlib.import_module("compare_survey_results")

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


def _five_seeds(**overrides):
    return {f"seed_{seed}": manifest(seed=seed, **overrides) for seed in range(5)}


def test_basic_the_block_records_every_identity_field():
    block = build_provenance(
        protocol_version="survey-v1.2",
        protocol_sha256="a" * 64,
        comparison_version="survey-comparison-v3",
        comparison_sha256="b" * 64,
        result_roots={"B2": "/roots/B2", "B1": "/roots/B1"},
        source_roots={"B1": "/remote/B1"},
        code_commit="c" * 40,
        code_commit_dirty=False,
    )

    assert block["protocol_version"] == "survey-v1.2"
    assert block["comparison_sha256"] == "b" * 64
    assert block["code_commit"] == "c" * 40
    assert block["code_commit_state"] == CODE_COMMIT_RECORDED
    # Sorted here rather than at render time, so the JSON and the Markdown cannot
    # disagree about the order, and two runs cannot either.
    assert list(block["input_result_roots"]) == ["B1", "B2"]


def test_basic_a_commit_that_was_not_read_is_recorded_as_unavailable():
    block = build_provenance(protocol_version="survey-v1.2", protocol_sha256="a" * 64)

    assert block["code_commit"] is None
    assert block["code_commit_state"] == CODE_COMMIT_UNAVAILABLE
    assert block["code_commit_state"] in CODE_COMMIT_STATES


def test_basic_the_input_roots_come_from_the_batch_config():
    assert input_result_roots(_config()) == {"B1": "/roots/B1", "B2": "/roots/B2"}
    # A batch the caller built by hand without a root is omitted, not invented.
    assert input_result_roots({"batches": [{"name": "B1"}, "not a batch"]}) == {}


def test_basic_the_source_roots_are_optional_and_recorded_when_present():
    assert recorded_source_roots(_config()) == {}
    config = _config() | {"source_roots": {"B1": "/remote/B1", "B2": "/remote/B2"}}
    assert recorded_source_roots(config) == {"B1": "/remote/B1", "B2": "/remote/B2"}
    # A non-mapping or a non-string value is a config defect, not a root.
    assert recorded_source_roots({"source_roots": ["/remote/B1"]}) == {}


def test_basic_the_entry_point_reports_name_their_inputs(audit_cli, summary_cli, tmp_path):
    tree = write_tree(tmp_path / "b1", _five_seeds())
    config_path = write_config(tmp_path / "roots.json", config_for({"B1": (tree, 5)}))
    out = tmp_path / "out"

    # The verdict is the CLI suite's subject, not this file's, and a five-seed
    # single-c stub is deliberately an incomplete tree.  Both entry points write
    # their report before deciding the exit code, which is what lets a reviewer
    # read a report whose verdict is not clean.
    audit_cli.main(["--config", str(config_path), "--out-dir", str(out / "a")])
    summary_cli.main(["--config", str(config_path), "--out-dir", str(out / "s")])

    for name, dirname in (("audit", "a"), ("summary", "s")):
        report = json.loads((out / dirname / f"{name}.json").read_text(encoding="utf-8"))
        block = report["provenance"]
        assert block["input_result_roots"] == {"B1": str(tree)}
        # Tolerant of a runner without git: what is pinned is that the question
        # was answered, not which answer this machine gives.
        assert block["code_commit_state"] in CODE_COMMIT_STATES
        # One source of truth: the block cannot disagree with the header.
        assert block["protocol_sha256"] == report["protocol_sha256"]

    # The Markdown a reviewer reads gains the identity, and still no clock.
    markdown = (out / "s" / "summary.md").read_text(encoding="utf-8")
    assert str(tree) in markdown
    assert "generated_at" not in markdown


def test_basic_the_entry_point_stamps_the_commit_it_resolved(
    audit_cli, summary_cli, monkeypatch, tmp_path
):
    """Pinned to a sentinel, so this holds on a runner with no git either.

    Without it the stamping could be deleted outright and every other test would
    still pass: the tolerant state check elsewhere accepts ``unavailable``, which
    is exactly what a dropped stamp produces.
    """
    tree = write_tree(tmp_path / "b1", _five_seeds())
    config_path = write_config(tmp_path / "roots.json", config_for({"B1": (tree, 5)}))
    out = tmp_path / "out"
    monkeypatch.setattr(audit_cli, "resolve_code_commit", lambda: ("f" * 40, True))
    monkeypatch.setattr(summary_cli, "resolve_code_commit", lambda: ("f" * 40, True))

    audit_cli.main(["--config", str(config_path), "--out-dir", str(out / "a")])
    summary_cli.main(["--config", str(config_path), "--out-dir", str(out / "s")])

    for name, dirname in (("audit", "a"), ("summary", "s")):
        report = json.loads((out / dirname / f"{name}.json").read_text(encoding="utf-8"))
        block = report["provenance"]
        assert block["code_commit"] == "f" * 40
        assert block["code_commit_state"] == CODE_COMMIT_RECORDED
        assert block["code_commit_dirty"] is True
        # The rendered report carries it too: an identity that exists only in the
        # JSON is not the one a reviewer reads.
        markdown = (out / dirname / f"{name}.md").read_text(encoding="utf-8")
        assert "f" * 40 in markdown
        assert "（工作区有未提交改动）" in markdown


def test_basic_the_attachment_carries_the_summarys_identity_forward(tmp_path):
    summary = {
        "schema_version": "survey-summary-1",
        "protocol_version": "survey-v1.2",
        "protocol_sha256": "a" * 64,
        "rows": [],
        "provenance": {
            "input_result_roots": {"B1": "/roots/B1"},
            "source_roots": {"B1": "/remote/B1"},
            "code_commit": "c" * 40,
            "code_commit_dirty": False,
        },
    }

    report = attach.build_comparison(summary, comparison=load_comparison_protocol())
    block = report["provenance"]

    # Carried, not re-derived: this entry point reads no whitelist and no git, so
    # re-deriving either here would let it disagree with the report it attached to.
    assert block["input_result_roots"] == {"B1": "/roots/B1"}
    assert block["code_commit"] == "c" * 40
    assert block["code_commit_state"] == CODE_COMMIT_RECORDED
    # The one field it does read for itself.
    assert block["comparison_sha256"] == comparison_digest(load_comparison_protocol())

    attach.write_report(report, tmp_path)
    markdown = (tmp_path / "comparison_summary.md").read_text(encoding="utf-8")
    assert "c" * 40 in markdown
    assert "/roots/B1" in markdown


def test_basic_a_summary_written_before_the_block_existed_is_not_padded():
    report = attach.build_comparison(
        {"protocol_version": "survey-v1.2", "protocol_sha256": "a" * 64, "rows": []},
        comparison=load_comparison_protocol(),
    )

    block = report["provenance"]
    assert block["input_result_roots"] == {}
    assert block["code_commit_state"] == CODE_COMMIT_UNAVAILABLE


@pytest.mark.parametrize(
    ("commit", "expected_state"),
    [(None, CODE_COMMIT_UNAVAILABLE), ("d" * 40, CODE_COMMIT_RECORDED)],
)
def test_param_stamping_records_the_commit_and_its_state(commit, expected_state):
    base = build_provenance(protocol_version="survey-v1.2", protocol_sha256="a" * 64)

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


def test_param_the_roots_render_in_batch_name_order():
    lines = render_provenance_lines(
        build_provenance(
            protocol_version="survey-v1.2",
            protocol_sha256="a" * 64,
            result_roots={"B3b": "/roots/3b", "B1": "/roots/1", "B3a": "/roots/3a"},
            code_commit="c" * 40,
        )
    )

    assert lines[1:] == [
        "- 输入结果根 `B1`：`/roots/1`",
        "- 输入结果根 `B3a`：`/roots/3a`",
        "- 输入结果根 `B3b`：`/roots/3b`",
    ]


def test_edge_a_git_failure_records_no_commit_and_says_so(monkeypatch, capsys):
    def boom(argv, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(survey_provenance.subprocess, "run", boom)

    # Not (None, False): "the question could not be asked" is a different answer
    # from "the work tree is clean", and a report that conflated them would claim
    # an identity it never read.
    assert resolve_code_commit() == (None, None)
    assert "warning:" in capsys.readouterr().err


def test_edge_a_missing_block_renders_nothing_and_a_rootless_one_renders_the_commit():
    assert render_provenance_lines(None) == []
    assert render_provenance_lines({}) == []

    only_commit = render_provenance_lines(
        build_provenance(
            protocol_version="survey-v1.2",
            protocol_sha256="a" * 64,
            code_commit="c" * 40,
            code_commit_dirty=True,
        )
    )
    assert only_commit == [f"- 代码 commit：`{'c' * 40}`（工作区有未提交改动）"]

    unavailable = render_provenance_lines(
        build_provenance(protocol_version="survey-v1.2", protocol_sha256="a" * 64)
    )
    assert unavailable == [
        f"- 代码 commit：不可用（`{CODE_COMMIT_UNAVAILABLE}`，不在 git 工作树内）"
    ]

    # A work tree whose cleanliness could not be read is not a clean one: the
    # Markdown must not print it the way it prints a clean tree.
    unknown = render_provenance_lines(
        build_provenance(
            protocol_version="survey-v1.2",
            protocol_sha256="a" * 64,
            code_commit="c" * 40,
            code_commit_dirty=None,
        )
    )
    assert unknown == [f"- 代码 commit：`{'c' * 40}`（工作区状态未能读取）"]


def test_determ_the_block_is_a_pure_function_of_its_arguments(monkeypatch):
    def boom(argv, **kwargs):
        raise AssertionError("build_provenance must not run a process")

    # Equality alone would not pin this: a git read answers the same twice.  So
    # the process call is made fatal, and the two builds below are what must
    # survive it.  A clock is already excluded by the equality check.
    monkeypatch.setattr(survey_provenance.subprocess, "run", boom)
    kwargs = {
        "protocol_version": "survey-v1.2",
        "protocol_sha256": "a" * 64,
        "result_roots": {"B2": "/roots/B2", "B1": "/roots/B1"},
    }

    assert build_provenance(**kwargs) == build_provenance(**kwargs)


def test_determ_the_rendering_carries_no_stamp():
    block = build_provenance(
        protocol_version="survey-v1.2",
        protocol_sha256="a" * 64,
        result_roots={"B1": "/roots/B1"},
        code_commit="c" * 40,
    )

    first, second = render_provenance_lines(block), render_provenance_lines(block)

    assert first == second
    # Stated structurally rather than by pattern-matching a rendered date: the
    # Markdown is a function of this block, and the block carries no time.
    assert not [key for key in block if "time" in key or "generated" in key]
