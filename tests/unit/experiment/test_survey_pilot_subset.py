# tests/unit/experiment/test_survey_pilot_subset.py

# ruff: noqa: N803, N806, F811, S101

"""Running one shard of the matrix, and being able to prove it was one shard.

The pilot is split across two hosts by dataset.  Nothing about that split is
self-enforcing: a driver that plans every unit regardless would have both hosts
train all 645 runs, and the duplicated work stays invisible until the disk bill
arrives.  So the filter has to narrow the *plan*, the estimate, the resume check
and the batches together -- narrowing only the report would look right and run
everything -- and the snapshot has to make the two shards checkable as a
partition rather than a claim.
"""

import json

import pytest
from _survey_pilot_helpers import (  # noqa: F401 - imported fixtures are used by name
    ALL_PRIORS,
    driver,
    splits_tree,
    unit_calls,
)

pytestmark = pytest.mark.unit


def _dry_run(driver, tmp_path, capsys, *extra: str) -> str:
    code = driver.main(
        [
            "--dry-run",
            "--results",
            str(tmp_path / "none"),
            "--splits",
            str(splits_tree(tmp_path)),
            *extra,
            *ALL_PRIORS,
        ]
    )
    assert code == 0
    return capsys.readouterr().out


def _snapshot(driver, tmp_path, *datasets: str) -> dict:
    path = tmp_path / "plan.json"
    argv = ["--dry-run", "--plan-json", str(path), "--results", str(tmp_path / "none")]
    argv += ["--splits", str(splits_tree(tmp_path))]
    if datasets:
        argv += ["--datasets", ",".join(datasets)]
    assert driver.main([*argv, *ALL_PRIORS]) == 0
    return json.loads(path.read_text(encoding="utf-8"))


def _keys(snapshot: dict) -> set[tuple]:
    return {
        (r["dataset"], r["method"], r["training_path"], r["mechanism"], r["c_token"], r["seed"])
        for r in snapshot["runs"]
    }


# --- the filter narrows the plan ---------------------------------------------


def test_basic_the_filter_narrows_the_plan_to_one_dataset(driver, tmp_path, capsys):
    printed = _dry_run(driver, tmp_path, capsys, "--datasets", "cifar10")

    assert "planned: 215 run(s)" in printed
    assert "pending:   215 (215 cifar10)" in printed
    # The disk figure is what a host is sized against, so it has to follow the
    # filter: sizing a shard against the whole matrix buys 300 GiB of nothing.
    assert "318.4 GiB for the whole pilot" in printed


def test_basic_without_the_flag_every_dataset_is_still_planned(driver, tmp_path, capsys):
    """The default stays the whole matrix: the flag narrows, it never widens."""
    printed = _dry_run(driver, tmp_path, capsys)

    assert "planned: 645 run(s)" in printed
    assert "324.5 GiB for the whole pilot" in printed


def test_param_the_filter_order_does_not_change_the_plan(driver, tmp_path, capsys):
    first = _dry_run(driver, tmp_path, capsys, "--datasets", "imdb,spambase")
    second = _dry_run(driver, tmp_path, capsys, "--datasets", "spambase,imdb")

    assert first == second


def test_basic_the_disk_estimate_follows_the_filter(driver, tmp_path, capsys):
    printed = _dry_run(driver, tmp_path, capsys, "--datasets", "imdb,spambase")

    assert "planned: 430 run(s)" in printed
    # 6.2, not the 5.3 + 0.8 the per-dataset lines show: those are each rounded
    # before printing, and the shard's own figure is the exact sum (6.1538...).
    assert "6.2 GiB for the whole pilot" in printed


# --- a mistyped name must not become an empty plan ---------------------------


def test_edge_an_unknown_dataset_name_is_refused_before_any_batch(
    driver, unit_calls, tmp_path, capsys
):
    """The failure this gate exists for: a typo plans nothing and exits 0.

    Both hosts would then believe the other one covered the shard.
    """
    code = driver.main(
        [
            "--results",
            str(tmp_path / "out"),
            "--splits",
            str(splits_tree(tmp_path)),
            "--datasets",
            "cifra10",
            *ALL_PRIORS,
        ]
    )

    assert code == 1
    assert unit_calls == []
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "cifra10" in err
    assert "cifar10" in err  # the names it would have accepted


def test_edge_an_empty_dataset_list_is_refused(driver, tmp_path, capsys):
    for value in ("", ",", " , "):
        code = driver.main(
            [
                "--dry-run",
                "--results",
                str(tmp_path),
                "--splits",
                str(splits_tree(tmp_path)),
                "--datasets",
                value,
            ]
        )
        assert code == 1, value
        assert capsys.readouterr().err.startswith("error: ")


# --- the snapshot makes the split auditable ----------------------------------


def test_basic_the_snapshot_records_the_runs_of_one_shard(driver, tmp_path):
    snapshot = _snapshot(driver, tmp_path, "cifar10")

    assert snapshot["datasets"] == ["cifar10"]
    assert snapshot["totals"]["planned"] == 215
    assert len(snapshot["runs"]) == 215
    assert {r["dataset"] for r in snapshot["runs"]} == {"cifar10"}
    assert all(
        {"method", "training_path", "mechanism", "c_token", "seed"} <= set(r)
        for r in snapshot["runs"]
    )


def test_determ_the_two_shards_partition_the_matrix(driver, tmp_path):
    """P2.1a / P2.1b: the union is the matrix and the halves do not overlap."""
    whole = _keys(_snapshot(driver, tmp_path))
    first = _keys(_snapshot(driver, tmp_path, "cifar10"))
    second = _keys(_snapshot(driver, tmp_path, "imdb", "spambase"))

    assert first & second == set()
    assert first | second == whole
    assert len(whole) == 645


def test_edge_the_snapshot_requires_a_dry_run(driver, unit_calls, tmp_path, capsys):
    path = tmp_path / "plan.json"
    code = driver.main(
        [
            "--plan-json",
            str(path),
            "--results",
            str(tmp_path / "out"),
            "--splits",
            str(splits_tree(tmp_path)),
            *ALL_PRIORS,
        ]
    )

    assert code == 1
    assert unit_calls == []
    assert not path.exists()
    assert capsys.readouterr().err.startswith("error: ")


# --- the filter has to reach execution, not just the report ------------------


def test_basic_the_filter_reaches_the_batches_not_only_the_report(driver, unit_calls, tmp_path):
    """A filter applied only where the plan is printed would run everything."""
    driver.main(
        [
            "--results",
            str(tmp_path / "out"),
            "--splits",
            str(splits_tree(tmp_path)),
            "--datasets",
            "spambase",
            *ALL_PRIORS,
        ]
    )

    assert len(unit_calls) == 19  # spambase's batches, not the matrix's 57


# --- a scoped count has to name its scope ------------------------------------


def test_basic_the_summary_declares_its_scope_when_the_plan_is_a_subset(
    driver, unit_calls, tmp_path, capsys
):
    """Two hosts each reporting "completed 215 of 215" reads as a finished matrix."""
    driver.main(
        [
            "--results",
            str(tmp_path / "out"),
            "--splits",
            str(splits_tree(tmp_path)),
            "--datasets",
            "spambase",
            *ALL_PRIORS,
        ]
    )

    assert "completed 0 of 215 run(s) in spambase" in capsys.readouterr().out


def test_basic_the_unfiltered_summary_is_unchanged(driver, unit_calls, tmp_path, capsys):
    """Without the flag the count has always meant the matrix, and still does."""
    driver.main(
        [
            "--results",
            str(tmp_path / "out"),
            "--splits",
            str(splits_tree(tmp_path)),
            *ALL_PRIORS,
        ]
    )

    out = capsys.readouterr().out
    assert "completed 0 of 645 run(s); 645 still pending" in out
    assert "run(s) in " not in out


def test_basic_a_dataset_only_request_says_nothing_more_than_it_always_did(
    driver, unit_calls, tmp_path, capsys
):
    """Narrowing by method and by path added a line, not a new spelling.

    A dataset's runs are all in scope when the dataset is what was asked for,
    so ``in spambase`` is a complete description and the scope line would only
    be noise -- and the hosts holding last week's command lines are still
    reading this output.
    """
    driver.main(
        [
            "--results",
            str(tmp_path / "out"),
            "--splits",
            str(splits_tree(tmp_path)),
            "--datasets",
            "spambase",
            *ALL_PRIORS,
        ]
    )

    out = capsys.readouterr().out
    assert "completed 0 of 215 run(s) in spambase; 215 still pending" in out
    assert "scope:" not in out


# --- a slice has to reach execution, and has to leave the protocol alone -----


def test_basic_the_scope_reaches_execution_without_rewriting_the_protocol(
    driver, unit_calls, tmp_path
):
    """The end-to-end property, asserted where it can fail silently.

    Three claims at once.  The methods a scope did not name never reach a
    subprocess -- a count of batches would not say so, since a filter that
    reached the report and not the loop runs the right *number* of the wrong
    batches.  The unit scripts are handed the protocol the request named, not a
    narrowed copy of it, because their manifests are bound to that file's
    digest and a copy would bind them to a protocol no reader has.  And the
    oracle batch carries no ``--method`` at all, so the parser here has to fall
    back to it rather than read a missing argument as an unnamed method.
    """
    driver.main(
        [
            "--results",
            str(tmp_path / "out"),
            "--splits",
            str(splits_tree(tmp_path)),
            "--datasets",
            "cifar10",
            "--methods",
            "self_pu,dist_pu",
            "--training-paths",
            "cnn_feature_adapter",
            *ALL_PRIORS,
        ]
    )

    assert len(unit_calls) == 6  # two methods, three mechanisms each
    methods = {
        call[call.index("--method") + 1] if "--method" in call else "pn_oracle"
        for call in unit_calls
    }
    assert methods == {"self_pu", "dist_pu"}
    assert {call[call.index("--training-path") + 1] for call in unit_calls} == {
        "cnn_feature_adapter"
    }
    for call in unit_calls:
        assert call[call.index("--protocol") + 1] == "survey-v1"


def test_param_a_name_that_covers_nothing_is_refused_before_any_batch(
    driver, unit_calls, tmp_path, capsys
):
    """Every name is in the matrix; one of them still runs nothing here.

    Two shapes of the same thing, and the second is the quiet one: when the
    whole intersection is empty the count would have been 0 and someone would
    notice, but when only *one* named value covers nothing the plan looks
    healthy and its own scope line claims the value was run.  Both have to be
    refused rather than planned, and the message has to name what would have
    worked -- ``pn_oracle`` beside the native CNN row and ``self_pu`` beside it
    both read as reasonable requests.
    """
    for methods, paths in (("pn_oracle", "native_cnn"), ("nnpu,self_pu", "native_cnn")):
        code = driver.main(
            [
                "--results",
                str(tmp_path / "out"),
                "--splits",
                str(splits_tree(tmp_path)),
                "--datasets",
                "cifar10",
                "--methods",
                methods,
                "--training-paths",
                paths,
                *ALL_PRIORS,
            ]
        )

        assert code == 1, methods
        assert unit_calls == []
        err = capsys.readouterr().err
        assert err.startswith("error: ")
    assert "available matching methods: ['nnpu']" in err
