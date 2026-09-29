# tests/unit/experiment/test_survey_pilot_scope_wiring.py

# ruff: noqa: N803, N806, F811, S101

"""A scope has to reach every reader of the matrix, not just the printed plan.

The failure this file is written against is silent and expensive: a driver that
narrows what it *reports* and not what it *reads* prints one shard, sizes the
host for one shard, resumes against one shard, and then trains the matrix.  So
each assertion here names the reader it holds to the scope -- the plan, the
checkpoint estimate, the split reads, the resume check, and the snapshot that
records what the request asked for.

The snapshot is also where protocol identity is kept honest.  The runs a slice
executes write manifests bound to the digest of the *whole* frozen matrix, so
the snapshot records that same digest: a slice can shrink what runs without
inventing a protocol its own runs would disagree with.
"""

import json
from pathlib import Path

import pytest
from _survey_pilot_helpers import (  # noqa: F401 - imported fixtures are used by name
    ALL_PRIORS,
    driver,
    splits_tree,
    unit_calls,
)

from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATH, digest, load_protocol

pytestmark = pytest.mark.unit

#: B3b: one dataset, two methods, one training path -- three axes that a
#: dataset list alone cannot express, since CIFAR-10's other four runnable
#: units are the same dataset and different shards.
B3B = [
    "--datasets",
    "cifar10",
    "--methods",
    "self_pu,dist_pu",
    "--training-paths",
    "cnn_feature_adapter",
]


def _dry_run(driver, tmp_path, capsys, *extra: str, results: Path | None = None) -> str:
    code = driver.main(
        [
            "--dry-run",
            "--results",
            str(results if results is not None else tmp_path / "none"),
            "--splits",
            str(splits_tree(tmp_path)),
            *extra,
            *ALL_PRIORS,
        ]
    )
    assert code == 0
    return capsys.readouterr().out


def _snapshot(driver, tmp_path, *extra: str, name: str = "plan.json") -> dict:
    path = tmp_path / name
    argv = ["--dry-run", "--plan-json", str(path), "--results", str(tmp_path / "none")]
    argv += ["--splits", str(splits_tree(tmp_path)), *extra]
    assert driver.main([*argv, *ALL_PRIORS]) == 0
    return json.loads(path.read_text(encoding="utf-8"))


def _finished_run(
    results: Path,
    *,
    dataset: str,
    method: str,
    training_path: str,
    run_view: str,
) -> Path:
    """A finished run's manifest, recorded against the split ``splits_tree`` writes.

    Only the payload matters: the resume scan reads manifests, not paths, so a
    run is placed by what it says rather than by where it sits.
    """
    path = results / dataset / method / "c_0.1" / "seed_0" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "execution_mode": "versioned_pilot",
        "seed": 0,
        "execution_unit": {
            "dataset": dataset,
            "method": method,
            "training_path": training_path,
        },
        "selection": {"OA": {"candidate_index": 0}},
        "run_view": run_view,
        "calibration_applied": run_view == "ts-compatible",
        "representation": {"split_sha256": "a" * 64},
    }
    if method != "pn_oracle":
        payload["generation"] = {"train": {"mechanism": "scar"}}
        payload["c_requested_token"] = "0.1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- the plan, the estimate and the reads all narrow together ----------------


def test_basic_the_plan_is_the_selection_and_not_the_dataset(driver, tmp_path, capsys):
    """CIFAR-10 is 215 runs; this shard of it is 70, and says which 70."""
    printed = _dry_run(driver, tmp_path, capsys, *B3B)

    assert "planned: 70 run(s)" in printed
    assert "pending:   70 (70 cifar10)" in printed
    assert (
        "scope: datasets=cifar10; methods=dist_pu,self_pu; "
        "training_paths=cnn_feature_adapter" in printed
    )


def test_basic_the_checkpoint_estimate_follows_the_selection(driver, tmp_path, capsys):
    """The figure a host is sized against is the shard's, not the matrix's.

    ``upu`` is closed-form and keeps no per-epoch state, so a slice of it has
    to price at nothing at all -- and the run count beside that figure is the
    slice's 105, not the matrix's 645 nor any single dataset's.
    """
    closed_form = _dry_run(driver, tmp_path, capsys, "--methods", "upu")

    assert "0 of 105 run(s) that write any" in closed_form
    assert "0.0 GiB for the whole pilot" in closed_form

    assert "10.3 GiB for the whole pilot" in _dry_run(driver, tmp_path, capsys, *B3B)
    # The dataset alone would have bought 308 GiB of native CNN checkpoints
    # this shard never writes.
    assert "318.4 GiB for the whole pilot" in _dry_run(
        driver, tmp_path, capsys, "--datasets", "cifar10"
    )


def test_basic_the_split_reads_are_limited_to_the_selected_datasets(driver, tmp_path, monkeypatch):
    """A shard reads its own splits, and the scope is what decides which are its own.

    Read off the arguments rather than off the result: scoping the *walk* is
    the property, and a read that opened every dataset's manifests and then
    discarded the rest would still be stopped by a defect in one of them.
    """
    seen: list[tuple[str, tuple[str, ...] | None]] = []
    for name in ("split_digests", "population_priors"):
        real = getattr(driver, name)

        def _record(root, *, datasets=None, _name=name, _real=real):
            seen.append((_name, None if datasets is None else tuple(datasets)))
            return _real(root, datasets=datasets)

        monkeypatch.setattr(driver, name, _record)

    argv = [
        "--dry-run",
        "--results",
        str(tmp_path / "none"),
        "--splits",
        str(splits_tree(tmp_path)),
    ]
    driver.main([*argv, *B3B, *ALL_PRIORS])
    assert seen == [("split_digests", ("cifar10",)), ("population_priors", ("cifar10",))]

    # The method axis narrows runs, never datasets: a slice that names no
    # dataset still needs the splits of every dataset whose runs it covers.
    seen.clear()
    driver.main([*argv, "--methods", "nnpu", *ALL_PRIORS])
    assert seen == [
        ("split_digests", ("cifar10", "imdb", "spambase")),
        ("population_priors", ("cifar10", "imdb", "spambase")),
    ]


def test_basic_results_recorded_outside_the_slice_are_not_counted_as_done(driver, tmp_path, capsys):
    """A finished run elsewhere is not evidence about this shard.

    Scoping the *plan* rather than the results walk is what makes this true:
    the resume scan reads every manifest under the root and keeps the ones the
    plan asked for, so narrowing the walk instead would change resume semantics
    rather than tighten a report.
    """
    results = tmp_path / "out"
    _finished_run(
        results,
        dataset="spambase",
        method="nnpu",
        training_path="native_2d",
        run_view="ts-compatible",
    )

    outside = _dry_run(driver, tmp_path, capsys, *B3B, results=results)
    assert "completed: 0 (none)" in outside
    assert "pending:   70" in outside

    # The same record placed inside the slice is counted, so the zero above is
    # the scan placing the run and not the scan finding nothing at all.
    _finished_run(
        results,
        dataset="cifar10",
        method="self_pu",
        training_path="cnn_feature_adapter",
        run_view="ts-compatible",
    )
    inside = _dry_run(driver, tmp_path, capsys, *B3B, results=results)
    assert "pending:   69" in inside


def test_basic_the_snapshot_records_the_digest_of_the_whole_frozen_protocol(driver, tmp_path):
    """The plan's identity is the matrix's, and the manifest's, not the slice's.

    Hashed with the function the unit script hashes the file with, so this is
    an equality a reader can check against a real manifest rather than against
    a second implementation of the same idea.
    """
    snapshot = _snapshot(driver, tmp_path, *B3B)

    assert snapshot["source_protocol_sha256"] == digest(load_protocol(PROTOCOL_PATH))


def test_basic_the_snapshot_records_the_selection_and_its_runnable_units(driver, tmp_path):
    """What was asked for, and what will actually run, are two different records.

    ``selection`` is the request; ``execution_units`` is the request resolved
    against the matrix.  The second one lists units that run, so a slice of a
    matrix that carries exclusions cannot claim a unit the matrix never runs.
    """
    snapshot = _snapshot(driver, tmp_path, *B3B)

    assert snapshot["snapshot_schema_version"] == "1.0"
    assert snapshot["protocol_version"] == load_protocol(PROTOCOL_PATH)["protocol_version"]
    assert snapshot["selection"] == {
        "datasets": ["cifar10"],
        "methods": ["dist_pu", "self_pu"],
        "training_paths": ["cnn_feature_adapter"],
    }
    assert snapshot["execution_units"] == [
        {"dataset": "cifar10", "method": "dist_pu", "training_path": "cnn_feature_adapter"},
        {"dataset": "cifar10", "method": "self_pu", "training_path": "cnn_feature_adapter"},
    ]
    assert snapshot["datasets"] == ["cifar10"]
    assert snapshot["totals"] == {"planned": 70, "completed": 0, "pending": 70}
    assert len(snapshot["runs"]) == 70


# --- what a scope changes besides the plan -----------------------------------


def test_edge_a_result_under_the_other_view_stays_pending(driver, tmp_path, capsys):
    """Resuming still tells the views apart once the matrix has been sliced.

    ``self_pu`` runs the calibrated view by default, so an OS result of the
    same unit is not this shard's completion -- and a scope is exactly the kind
    of change that would tempt a driver to fold the two together.
    """
    results = tmp_path / "out"
    _finished_run(
        results,
        dataset="cifar10",
        method="self_pu",
        training_path="cnn_feature_adapter",
        run_view="os-compatible",
    )

    printed = _dry_run(driver, tmp_path, capsys, *B3B, results=results)

    assert "completed: 0 (none)" in printed
    assert "pending:   70" in printed


def test_edge_an_explicit_ts_is_accepted_once_the_scope_excludes_os_native_methods(
    driver, unit_calls, tmp_path, capsys
):
    """The explicit view is judged on the plan, and a scope is what the plan is.

    The refusal exists because ``lbe`` and the oracle cannot carry TS
    calibration, and on the whole matrix it fires.  A shard that contains
    neither of them is a plan for which the request is legal -- which is a
    consequence of filtering rather than a rule of its own, so it is pinned
    here next to the refusal it depends on.
    """
    argv = ["--results", str(tmp_path / "out"), "--splits", str(splits_tree(tmp_path))]

    refused = driver.main([*argv, "--datasets", "cifar10", "--os-or-ts", "ts", *ALL_PRIORS])
    assert refused == 1
    assert unit_calls == []
    assert "lbe" in capsys.readouterr().err  # named, not refused in general

    driver.main([*argv, *B3B, "--os-or-ts", "ts", *ALL_PRIORS])
    assert unit_calls
    assert all(call[call.index("--os-or-ts") + 1] == "ts" for call in unit_calls)


def test_edge_a_custom_protocol_is_narrowed_and_digested_as_itself(driver, tmp_path):
    """The matrix a request named is the one it is sliced from and vouched for.

    A digest taken from the shipped matrix would claim every run of this shard
    was bound to a protocol it never loaded.
    """
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["execution_units"] = [
        unit for unit in payload["execution_units"] if unit["dataset"] == "cifar10"
    ]
    custom = tmp_path / "custom_protocol.json"
    custom.write_text(json.dumps(payload), encoding="utf-8")

    snapshot = _snapshot(driver, tmp_path, "--protocol", str(custom), *B3B)

    assert snapshot["source_protocol_sha256"] == digest(load_protocol(custom))
    assert snapshot["source_protocol_sha256"] != digest(load_protocol(PROTOCOL_PATH))
    assert snapshot["totals"]["planned"] == 70


# --- the snapshot stays comparable, and stays an output ----------------------


def test_param_the_snapshot_is_still_refused_without_a_dry_run(
    driver, unit_calls, tmp_path, capsys
):
    path = tmp_path / "plan.json"
    code = driver.main(
        [
            "--plan-json",
            str(path),
            "--results",
            str(tmp_path / "out"),
            "--splits",
            str(splits_tree(tmp_path)),
            *B3B,
            *ALL_PRIORS,
        ]
    )

    assert code == 1
    assert unit_calls == []
    assert not path.exists()
    assert capsys.readouterr().err.startswith("error: ")


def test_param_a_value_named_twice_is_recorded_once(driver, tmp_path):
    """An axis names a set, so the record of it is a set's record.

    Deduplication is only observable where the request is written down: the
    narrowing works off membership, so a repeated value changes no run.  That
    is exactly why it is pinned here rather than on the selector -- the snapshot
    is compared against another host's, and a value listed twice in one file
    and once in the other is a difference that is not a shard.
    """
    repeated = _snapshot(
        driver,
        tmp_path,
        "--datasets",
        "cifar10",
        "--methods",
        "self_pu,dist_pu,self_pu",
        "--training-paths",
        "cnn_feature_adapter",
        name="repeated.json",
    )
    once = _snapshot(driver, tmp_path, *B3B, name="once.json")

    assert repeated["selection"]["methods"] == ["dist_pu", "self_pu"]
    assert repeated == once
    assert (tmp_path / "repeated.json").read_bytes() == (tmp_path / "once.json").read_bytes()


def test_determ_the_same_slice_written_in_another_order_writes_the_same_snapshot(driver, tmp_path):
    """Which order an axis was typed in is not part of the shard.

    Two hosts are handed the same list in whatever order each was told, and
    those files exist to be compared; a difference in them that is not a run
    would make the comparison say the shards disagree when they do not.
    """
    first = _snapshot(driver, tmp_path, *B3B, name="first.json")
    reordered = _snapshot(
        driver,
        tmp_path,
        "--datasets",
        "cifar10",
        "--methods",
        "dist_pu,self_pu",
        "--training-paths",
        "cnn_feature_adapter",
        name="second.json",
    )

    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()
    assert first == reordered


def test_determ_two_slices_differ_in_their_selection_but_not_in_the_protocol(driver, tmp_path):
    """Slicing changes what runs, never what the runs are bound to.

    This is the invariant the whole change rests on: if a slice moved the
    digest, its manifests would carry a protocol no reader of the plan has, and
    the aggregation gate would refuse the batch it was told to trust.
    """
    b3b = _snapshot(driver, tmp_path, *B3B, name="b3b.json")
    b4 = _snapshot(
        driver,
        tmp_path,
        "--datasets",
        "cifar10",
        "--methods",
        "nnpu",
        "--training-paths",
        "native_cnn",
        name="b4.json",
    )

    assert b3b["selection"] != b4["selection"]
    assert b3b["source_protocol_sha256"] == b4["source_protocol_sha256"]
    assert b3b["source_protocol_sha256"] == digest(load_protocol(PROTOCOL_PATH))


def test_determ_an_unfiltered_request_records_no_selection(driver, tmp_path):
    """Naming no axis is recorded as naming none, not as naming everything.

    The two are different requests -- an empty list is refused, and the plan is
    the matrix either way -- so the record has to show which one was made.
    """
    snapshot = _snapshot(driver, tmp_path)

    assert snapshot["selection"] == {
        "datasets": None,
        "methods": None,
        "training_paths": None,
    }
    assert snapshot["datasets"] == ["cifar10", "imdb", "spambase"]
    assert snapshot["totals"]["planned"] == 645
