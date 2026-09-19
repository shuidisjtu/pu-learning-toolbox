# tests/unit/experiment/test_pilot_plan.py

# ruff: noqa: N803, N806, S101

"""What the driver plans, and what it refuses to count as already done.

The synthetic matrix keeps the arithmetic checkable by hand; the real protocol
is used where the point is that the shipped matrix is expanded correctly.  The
recorded-manifest cases matter most: a driver that miscounts completion either
re-runs paid-for work or, far worse, skips a run it never did.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from pu_toolbox.experiment.pilot_plan import (
    PN_ORACLE,
    PilotRun,
    batch_command,
    batches,
    completed_runs,
    estimate_checkpoint_bytes,
    manifest_identity,
    pending_runs,
    planned_runs,
    population_priors,
)
from pu_toolbox.experiment.survey_protocol import load_protocol

pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"

_MLP128 = {"hidden_dims": [128], "activation": "relu"}


def _protocol() -> dict:
    """A four-unit matrix: two runnable counting units, an oracle, a refusal."""
    return {
        "seeds": [0, 1],
        "c_tokens": {"scar": ["0.1", "0.3"], "sar_lbe_a": ["0.05"]},
        "candidate_pool": [{}],
        "budgets": {"closed_form": {}, "minibatch": {"epochs": 200}},
        "backbone_specs": {"mlp128": _MLP128},
        "execution_units": [
            {
                "dataset": "spambase",
                "method": "upu",
                "training_path": "native_2d",
                "budget": "closed_form",
                "backbone": "none",
                "runnable": True,
            },
            {
                "dataset": "spambase",
                "method": "nnpu",
                "training_path": "native_2d",
                "budget": "minibatch",
                "backbone": "mlp128",
                "runnable": True,
            },
            {
                "dataset": "spambase",
                "method": PN_ORACLE,
                "training_path": "native_2d",
                "budget": "minibatch",
                "backbone": "mlp128",
                "runnable": True,
            },
            {
                "dataset": "spambase",
                "method": "kldce",
                "training_path": "native_2d",
                "budget": "closed_form",
                "backbone": "none",
                "runnable": False,
                "non_runnable_reason": "requires flip_probability",
            },
        ],
    }


def _manifest(dataset, method, training_path, seed, **overrides) -> dict:
    payload = {
        "execution_mode": "versioned_pilot",
        "seed": seed,
        "execution_unit": {
            "dataset": dataset,
            "method": method,
            "training_path": training_path,
        },
        "generation": {"train": {"mechanism": "scar"}},
        "c_requested_token": "0.1",
        # What a finished run has and a failed one does not: the runner writes
        # an empty selection when every candidate was excluded, still under the
        # completed execution mode.
        "selection": {"OA": {"candidate_index": 0}},
    }
    payload.update(overrides)
    return payload


def _write(root: Path, relative: str, payload: dict) -> Path:
    path = root / relative / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- what the protocol asks for ----------------------------------------------


def test_basic_planned_runs_expand_only_the_runnable_units():
    runs = planned_runs(_protocol())
    # Two counting units across 2 seeds and 3 c tokens, plus the oracle once
    # per seed: an oracle is defined by a clean view and has no c to vary.
    assert len(runs) == 2 * 2 * 3 + 2
    assert len([run for run in runs if run.is_oracle]) == 2
    assert all(run.method != "kldce" for run in runs)
    assert {run.c_token for run in runs if run.is_oracle} == {None}

    shipped = planned_runs(load_protocol())
    assert len(shipped) == 645
    assert len([run for run in shipped if run.is_oracle]) == 15


def test_determ_planning_the_same_matrix_twice_gives_the_same_order():
    """The driver's batch order has to be reproducible, not merely stable."""
    first = [run.key for run in planned_runs(_protocol())]
    second = [run.key for run in planned_runs(_protocol())]
    assert first == second


# --- what counts as done -----------------------------------------------------


def test_basic_only_a_selected_run_counts_and_everything_else_does_not(tmp_path):
    """The five ways a manifest fails to be a finished run.

    The third is the one worth stating: a run whose candidates were *all*
    excluded -- every attempt failed -- also writes the completed execution
    mode, with an empty selection beside its failures.  Reading the mode alone
    would file that as done and never retry it.  A counting row is saved by the
    c token it never gets; an oracle needs no c token at all, so the hole would
    land exactly on the oracle rows.
    """
    _write(tmp_path, "good", _manifest("spambase", "upu", "native_2d", 0))
    _write(
        tmp_path,
        "refused",
        _manifest("spambase", "upu", "native_2d", 1, execution_mode="rejected_versioned_pilot"),
    )
    excluded = _manifest("spambase", PN_ORACLE, "native_2d", 0)
    del excluded["c_requested_token"]
    _write(
        tmp_path,
        "excluded",
        {**excluded, "selection": {}, "test_results": {}, "failures": [{"stage": "training"}]},
    )
    without_token = _manifest("spambase", "upu", "native_2d", 1)
    del without_token["c_requested_token"]
    _write(tmp_path, "no-token", without_token)
    _write(
        tmp_path,
        "smoke",
        _manifest("spambase", "upu", "native_2d", 2, execution_mode="technical_smoke"),
    )
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken/manifest.json").write_text("{not json", encoding="utf-8")

    assert set(completed_runs(tmp_path)) == {("spambase", "upu", "native_2d", "scar", "0.1", 0)}


def test_edge_a_run_from_a_rebuilt_split_is_not_done(tmp_path):
    """A run is evidence about the data it ran on, and only that data.

    P1.2/P1.4 rebuilt every split on 2026-09-19.  A manifest from before that
    describes splits the pilot no longer has; counting it as done would report
    the pilot complete while holding results that belong to no split on disk.
    """
    seed0, seed1 = (
        _manifest("spambase", "upu", "native_2d", 0),
        _manifest("spambase", "upu", "native_2d", 1),
    )
    seed0["representation"] = {"split_sha256": "old"}
    seed1["representation"] = {"split_sha256": "new"}
    _write(tmp_path, "seed0", seed0)
    _write(tmp_path, "seed1", seed1)
    key0 = ("spambase", "upu", "native_2d", "scar", "0.1", 0)
    key1 = ("spambase", "upu", "native_2d", "scar", "0.1", 1)

    # Matching digests: both runs still describe the splits on disk.
    both = {("spambase", 0): "old", ("spambase", 1): "new"}
    assert set(completed_runs(tmp_path, splits=both)) == {key0, key1}
    # Seed 0's split has been rebuilt since: that run is about other data.
    rebuilt = {("spambase", 0): "rebuilt", ("spambase", 1): "new"}
    assert set(completed_runs(tmp_path, splits=rebuilt)) == {key1}
    # A split the mapping does not cover cannot be confirmed, so it is not done.
    partial = {("spambase", 1): "new"}
    assert set(completed_runs(tmp_path, splits=partial)) == {key1}


def test_edge_splits_that_disagree_on_the_population_prior_are_refused(tmp_path):
    """Protocol §3.1 makes it one constant per dataset, not a per-seed choice."""

    def _split(seed: int, prior: float) -> None:
        path = tmp_path / "spambase" / f"split_{seed}" / "split_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"dataset": "spambase", "seed": seed, "class_prior": {"population": prior}}),
            encoding="utf-8",
        )

    _split(0, 0.4)
    assert population_priors(tmp_path) == {"spambase": 0.4}

    _split(1, 0.45)
    with pytest.raises(ValueError, match="declares two population priors"):
        population_priors(tmp_path)


def test_basic_pending_is_the_plan_minus_what_the_records_place(tmp_path):
    protocol = _protocol()
    _write(tmp_path, "a", _manifest("spambase", "upu", "native_2d", 0))
    _write(tmp_path, "b", _manifest("spambase", PN_ORACLE, "native_2d", 1))
    pending, done = pending_runs(protocol, tmp_path)
    assert len(pending) + len(done) == len(planned_runs(protocol))
    assert len(done) == 2
    assert ("spambase", "upu", "native_2d", "scar", "0.1", 0) not in {run.key for run in pending}


# --- batching ----------------------------------------------------------------


def test_basic_a_complete_grid_is_one_batch_carrying_exactly_it():
    runs = tuple(
        PilotRun("spambase", "nnpu", "native_2d", "scar", token, seed)
        for seed in (0, 1)
        for token in ("0.1", "0.3")
    )
    batched = batches(runs)
    assert len(batched) == 1
    assert set(batched[0].covers) == {run.key for run in runs}
    command = batch_command(batched[0], splits_root="data/splits", out_root="results/survey")
    assert command[0] == str(Path("data/splits/spambase"))
    assert command[command.index("--seeds") + 1] == "0,1"
    assert command[command.index("--c") + 1] == "0.1,0.3"
    assert command[command.index("--out-dir") + 1] == "results/survey"


def test_edge_a_partly_finished_unit_splits_per_seed_and_runs_only_what_is_left():
    """The script runs seeds x tokens, so a union batch would re-run done cells.

    Re-running is not free: it trains again and writes another
    ``checkpoints/attempt-*`` directory that nothing reuses.
    """
    done = ("spambase", "nnpu", "native_2d", "scar", "0.1", 0)
    runs = tuple(
        PilotRun("spambase", "nnpu", "native_2d", "scar", token, seed)
        for seed in (0, 1)
        for token in ("0.1", "0.3")
        if ("spambase", "nnpu", "native_2d", "scar", token, seed) != done
    )

    batched = batches(runs)
    covered = {key for batch in batched for key in batch.covers}
    assert covered == {run.key for run in runs}
    # One batch per seed, none carrying the pair that would re-run the cell
    # already on disk.
    assert len(batched) == 2
    assert all(len(batch.seeds) == 1 for batch in batched)


def test_basic_an_oracle_batch_carries_no_method_mechanism_or_class_prior():
    """An oracle is a clean label view; the script rejects those beside it.

    It does *not* reject ``--c`` -- only ``--oracle`` with ``--method`` or
    ``--class-prior``.  The batch leaves ``--c`` off because an oracle has no c
    to vary, not because the script would refuse it.
    """
    batch = batches((PilotRun("spambase", PN_ORACLE, "native_2d", None, None, 0),))[0]
    command = batch_command(
        batch, splits_root="data/splits", out_root="results/survey", device="cuda"
    )
    assert "--oracle" in command
    assert "--method" not in command
    assert "--labeling-mechanism" not in command
    assert "--c" not in command
    assert "--class-prior" not in command


def test_basic_the_command_passes_through_only_what_the_caller_supplied():
    """Inventing a device or a prior here is the divergence the driver avoids."""
    batch = batches((PilotRun("spambase", "nnpu", "native_2d", "scar", "0.1", 0),))[0]
    bare = batch_command(batch, splits_root="data/splits", out_root="results/survey")
    assert "--device" not in bare
    assert "--class-prior" not in bare

    given = batch_command(
        batch,
        splits_root="data/splits",
        out_root="results/survey",
        class_prior=0.39,
        device="cuda",
        extraction_batch_size=64,
    )
    # A prior the method needs must actually reach the run: without it the
    # script refuses the row before reading a single split.
    assert given[given.index("--class-prior") + 1] == "0.39"
    assert given[given.index("--device") + 1] == "cuda"
    assert given[given.index("--extraction-batch-size") + 1] == "64"


def test_basic_a_run_key_of_none_mechanism_is_reported_as_the_oracle() -> None:
    payload = _manifest("spambase", PN_ORACLE, "native_2d", 3)
    payload.pop("c_requested_token")
    assert manifest_identity(payload) == ("spambase", PN_ORACLE, "native_2d", None, None, 3)


# --- what the pilot needs on disk --------------------------------------------


def _estimate(protocol, **kwargs):
    return estimate_checkpoint_bytes(protocol, input_dims={"spambase": 57}, **kwargs)


def test_basic_the_estimate_separates_one_run_from_all_of_them():
    """Retries persist into their own attempt directory, so the reserve is real.

    Reporting the reserve as the accumulation would size a host for storage
    that only materialises when runs have to be retried; reporting the
    accumulation as the reserve would let a run be refused for lack of space
    the guard insists on.
    """
    estimate = _estimate(_protocol())
    # 4 * (57*128 + 128) + 4 * (128*1 + 1) bytes per component, times the
    # unit's 200 epochs at one attempt, times the single candidate.
    per_run = 4 * (57 * 128 + 128 + 128 + 1) * 200
    assert estimate["peak_bytes_per_run"] == per_run
    assert estimate["peak_unit"] == "spambase/nnpu/native_2d"
    # nnpu runs 2 seeds x 3 tokens; the oracle once per seed; upu never.
    assert estimate["total_bytes"] == per_run * 8
    assert estimate["units"] == 3
    assert estimate["attempts_reserved"] == 2
    assert estimate["guard_peak_bytes_per_run"] == estimate["peak_bytes_per_run"] * 2
    assert estimate["guard_total_bytes"] == estimate["total_bytes"] * 2


def test_edge_a_unit_that_writes_no_checkpoints_is_recorded_as_zero():
    """Closed-form rows keep no per-epoch state.

    Recorded as an explicit zero rather than left out, because absence cannot
    be told apart from a unit this estimate never looked at.
    """
    estimate = _estimate(_protocol())
    assert estimate["per_unit_bytes"]["spambase/upu/native_2d"] == {
        "runs": 6,
        "bytes_per_run": 0,
        "guard_bytes_per_run": 0,
    }
    assert estimate["total_bytes"] > 0


def test_param_an_mlp_unit_without_its_dimension_is_refused():
    with pytest.raises(ValueError, match="supply one for 'spambase'"):
        estimate_checkpoint_bytes(_protocol(), input_dims={})


# --- the driver --------------------------------------------------------------


def test_basic_the_driver_reports_the_plan_without_running_anything(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "run_survey_pilot", _SCRIPTS / "run_survey_pilot.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    assert driver.main(["--dry-run", "--results", str(tmp_path / "none")]) == 0
    printed = capsys.readouterr().out
    assert "planned: 645 run(s)" in printed
    assert "pending:   645" in printed
    assert "peak per run" in printed
