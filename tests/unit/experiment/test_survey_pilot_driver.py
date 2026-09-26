# tests/unit/experiment/test_survey_pilot_driver.py

# ruff: noqa: N803, N806, S101

"""The driver's gates, which decide whether a pilot starts at all.

A driver that refuses to start is recoverable; one that starts without the
population class prior, or with a mistyped one, trains runs that are all
wasted and says nothing.  The unit script is stubbed here so a test failure
means the gate let something through rather than that a model was trained.
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
_ALL_PRIORS = ["--class-prior", "spambase=0.39,imdb=0.5,cifar10=0.1"]


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location(
        "run_survey_pilot", _SCRIPTS / "run_survey_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def unit_calls(driver, monkeypatch):
    """Capture what the driver would have executed instead of executing it."""
    recorded: list[list[str]] = []

    def _record(argv, check=False):
        recorded.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(driver.subprocess, "run", _record)
    return recorded


def _splits(root: Path, *, prior: float | None = None, datasets=("cifar10", "imdb", "spambase")):
    for dataset in datasets:
        path = root / dataset / "split_0" / "split_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"dataset": dataset, "seed": 0, "indices_sha256": "a" * 64}
        if prior is not None:
            payload["class_prior"] = {"population": prior}
        path.write_text(json.dumps(payload), encoding="utf-8")
    return root


# --- the safety property the F1 defect violated ------------------------------


def test_edge_without_a_class_prior_no_run_is_started(driver, unit_calls, tmp_path, capsys):
    """The gate must precede the batches, not surface at the hundredth one."""
    code = driver.main(["--results", str(tmp_path / "out"), "--splits", str(_splits(tmp_path))])

    assert code == 1
    assert unit_calls == []
    assert "population class prior" in capsys.readouterr().err


def test_edge_a_prior_contradicting_the_splits_is_refused(driver, unit_calls, tmp_path, capsys):
    """A mistyped prior would train runs on the wrong constant, silently.

    Nothing downstream catches it: the aggregation gates compare budget and
    representation, not the prior.
    """
    splits = _splits(tmp_path, prior=0.39)
    out = str(tmp_path / "out")
    contradicted = ["--class-prior", "spambase=0.039,imdb=0.5,cifar10=0.1"]

    assert driver.main(["--results", out, "--splits", str(splits), *contradicted]) == 1
    assert unit_calls == []
    assert "--allow-prior-override" in capsys.readouterr().err

    # Saying so explicitly lets it through, and says so again on the way.
    code = driver.main(
        ["--results", out, "--splits", str(splits), *contradicted, "--allow-prior-override"]
    )
    assert unit_calls  # it did start batches this time
    assert code == 1  # the stub writes no manifests, so everything stays pending
    assert "overriding spambase" in capsys.readouterr().err


# --- what the driver hands the unit script -----------------------------------


def test_basic_the_recorded_prior_reaches_the_unit_script(driver, unit_calls, tmp_path):
    """π is read from the splits; needing the operator to retype it invites drift."""
    driver.main(
        ["--results", str(tmp_path / "out"), "--splits", str(_splits(tmp_path, prior=0.42))]
    )

    assert unit_calls
    for argv in unit_calls:
        script = " ".join(argv)
        if "--oracle" in argv:
            assert "--class-prior" not in argv
        else:
            assert "--class-prior" in argv
            assert argv[argv.index("--class-prior") + 1] == "0.42", script


def test_basic_dry_run_reports_the_plan_without_starting_a_batch(
    driver, unit_calls, tmp_path, capsys
):
    # An explicit splits root: defaulting to ``data/splits`` would make this
    # depend on whatever priors the developer's own artifacts happen to record.
    code = driver.main(
        [
            "--dry-run",
            "--results",
            str(tmp_path / "none"),
            "--splits",
            str(_splits(tmp_path)),
            *_ALL_PRIORS,
        ]
    )

    assert code == 0
    assert unit_calls == []
    printed = capsys.readouterr().out
    assert "planned: 645 run(s)" in printed
    assert "pending:   645" in printed
    assert "peak per run" in printed


# --- parameter errors and determinism ----------------------------------------


def test_param_a_malformed_dimension_or_prior_is_reported_not_raised(driver, tmp_path, capsys):
    for flag, value in (("--input-dims", "spambase"), ("--class-prior", "spambase=")):
        assert driver.main(["--dry-run", "--results", str(tmp_path), flag, value]) == 1
        assert capsys.readouterr().err.startswith("error: ")


def test_determ_the_batch_order_is_the_same_on_a_second_pass(driver, unit_calls, tmp_path):
    argv = ["--results", str(tmp_path / "out"), "--splits", str(_splits(tmp_path)), *_ALL_PRIORS]

    driver.main(argv)
    first = list(unit_calls)
    unit_calls.clear()
    driver.main(argv)

    assert first == unit_calls


# --- the view a resumed unit is held to ---------------------------------------


def _resume_protocol(*methods: str) -> dict:
    """A matrix whose methods do not share one default view.

    Kept small on purpose: the question is which view each unit resolves to, and
    the shipped 645-run matrix would answer it far more slowly.
    """
    return {
        "seeds": [0],
        "c_tokens": {"scar": ["0.1"]},
        "candidate_pool": [{}],
        "budgets": {"minibatch": {"epochs": 200}},
        "backbone_specs": {"mlp128": {"hidden_dims": [128], "activation": "relu"}},
        "execution_units": [
            {
                "dataset": "spambase",
                "method": method,
                "training_path": "native_2d",
                "budget": "minibatch",
                "backbone": "mlp128",
                "runnable": True,
            }
            for method in methods
        ],
    }


def _write_run(results: Path, *, run_view: str) -> Path:
    """A finished run's manifest, on the split ``_splits`` writes."""
    path = results / "spambase" / "nnpu" / "c_0.1" / "seed_0" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "execution_mode": "versioned_pilot",
                "seed": 0,
                "execution_unit": {
                    "dataset": "spambase",
                    "method": "nnpu",
                    "training_path": "native_2d",
                },
                "generation": {"train": {"mechanism": "scar"}},
                "c_requested_token": "0.1",
                "selection": {"OA": {"candidate_index": 0}},
                "run_view": run_view,
                "calibration_applied": run_view == "ts-compatible",
                "representation": {"split_sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_basic_each_method_resolves_to_its_own_default_view(driver):
    """One global expectation cannot describe this matrix.

    ``nnpu`` is native to TS and wired for calibration, so it defaults to the
    calibrated view; ``upu`` is native to TS but declares no ``os_or_ts`` hook
    and falls back to OS; ``lbe`` is native to OS; the oracle is never
    calibrated.  The driver has to resolve each one the way the unit script
    will, or it holds a resumed unit to a view that run was never going to use.
    """
    protocol = _resume_protocol("nnpu", "upu", "lbe", "pn_oracle")
    planned = driver.planned_runs(protocol)

    views = driver.expected_run_views(planned, None)

    assert {run.method: views[run.key] for run in planned} == {
        "nnpu": "ts-compatible",
        "upu": "os-compatible",
        "lbe": "os-compatible",
        "pn_oracle": "os-compatible",
    }


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, "ts-compatible"), ("os", "os-compatible"), ("ts", "ts-compatible")],
)
def test_param_a_request_selects_the_view_the_unit_script_would_use(driver, requested, expected):
    protocol = _resume_protocol("nnpu")

    views = driver.expected_run_views(driver.planned_runs(protocol), requested)

    assert set(views.values()) == {expected}


def test_edge_an_explicit_ts_on_an_os_native_method_is_refused_up_front(driver):
    """The unit script refuses this per run; the driver must refuse it before one."""
    protocol = _resume_protocol("lbe")

    with pytest.raises(ValueError, match="native to 'os'"):
        driver.expected_run_views(driver.planned_runs(protocol), "ts")


@pytest.mark.parametrize(
    ("method", "reason"),
    [
        ("lbe", "declared native to 'os'"),
        ("upu", "declares no os_or_ts parameter"),
        ("pn_oracle", "does not apply"),
    ],
)
def test_param_an_explicit_ts_is_refused_before_any_batch(
    driver, unit_calls, tmp_path, capsys, monkeypatch, method, reason
):
    """Every way a calibrated request can be impossible is refused at start-up.

    The unit script refuses each of these too, but per run -- so a pilot would
    meet it at whichever batch happened to hit that method, with the batches
    before it already paid for.  The three reasons are genuinely different: a
    method native to OS, one native to TS whose estimator has no ``os_or_ts``
    hook, and the oracle, which has no PU view to calibrate at all.  Each has to
    arrive as a readable error, not as a traceback from the batch that found it.
    """
    monkeypatch.setattr(driver, "load_protocol", lambda: _resume_protocol(method))
    splits = _splits(tmp_path, datasets=("spambase",))

    code = driver.main(
        ["--results", str(tmp_path / "out"), "--splits", str(splits), "--os-or-ts", "ts"]
    )

    assert code == 1
    assert unit_calls == []
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert reason in err
    assert "Traceback" not in err


def test_edge_a_runnable_method_the_registry_cannot_supply_is_reported(
    driver, unit_calls, tmp_path, capsys, monkeypatch
):
    """A driver error, reported as one: readable, no traceback, no batch started."""
    monkeypatch.setattr(driver, "load_protocol", lambda: _resume_protocol("no_such_method"))
    splits = _splits(tmp_path, datasets=("spambase",))

    code = driver.main(["--results", str(tmp_path / "out"), "--splits", str(splits)])

    assert code == 1
    assert unit_calls == []
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "estimator class" in err
    assert "Traceback" not in err


def test_basic_an_os_result_leaves_a_calibrated_unit_pending(
    driver, unit_calls, tmp_path, capsys, monkeypatch
):
    """F11 end to end: the default request is calibrated, so an OS run is not it."""
    monkeypatch.setattr(driver, "load_protocol", lambda: _resume_protocol("nnpu"))
    splits = _splits(tmp_path, datasets=("spambase",))
    results = tmp_path / "results"
    _write_run(results, run_view="os-compatible")

    code = driver.main(["--dry-run", "--results", str(results), "--splits", str(splits)])

    assert code == 0
    assert unit_calls == []
    assert "pending:   1" in capsys.readouterr().out
