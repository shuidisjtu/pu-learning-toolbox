# ruff: noqa: N803, F811
"""Real unit CLI, separate profile outputs, metadata, batch/resume identities."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from _profile_identity_helpers import draft, write_matrix
from _survey_script_helpers import make_splits, survey_script  # noqa: F401

from pu_toolbox.experiment.pilot_plan import (
    PilotScope,
    completed_runs,
    estimate_checkpoint_bytes,
    pending_runs,
    planned_runs,
    select_execution_units,
)
from pu_toolbox.experiment.survey_profile_identity import FIELD
from pu_toolbox.experiment.survey_protocol import digest

pytestmark = pytest.mark.unit


def driver():
    path = Path(__file__).resolve().parents[3] / "scripts/run_survey_pilot.py"
    spec = importlib.util.spec_from_file_location("profile_identity_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(script, directory, protocol, name="upu_recipe", prepare=make_splits):
    inputs = directory / "input"
    inputs.mkdir(parents=True)
    prepare(inputs)
    matrix = write_matrix(directory, protocol)
    output = directory / "result"
    code = script.main(
        [
            str(inputs),
            "--protocol",
            str(matrix),
            "--dataset",
            "spambase",
            "--method",
            name,
            "--seeds",
            "0",
            "--c",
            "0.5",
            "--class-prior",
            "0.4",
            "--os-or-ts",
            "os",
            "--out-dir",
            str(output),
        ]
    )
    return code, output


def test_basic_real_unit_cli_writes_profile_algorithm_and_correct_native_ledger(
    survey_script, tmp_path
):
    code, output = execute(survey_script, tmp_path, draft())
    assert code == 0
    path = next(output.rglob("manifest.json"))
    manifest = json.loads(path.read_text())
    assert manifest["profile_id"] == manifest["execution_unit"]["method"] == "upu_recipe"
    assert manifest["algorithm_id"] == "upu" and FIELD in manifest
    assert not manifest["formal_eligible"]
    assert "profile_identity_extension_engineering_only" in manifest["formal_blockers"]
    assert set(manifest["selection"]) == {"PA", "OA"}
    assert manifest["comparison_registration_status"] == "missing_versioned_profile_mapping"
    assert "comparison" not in manifest
    ledger = json.loads((path.parent / "method_ledger_entry.json").read_text())
    assert ledger["class"] == "UPUClassifier"


def test_determ_same_algorithm_profiles_keep_separate_outputs_and_equal_metrics(
    survey_script, tmp_path
):
    new = draft()
    new["method_profiles"]["upu_other"] = copy.deepcopy(new["method_profiles"]["upu_recipe"])
    other = copy.deepcopy(new["execution_units"][0])
    other["method"] = "upu_other"
    new["execution_units"].append(other)
    manifests = []
    for name in ("upu_recipe", "upu_other"):
        code, output = execute(survey_script, tmp_path / name, new, name)
        assert code == 0
        manifests.append(json.loads(next(output.rglob("manifest.json")).read_text()))
    assert manifests[0]["generation"] == manifests[1]["generation"]
    assert manifests[0]["test_results"] == manifests[1]["test_results"]
    assert manifests[0]["profile_id"] != manifests[1]["profile_id"]
    assert manifests[0]["algorithm_id"] == manifests[1]["algorithm_id"] == "upu"


def test_basic_planner_resolves_prior_view_and_checkpoint_components_from_algorithm():
    protocol = draft("nnpu")
    runs = planned_runs(protocol)
    pilot = driver()
    assert pilot._missing_priors(protocol, runs, {}) == ["spambase"]
    assert pilot._missing_priors(protocol, runs, {"spambase": 0.4}) == []
    assert set(pilot.expected_run_views(runs, "os", protocol=protocol).values()) == {
        "os-compatible"
    }
    assert set(pilot.expected_run_views(runs, None, protocol=protocol).values()) == {
        "ts-compatible"
    }
    estimate = estimate_checkpoint_bytes(protocol, input_dims={"spambase": 3})
    assert estimate["total_bytes"] > 0


@pytest.mark.parametrize(
    "fault", ["missing", "algorithm", "profile", "protocol", "bool_seed", "schema"]
)
def test_param_resume_refuses_missing_or_inconsistent_profile_bindings(
    fault, survey_script, tmp_path
):
    protocol = draft()
    code, output = execute(survey_script, tmp_path, protocol)
    assert code == 0
    runs = planned_runs(protocol)
    views = {run.key: "os-compatible" for run in runs}
    pending, done = pending_runs(protocol, output, expected_views=views)
    assert not pending and len(done) == 1
    path = next(output.rglob("manifest.json"))
    payload = json.loads(path.read_text())
    if fault == "missing":
        payload.pop("algorithm_id")
    elif fault == "algorithm":
        payload["algorithm_id"] = "nnpu"
    elif fault == "profile":
        payload["profile_id"] = "upu_other"
    elif fault == "protocol":
        payload["protocol_sha256"] = "0" * 64
    elif fault == "bool_seed":
        payload["seed"] = False
    else:
        payload[FIELD] = "unknown"
    path.write_text(json.dumps(payload))
    pending, done = pending_runs(protocol, output, expected_views=views)
    assert len(pending) == 1 and not done
    if fault in ("missing", "profile", "bool_seed", "schema"):
        assert not completed_runs(output)


def test_edge_narrowed_resume_uses_full_source_protocol_identity(survey_script, tmp_path):
    protocol = draft()
    unused = copy.deepcopy(protocol["execution_units"][0])
    unused["dataset"] = "imdb"
    protocol["execution_units"].append(unused)
    code, output = execute(survey_script, tmp_path, protocol)
    assert code == 0
    selected = select_execution_units(protocol, PilotScope(datasets=("spambase",)))
    views = {run.key: "os-compatible" for run in planned_runs(selected)}
    pending, done = pending_runs(
        selected, output, expected_views=views, source_protocol_sha256=digest(protocol)
    )
    assert not pending and len(done) == 1
    pending, done = pending_runs(selected, output, expected_views=views)
    assert pending and not done  # The narrowed matrix is not what executed.


def test_edge_partial_expected_profile_identity_cannot_turn_resume_into_legacy(tmp_path):
    run = planned_runs(draft())[0]
    with pytest.raises(ValueError, match="complete"):
        completed_runs(tmp_path, expected_profile_identities={run.key: {}})


def test_edge_full_meta_cli_refuses_before_loading_any_partitions(
    survey_script, tmp_path, monkeypatch
):
    protocol = draft("self_pu", "self_pu_full_meta")
    profile = protocol["method_profiles"]["self_pu_full_meta"]
    profile["supervision"] = "independent_clean_support"
    profile["params"].update(require_validation=True, distillation_scope="paper_population")
    path = write_matrix(tmp_path, protocol)

    def forbidden(*args, **kwargs):
        pytest.fail("support-dependent unit reached the four-role loader")

    monkeypatch.setattr(survey_script, "load_split_parts", forbidden)
    assert (
        survey_script.main(
            [
                "missing",
                "--protocol",
                str(path),
                "--dataset",
                "spambase",
                "--method",
                "self_pu_full_meta",
            ]
        )
        == 1
    )


def test_determ_real_nnpu_profile_matches_plain_native_epoch_weights(survey_script, tmp_path):
    torch = pytest.importorskip("torch")
    from pu_toolbox.experiment.strategies import SCARGenerator
    from pu_toolbox.experiment.survey_execution import assemble_model
    from pu_toolbox.experiment.survey_protocol import load_protocol

    def enough_validation_labels(directory):
        features = np.random.RandomState(0).normal(size=(200, 3))
        truth = np.tile([1, 1, 0, 0, 0], 40)
        boundaries = {
            "train": (0, 120),
            "pu_val": (120, 140),
            "clean_val": (140, 160),
            "test": (160, 200),
        }
        for role, (a, b) in boundaries.items():
            np.savez(
                directory / f"{role}.npz", X=features[a:b], y=truth[a:b], indices=np.arange(a, b)
            )
        (directory / "split_manifest.json").write_text(
            json.dumps(
                {
                    "dataset": "spambase",
                    "seed": 0,
                    "class_prior": {"population": 0.4},
                    "role_sizes": {role: b - a for role, (a, b) in boundaries.items()},
                }
            )
        )

    protocol = draft("nnpu")
    code, output = execute(
        survey_script, tmp_path, protocol, "nnpu_recipe", prepare=enough_validation_labels
    )
    assert code == 0
    manifest = json.loads(next(output.rglob("manifest.json")).read_text())
    parts = survey_script.load_split_parts(tmp_path / "input")
    observed, _ = SCARGenerator().generate(parts[0].X, parts[0].labels, 0.5, 0)
    val_observed, _ = SCARGenerator().generate(parts[1].X, parts[1].labels, 0.5, 0)
    legacy = load_protocol()
    row = next(
        row
        for row in legacy["execution_units"]
        if row["dataset"] == "spambase" and row["method"] == "nnpu"
    )
    plain = assemble_model(legacy, row, 3, seed=0, params={}, class_prior=0.4, device="cpu")
    plain.set_params(max_epochs=2)
    weights = []

    def record(epoch, fitted):
        weights.append(
            {
                name: value.detach().cpu().clone()
                for name, value in fitted.model_.state_dict().items()
            }
        )

    plain.fit(
        parts[0].X,
        observed,
        class_prior=0.4,
        os_or_ts="os",
        validation_data=(parts[1].X, val_observed),
        epoch_callback=record,
    )
    rows = manifest["candidate_runs"][0]["epoch_checkpoints"]
    assert len(rows) == len(weights) == 2
    for row, expected in zip(rows, weights, strict=True):
        actual = torch.load(row["path"], map_location="cpu", weights_only=True)
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
    assert not manifest["formal_eligible"] and "comparison" not in manifest


def test_basic_actual_pilot_process_and_second_pass_resume_are_profile_aware(tmp_path, monkeypatch):
    from pu_toolbox.experiment.survey_protocol import digest

    pilot = driver()
    protocol = draft()
    excluded = copy.deepcopy(protocol["execution_units"][0])
    excluded["dataset"] = "imdb"
    protocol["execution_units"].append(excluded)
    matrix = write_matrix(tmp_path, protocol)
    inputs = tmp_path / "splits" / "spambase" / "split_0"
    inputs.mkdir(parents=True)
    make_splits(inputs)
    identity = {}
    for role in ("train", "pu_val", "clean_val", "test"):
        with np.load(inputs / f"{role}.npz") as part:
            identity[role] = part["indices"].tolist()
    manifest_path = inputs / "split_manifest.json"
    split_manifest = json.loads(manifest_path.read_text())
    split_manifest["indices_sha256"] = digest(identity)
    manifest_path.write_text(json.dumps(split_manifest))
    # The standalone subprocess imports from this checkout, not an installed old wheel.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[3]))
    snapshot = tmp_path / "plan.json"
    args = [
        "--protocol",
        str(matrix),
        "--splits",
        str(tmp_path / "splits"),
        "--results",
        str(tmp_path / "results"),
        "--datasets",
        "spambase",
        "--os-or-ts",
        "os",
        "--plan-json",
        str(snapshot),
    ]
    assert pilot.main([*args, "--dry-run"]) == 0
    plan = json.loads(snapshot.read_text())
    assert plan["profile_identities"]["upu_recipe"]["algorithm_id"] == "upu"
    assert plan["totals"]["pending"] == 1
    assert pilot.main(args[:-2]) == 0  # --plan-json is deliberately dry-run only.

    def forbidden(*args, **kwargs):
        pytest.fail("second pass must resume rather than retrain")

    monkeypatch.setattr(pilot, "_run_batch", forbidden)
    assert pilot.main([*args, "--dry-run"]) == 0
    plan = json.loads(snapshot.read_text())
    assert plan["totals"]["completed"] == 1 and plan["totals"]["pending"] == 0


def test_edge_cleared_blockers_cannot_promote_engineering_profile(survey_script, tmp_path):
    from pu_toolbox.experiment.survey_protocol import validate_comparable_manifests

    protocol = draft()
    protocol["review_status"] = "accepted"  # Adversarial scratch data, not a human signature.
    protocol["formal_blockers"] = []
    code, output = execute(survey_script, tmp_path, protocol)
    assert code == 0
    manifest = json.loads(next(output.rglob("manifest.json")).read_text())
    assert not manifest["formal_eligible"]
    assert "profile_identity_extension_engineering_only" in manifest["formal_blockers"]
    with pytest.raises(ValueError, match="formal"):
        validate_comparable_manifests([manifest])
