"""Pinned, explicit CPU PUET/RP plans execute and retain attempt evidence."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_classical_candidate_experiment as script  # noqa: E402

pytestmark = pytest.mark.unit


def product(tmp_path):
    directory = tmp_path / "input"
    directory.mkdir()
    for i, role in enumerate(script.ROLES):
        labels = np.tile([0, 1], 18)
        np.savez(
            directory / f"{role}.npz",
            X=np.random.RandomState(i).normal(size=(36, 3)) + labels[:, None],
            y=labels,
            indices=np.arange(36) + i * 36,
        )
    return directory


def plan(directory, method="puet", **overrides):
    parameters = (
        dict(
            n_estimators=3,
            max_depth=3,
            min_samples_leaf=1,
            max_features="all",
            max_candidates=1,
            bootstrap=False,
        )
        if method == "puet"
        else dict(n_cv_folds=3, frac_pos2neg=None, min_retained=2)
    )
    return script.build_plan(
        directory,
        **(
            dict(
                method=method,
                dataset="synthetic",
                prior=0.5,
                prior_source="generator",
                seed=1,
                view="os",
                parameters=parameters,
                c_token="0.5",
            )
            | overrides
        ),
    )


@pytest.mark.parametrize("method,view", [("puet", "os"), ("puet", "ts"), ("rp", "os")])
def test_basic_determ_four_role_selection_and_non_epoch_work_replay(tmp_path, method, view):
    directory = product(tmp_path)
    planned = plan(directory, method, view=view)
    assert planned == plan(directory, method, view=view)
    work = 3 if method == "puet" else 4
    assert planned["work_units_per_attempt"] == work
    assert planned["two_attempt_work_upper_bound"] == 2 * work
    assert planned["neural_epoch_count"] == 0 and not planned["formal_admission"]
    summary = script.execute(planned, tmp_path / "output")
    assert summary["accepted_engineering_run"] and summary["selected_models_replayed"]
    assert summary["warning_count"] == summary["retry_count"] == 0
    assert summary["training_provenance"]["work_completed"] == work
    assert summary["training_provenance"]["calibration_applied"] is (view == "ts")
    assert len(summary["attempt_receipts_sha256"]) == 1 and summary["neural_epoch_count"] == 0
    assert set(summary["selection"]) == {"PA", "OA"}
    assert all(s["epoch"] is None for s in summary["selection"].values())
    assert not list((tmp_path / "output").rglob("*.pt"))
    if method == "rp":
        assert not summary["training_provenance"]["external_class_prior_used"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"prior": True},
        {"seed": True},
        {"prior_source": ""},
        {"c_token": "0.50"},
        {"parameters": {}},
        {"parameters": {"bootstrap": "false"}},
    ],
)
def test_param_ambiguous_or_incomplete_configuration_refused(tmp_path, overrides):
    with pytest.raises(ValueError):
        plan(product(tmp_path), **overrides)


def test_param_rp_ts_and_insufficient_observed_folds_refused(tmp_path):
    directory = product(tmp_path)
    with pytest.raises(ValueError, match="native OS"):
        plan(directory, "rp", view="ts")
    with pytest.raises(ValueError, match="n_cv_folds rows"):
        plan(directory, "rp", parameters=dict(n_cv_folds=20, frac_pos2neg=None, min_retained=2))


def test_edge_changed_plan_source_or_environment_refused_before_output(tmp_path, monkeypatch):
    planned = plan(product(tmp_path))
    changed = copy.deepcopy(planned)
    changed["parameters"]["n_estimators"] += 1
    with pytest.raises(ValueError, match="plan changed"):
        script.execute(changed, tmp_path / "bad")
    original = script.file_hash
    with monkeypatch.context() as patch:
        patch.setattr(
            script,
            "file_hash",
            lambda path: "changed" if str(path).endswith("risk/puet.py") else original(path),
        )
        with pytest.raises(ValueError, match="identity changed"):
            script.execute(planned, tmp_path / "bad")
    monkeypatch.setattr(script, "version", lambda name: "different sklearn")
    with pytest.raises(ValueError, match="identity changed"):
        script.execute(planned, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_edge_overlapping_roles_and_mismatched_features_rejected(tmp_path):
    directory = product(tmp_path)
    path = directory / "test.npz"
    with np.load(path) as original:
        features, labels, indices = original["X"], original["y"], original["indices"]
    np.savez(path, X=features, y=labels, indices=np.arange(36))
    with pytest.raises(ValueError, match="overlap"):
        plan(directory)
    np.savez(path, X=features[:, :2], y=labels, indices=indices)
    with pytest.raises(ValueError, match="feature count"):
        plan(directory)


def test_edge_output_overlap_and_existing_output_preserve_files(tmp_path):
    directory = product(tmp_path)
    planned = plan(directory)
    for target in (tmp_path, directory, directory / "nested"):
        with pytest.raises(ValueError, match="overlap"):
            script.execute(planned, target)
    output = tmp_path / "already-exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        script.execute(planned, output)
    assert not list(output.iterdir())


def test_edge_failed_retry_keeps_both_partial_receipts_and_failure_manifest(tmp_path, monkeypatch):
    planned = plan(product(tmp_path))
    original = script.PUExtraTreesClassifier._build_tree
    calls = []

    def fail_even(self, *args, **kwargs):
        calls.append(1)
        if len(calls) % 2 == 0:
            raise TypeError("injected tree error")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(script.PUExtraTreesClassifier, "_build_tree", fail_even)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="all candidate runs failed"):
        script.execute(planned, output)
    failure = json.loads((output / "failure.json").read_text())
    assert len(calls) == 4 and not failure["accepted_engineering_run"]
    assert len(failure["attempt_receipts"]) == 2
    for relative in failure["attempt_receipts"]:
        receipt = json.loads((output / relative).read_text())
        assert receipt["status"] == "failed" and receipt["work_completed"] == 1
        assert receipt["work_started"] == 2 and receipt["error_type"] == "TypeError"
    assert (output / "manifest.json").is_file() and not (output / "summary.json").exists()


def test_basic_cli_read_only_plan_does_not_fit_or_create_artifacts(tmp_path, capsys, monkeypatch):
    directory = product(tmp_path)
    requested = tmp_path / "parameters.json"
    requested.write_text(json.dumps(plan(directory)["parameters"]))
    monkeypatch.setattr(
        script.PUExtraTreesClassifier, "fit", lambda *a, **k: pytest.fail("plan fitted")
    )
    result = script.main(
        [
            "--method",
            "puet",
            "--data-dir",
            str(directory),
            "--dataset",
            "synthetic",
            "--class-prior",
            "0.5",
            "--prior-source",
            "generator",
            "--seed",
            "1",
            "--view",
            "os",
            "--parameters",
            str(requested),
            "--c",
            "0.5",
        ]
    )
    assert result == 0 and json.loads(capsys.readouterr().out)["execution_device"] == "cpu"
    assert not list(tmp_path.rglob("cost.json"))
