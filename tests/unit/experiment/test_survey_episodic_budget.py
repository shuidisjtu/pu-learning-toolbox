# ruff: noqa: N803, F811
"""Real unit execution, native stopping, paid episodes and LZO budget evidence."""

import copy
import functools
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from _episodic_budget_helpers import matrix, training_data
from _profile_identity_helpers import write_matrix
from _survey_script_helpers import make_splits, survey_script  # noqa: F401

from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer
from pu_toolbox.experiment.pilot_plan import estimate_checkpoint_bytes
from pu_toolbox.experiment.survey_execution import assemble_model
from pu_toolbox.experiment.survey_protocol import load_protocol
from pu_toolbox.experiment.survey_staged_budget import (
    FIELD,
    SCHEMA,
    checkpoint_coverage,
    validate_contract,
    validate_estimator,
    verify_trajectory,
)

pytestmark = pytest.mark.unit


def fit(algorithm, tmp_path, *, seed=0, view="os", **overrides):
    protocol = matrix(algorithm, **overrides)
    model = assemble_model(
        protocol,
        protocol["execution_units"][0],
        3,
        seed=seed,
        params={},
        class_prior=0.4,
        device="cpu",
    )
    features, labels = training_data()
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        model, features, labels, class_prior=0.4, os_or_ts=view
    )
    return protocol, model, trajectory, features, labels


@pytest.mark.parametrize(
    "algorithm,view",
    [
        ("robust_pu", "os"),
        ("robust_pu", "ts"),
        ("split_pu", "os"),
        ("split_pu", "ts"),
        ("holistic_pu", "os"),
    ],
)
def test_basic_real_cli_verifies_paid_stages_and_bounded_storage(
    algorithm, view, survey_script, tmp_path
):
    protocol = matrix(algorithm)
    source = write_matrix(tmp_path, protocol)
    assert load_protocol(source) == protocol
    inputs = tmp_path / "input"
    inputs.mkdir()
    make_splits(inputs)
    output = tmp_path / "results"
    assert (
        survey_script.main(
            [
                str(inputs),
                "--protocol",
                str(source),
                "--dataset",
                "spambase",
                "--method",
                f"{algorithm}_recipe",
                "--seeds",
                "0",
                "--c",
                ".5",
                "--class-prior",
                ".4",
                "--os-or-ts",
                view,
                "--out-dir",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads(next(output.rglob("manifest.json")).read_text())
    assert not manifest["failures"] and not manifest["formal_eligible"]
    assert manifest["selection_checkpoint_scope"] == "independent_per_epoch"
    assert set(manifest["selection"]) == {"PA", "OA"}
    candidate = manifest["candidate_runs"][0]
    receipt = candidate["staged_budget_receipt"]
    cap = protocol["budgets"]["native_stages"][FIELD]["checkpoint_epoch_count"]
    assert receipt["checkpoint_epoch_bound"] == cap
    assert receipt["checkpoint_epoch_count"] == len(candidate["epoch_checkpoints"]) <= cap
    assert not receipt["formal_budget_approved"]
    assert sum(receipt["stage_optimizer_steps"].values()) == receipt["optimizer_steps"]
    expected = {"robust_pu": 14, "holistic_pu": 14}.get(algorithm)
    if expected is not None:
        assert receipt["optimizer_steps"] == expected
    estimate = estimate_checkpoint_bytes(protocol, input_dims={"spambase": 3})
    assert estimate["peak_bytes_per_run"] == cap * 2**20
    assert manifest["resources"]["checkpoint_disk_preflight"]["required_bytes"] == cap * 2**21


@pytest.mark.parametrize(
    "algorithm,overrides,view",
    [
        ("robust_pu", {}, "os"),
        ("split_pu", {}, "os"),
        ("robust_pu", {}, "ts"),
        ("split_pu", {}, "ts"),
        ("holistic_pu", {}, "os"),
        ("holistic_pu", {"warmup_selection": "lzo_positive_loss"}, "os"),
        (
            "holistic_pu",
            {"warmup_selection": "lzo_positive_loss", "pseudo_pn_initialization": "reinitialize"},
            "os",
        ),
    ],
)
@pytest.mark.parametrize("seed", [0, 7])
def test_determ_native_state_is_unchanged_by_recording_and_budget_verification(
    algorithm, overrides, view, seed, tmp_path
):
    torch = pytest.importorskip("torch")
    protocol = matrix(algorithm, **overrides)
    model = assemble_model(
        protocol,
        protocol["execution_units"][0],
        3,
        seed=seed,
        params={},
        class_prior=0.4,
        device="cpu",
    )
    plain = copy.deepcopy(model)
    features, labels = training_data()
    plain.fit(features, labels, class_prior=0.4, os_or_ts=view)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        model, features, labels, class_prior=0.4, os_or_ts=view
    )
    receipt = verify_trajectory(
        model, trajectory, protocol["budgets"]["native_stages"][FIELD], features, labels, view
    )
    assert receipt["optimizer_steps"] == plain.optimizer_steps_
    assert model.history_ == plain.history_
    for name, tensor in plain.model_.state_dict().items():
        torch.testing.assert_close(tensor, model.model_.state_dict()[name], rtol=0, atol=0)
    np.testing.assert_array_equal(
        plain.decision_function(features),
        trajectory.checkpoints[-1].restore().decision_function(features),
    )
    if algorithm == "holistic_pu" and overrides:
        assert receipt["lzo"]["forward_rows"] == 4 * 3
        assert receipt["optimizer_steps"] == 14


@pytest.mark.parametrize(
    "fault",
    ["old_schema", "missing", "bool", "bound", "threshold", "stages", "infinite", "unknown"],
)
def test_param_invalid_episode_contracts_fail_before_loading(fault, tmp_path):
    protocol = matrix("split_pu")
    contract = protocol["budgets"]["native_stages"][FIELD]
    if fault == "old_schema":
        contract["schema_version"] = SCHEMA
    elif fault == "missing":
        del contract["parameter_values"]["rounds"]
    elif fault == "bool":
        contract["parameter_values"]["split_epochs"] = True
    elif fault == "bound":
        contract["checkpoint_epoch_count"] -= 1
    elif fault == "threshold":
        contract["parameter_values"]["agreement_threshold"] = 1.0
    elif fault == "stages":
        contract["eligible_stages"] = ["student"]
    elif fault == "infinite":
        contract["parameter_values"]["learning_rate"] = float("inf")
    else:
        contract["parameter_values"]["unapproved_extra_budget"] = 1
    with pytest.raises(ValueError):
        validate_contract(contract)
    with pytest.raises(ValueError):
        load_protocol(write_matrix(tmp_path, protocol))


@pytest.mark.parametrize(
    "fault", ["round", "step", "missing_checkpoint", "stopping_rule", "partition", "bytes"]
)
def test_param_native_splitter_receipts_cannot_hide_invalid_work(fault, tmp_path):
    protocol, model, trajectory, features, labels = fit("split_pu", tmp_path)
    contract = protocol["budgets"]["native_stages"][FIELD]
    if fault == "round":
        trajectory.checkpoints[-1].training_context["round_index"] = 100
    elif fault == "step":
        trajectory.checkpoints[-1].training_context["optimizer_steps"] -= 1
    elif fault == "missing_checkpoint":
        trajectory.checkpoints.pop()
    elif fault == "stopping_rule":
        model.history_["split_agreement"] = [1.0, 1.0]
    elif fault == "partition":
        model.n_hard_ = 0
    else:
        contract["checkpoint_bytes_per_component"] = 1
    with pytest.raises(ValueError):
        verify_trajectory(model, trajectory, contract, features, labels, "os")


def test_edge_zero_pretrain_keeps_episode_inner_cost_not_phantom_snapshots(tmp_path):
    protocol, model, trajectory, features, labels = fit("robust_pu", tmp_path, pretrain_epochs=0)
    contract = protocol["budgets"]["native_stages"][FIELD]
    receipt = verify_trajectory(model, trajectory, contract, features, labels, "os")
    assert receipt["checkpoint_epoch_count"] == 2
    assert receipt["stage_optimizer_steps"] == {"pretrain": 0, "self_paced": 12}
    assert checkpoint_coverage(trajectory, protocol["budgets"]["native_stages"])
    with pytest.raises(ValueError, match="nonzero pretrain"):
        verify_trajectory(model, trajectory, contract, features, labels, "ts")


@pytest.mark.parametrize("threshold,expected", [(0.01, 1), (0.99999, 4)])
def test_edge_native_splitter_early_stop_and_cap_both_have_complete_coverage(
    threshold, expected, tmp_path
):
    protocol, model, trajectory, features, labels = fit(
        "split_pu", tmp_path, seed=7, agreement_threshold=threshold
    )
    contract = protocol["budgets"]["native_stages"][FIELD]
    receipt = verify_trajectory(model, trajectory, contract, features, labels, "os")
    assert receipt["splitter_epochs_completed"] == expected
    assert receipt["checkpoint_epoch_count"] == 1 + expected + 2 * 2
    assert receipt["checkpoint_epoch_bound"] == 9
    assert checkpoint_coverage(trajectory, protocol["budgets"]["native_stages"])


@pytest.mark.parametrize(
    "fault",
    ["refund", "forward_rows", "forward_batches", "selected", "foreign_pair", "optimizer_reset"],
)
def test_param_lzo_rollback_cannot_refund_paid_warmup_or_validation_work(fault, tmp_path):
    protocol, model, trajectory, features, labels = fit(
        "holistic_pu", tmp_path, warmup_selection="lzo_positive_loss", seed=0
    )
    contract = protocol["budgets"]["native_stages"][FIELD]
    receipt = verify_trajectory(model, trajectory, contract, features, labels, "os")
    assert receipt["optimizer_steps"] == 14
    assert receipt["lzo"]["selected_warmup_epoch"] == 2
    assert receipt["lzo"]["discarded_state_optimizer_steps_still_paid"] == 4
    if fault == "refund":
        model.optimizer_steps_ -= 4
    elif fault == "forward_rows":
        model.lzo_evaluated_rows_ -= 3
    elif fault == "forward_batches":
        model.lzo_forward_batches_ -= 1
    elif fault == "selected":
        model.selected_warmup_epoch_ = 1
    elif fault == "foreign_pair":
        model.lzo_mixup_indices_[0, 0] = len(features) + 100
    else:
        model.pseudo_pn_optimizer_reset_ = True
    with pytest.raises(ValueError):
        verify_trajectory(model, trajectory, contract, features, labels, "os")


@pytest.mark.parametrize(
    "algorithm,overrides",
    [
        ("robust_pu", {"pretrain_epochs": 0}),
        ("holistic_pu", {}),
    ],
)
def test_edge_unsupported_view_is_refused_before_a_training_attempt(
    algorithm, overrides, survey_script, tmp_path, monkeypatch
):
    from pu_toolbox.registry import get_algorithm

    protocol = matrix(algorithm, **overrides)
    source = write_matrix(tmp_path, protocol)
    load_protocol(source)
    klass = get_algorithm(algorithm)

    @functools.wraps(klass.fit)
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported view must not spend a native training attempt")

    monkeypatch.setattr(klass, "fit", forbidden)
    inputs = tmp_path / "input"
    inputs.mkdir()
    make_splits(inputs)
    assert (
        survey_script.main(
            [
                str(inputs),
                "--protocol",
                str(source),
                "--dataset",
                "spambase",
                "--method",
                f"{algorithm}_recipe",
                "--seeds",
                "0",
                "--c",
                ".5",
                "--class-prior",
                ".4",
                "--os-or-ts",
                "ts",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )


def test_basic_actual_split_pilot_resumes_after_native_early_stop(tmp_path, monkeypatch):
    from pu_toolbox.experiment.survey_protocol import digest

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "episodic_pilot", root / "scripts/run_survey_pilot.py"
    )
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    protocol = matrix("split_pu")
    source = write_matrix(tmp_path, protocol)
    inputs = tmp_path / "splits" / "spambase" / "split_0"
    inputs.mkdir(parents=True)
    make_splits(inputs)
    identities = {}
    for role in ("train", "pu_val", "clean_val", "test"):
        with np.load(inputs / f"{role}.npz") as data:
            identities[role] = data["indices"].tolist()
    split_path = inputs / "split_manifest.json"
    split = json.loads(split_path.read_text())
    split["indices_sha256"] = digest(identities)
    split_path.write_text(json.dumps(split))
    monkeypatch.setenv("PYTHONPATH", str(root))
    output = tmp_path / "results"
    args = [
        "--protocol",
        str(source),
        "--splits",
        str(tmp_path / "splits"),
        "--results",
        str(output),
        "--datasets",
        "spambase",
        "--os-or-ts",
        "os",
    ]
    assert pilot.main(args) == 0
    manifest = json.loads(next(output.rglob("manifest.json")).read_text())
    receipt = manifest["candidate_runs"][0]["staged_budget_receipt"]
    assert receipt["checkpoint_epoch_count"] < receipt["checkpoint_epoch_bound"]
    assert manifest["selection_checkpoint_scope"] == "independent_per_epoch"

    def forbidden(*args, **kwargs):
        pytest.fail("completed native early-stop trajectory must resume rather than retrain")

    monkeypatch.setattr(pilot, "_run_batch", forbidden)
    snapshot = tmp_path / "plan.json"
    assert pilot.main([*args, "--dry-run", "--plan-json", str(snapshot)]) == 0
    assert json.loads(snapshot.read_text())["totals"]["completed"] == 1


@pytest.mark.parametrize("field", ["model", "encoder"])
def test_edge_injected_module_cannot_masquerade_as_default_native_architecture(field):
    torch = pytest.importorskip("torch")
    protocol = matrix("robust_pu")
    model = assemble_model(
        protocol,
        protocol["execution_units"][0],
        3,
        seed=0,
        params={},
        class_prior=0.4,
        device="cpu",
    )
    model.set_params(**{field: torch.nn.Linear(3, 1)})
    with pytest.raises(ValueError, match="without injected modules"):
        validate_estimator(model, protocol["budgets"]["native_stages"][FIELD])
