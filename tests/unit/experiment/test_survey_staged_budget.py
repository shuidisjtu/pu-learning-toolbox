# ruff: noqa: N803, F811
"""Explicit staged budgets, native equivalence and real versioned unit runs."""

import copy
import json

import numpy as np
import pytest
from _profile_identity_helpers import draft, write_matrix
from _survey_script_helpers import make_splits, survey_script  # noqa: F401

from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer
from pu_toolbox.experiment.pilot_plan import estimate_checkpoint_bytes
from pu_toolbox.experiment.survey_execution import assemble_model
from pu_toolbox.experiment.survey_protocol import load_protocol, unit_checkpoint_bytes
from pu_toolbox.experiment.survey_staged_budget import (
    FIELD,
    SCHEMA,
    checkpoint_epochs,
    validate_contract,
    validate_estimator,
    verify_trajectory,
)

pytestmark = pytest.mark.unit


def matrix(algorithm):
    """Synthetic short budgets only, not production defaults or approval."""
    protocol = draft("nnpu", f"{algorithm}_recipe")
    params = (
        dict(
            max_epochs=2,
            classifier_epochs=3,
            batch_size=8,
            hidden_dim=4,
            latent_dim=3,
            learning_rate=3e-4,
        )
        if algorithm == "genpu"
        else dict(
            warmup_epochs=1,
            pu_epochs=2,
            positive_batch_size=2,
            unlabeled_batch_size=8,
            hidden_dim=4,
            depth=1,
            warmup_learning_rate=1e-4,
            learning_rate=1e-3,
            warmup_weight_decay=5e-4,
            weight_decay=1e-4,
        )
    )
    contract = dict(
        schema_version=SCHEMA,
        algorithm_id=algorithm,
        parameter_values=params,
        checkpoint_epoch_count=3,
        checkpoint_bytes_per_component=2**20,
        eligible_stages=["synthetic_pn"] if algorithm == "genpu" else ["warmup", "pu_mixup"],
    )
    profile = protocol["method_profiles"][f"{algorithm}_recipe"]
    profile.update(algorithm=algorithm, budget="explicit_stages", params=copy.deepcopy(params))
    protocol["budgets"] = {"explicit_stages": {FIELD: contract}}
    row = protocol["execution_units"][0]
    row.update(
        budget="explicit_stages",
        model_family=f"{algorithm}_native",
        backbone=f"{algorithm}_native_mlp",
    )
    protocol["backbone_specs"][row["backbone"]] = {
        "algorithm_id": algorithm,
        "parameter_values": {
            key: params[key] for key in ("hidden_dim", "latent_dim", "depth") if key in params
        },
    }
    return protocol


@pytest.mark.parametrize("algorithm", ["genpu", "pulda"])
@pytest.mark.parametrize("view", ["os", "ts"])
def test_basic_real_cli_uses_native_stage_costs_and_checkpoint_storage(
    algorithm, view, survey_script, tmp_path
):
    protocol = matrix(algorithm)
    inputs = tmp_path / "input"
    inputs.mkdir()
    make_splits(inputs)
    path = write_matrix(tmp_path, protocol)
    assert load_protocol(path) == protocol
    estimate = estimate_checkpoint_bytes(protocol, input_dims={"spambase": 3})
    assert estimate["peak_bytes_per_run"] == 3 * 2**20
    assert estimate["guard_peak_bytes_per_run"] == 6 * 2**20
    output = tmp_path / "results"
    assert (
        survey_script.main(
            [
                str(inputs),
                "--protocol",
                str(path),
                "--dataset",
                "spambase",
                "--method",
                f"{algorithm}_recipe",
                "--seeds",
                "0",
                "--c",
                "0.5",
                "--class-prior",
                "0.4",
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
    candidate = manifest["candidate_runs"][0]
    receipt = candidate["staged_budget_receipt"]
    steps = 3 if view == "ts" and algorithm == "genpu" else 2
    assert receipt["optimizer_steps"] == steps * (13 if algorithm == "genpu" else 3)
    assert receipt["checkpoint_epoch_count"] == 3
    assert not receipt["formal_budget_approved"]
    assert (
        receipt["eligible_stages"]
        == protocol["budgets"]["explicit_stages"][FIELD]["eligible_stages"]
    )
    assert all(0 < size <= 2**20 for size in receipt["checkpoint_bytes"])
    assert len(candidate["epoch_checkpoints"]) == 3
    assert set(manifest["selection"]) == {"PA", "OA"}
    assert manifest["selection_checkpoint_scope"] == "independent_per_epoch"
    assert manifest["resources"]["checkpoint_disk_preflight"]["required_bytes"] == 6 * 2**20


@pytest.mark.parametrize("algorithm", ["genpu", "pulda"])
@pytest.mark.parametrize("seed", [0, 7])
def test_determ_staged_verifier_preserves_native_weights_and_training(seed, algorithm, tmp_path):
    torch = pytest.importorskip("torch")
    protocol = matrix(algorithm)
    row = protocol["execution_units"][0]
    model = assemble_model(protocol, row, 3, seed=seed, params={}, class_prior=0.4, device="cpu")
    plain = copy.deepcopy(model)
    features = np.random.RandomState(0).normal(size=(18, 3)).astype("float32")
    y = np.r_[np.ones(3, int), np.zeros(15, int)]
    plain.fit(features, y, class_prior=0.4, os_or_ts="ts")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        model, features, y, class_prior=0.4, os_or_ts="ts"
    )
    receipt = verify_trajectory(
        model, trajectory, protocol["budgets"][row["budget"]][FIELD], features, y, "ts"
    )
    assert receipt["checkpoint_epoch_count"] == 3
    assert model.history_ == plain.history_
    for key, tensor in plain.model_.state_dict().items():
        torch.testing.assert_close(tensor, model.model_.state_dict()[key], rtol=0, atol=0)
    np.testing.assert_array_equal(
        plain.decision_function(features),
        trajectory.checkpoints[-1].restore().decision_function(features),
    )


@pytest.mark.parametrize(
    "fault", ["schema", "missing_param", "bool", "count", "stages", "bytes", "approval"]
)
def test_param_bad_contracts_are_rejected(fault):
    contract = matrix("genpu")["budgets"]["explicit_stages"][FIELD]
    if fault == "schema":
        contract["schema_version"] = True
    elif fault == "missing_param":
        del contract["parameter_values"]["classifier_epochs"]
    elif fault == "bool":
        contract["parameter_values"]["batch_size"] = True
    elif fault == "count":
        contract["checkpoint_epoch_count"] = 2
    elif fault == "stages":
        contract["eligible_stages"] = ["gan"]
    elif fault == "bytes":
        contract["checkpoint_bytes_per_component"] = False
    else:
        contract["formal_budget_approved"] = True
    with pytest.raises(ValueError):
        validate_contract(contract)


@pytest.mark.parametrize(
    "fault",
    ["missing", "legacy", "wrong_algorithm", "unlocked", "unit_budget", "architecture", "cnn"],
)
def test_param_bad_matrix_budget_bindings_refuse_before_data(fault, tmp_path):
    protocol = matrix("genpu")
    profile = protocol["method_profiles"]["genpu_recipe"]
    if fault == "missing":
        protocol["budgets"]["explicit_stages"] = {"epochs": 2}
    elif fault == "legacy":
        protocol["budgets"]["explicit_stages"]["epochs"] = 2
    elif fault == "wrong_algorithm":
        profile["algorithm"] = "upu"
    elif fault == "unlocked":
        del profile["params"]["batch_size"]
    elif fault == "unit_budget":
        protocol["budgets"]["other"] = {"epochs": 2}
        protocol["execution_units"][0]["budget"] = "other"
    elif fault == "architecture":
        protocol["execution_units"][0]["model_family"] = "shared_mlp"
    else:
        protocol["execution_units"][0]["training_path"] = "native_cnn"
    with pytest.raises(ValueError):
        load_protocol(write_matrix(tmp_path, protocol))


@pytest.mark.parametrize("fault", ["stage", "steps", "coverage", "bytes", "paid_gan"])
def test_param_actual_trajectory_faults_are_not_accepted(fault, tmp_path):
    protocol = matrix("genpu")
    model = assemble_model(
        protocol,
        protocol["execution_units"][0],
        3,
        seed=0,
        params={},
        class_prior=0.4,
        device="cpu",
    )
    features = np.random.RandomState(0).normal(size=(18, 3)).astype("float32")
    y = np.r_[np.ones(3, int), np.zeros(15, int)]
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        model, features, y, class_prior=0.4, os_or_ts="os"
    )
    contract = protocol["budgets"]["explicit_stages"][FIELD]
    if fault == "stage":
        trajectory.checkpoints[0].training_context["stage"] = "gan"
    elif fault == "steps":
        trajectory.checkpoints[0].training_context["optimizer_steps"] = 1
    elif fault == "coverage":
        trajectory.checkpoints.pop()
    elif fault == "bytes":
        contract["checkpoint_bytes_per_component"] = 1
    else:
        model.stage_optimizer_steps_["positive_generator"] = 0
    with pytest.raises(ValueError):
        verify_trajectory(model, trajectory, contract, features, y, "os")


def test_determ_legacy_budget_and_storage_are_unchanged():
    protocol = load_protocol()
    row = next(
        row
        for row in protocol["execution_units"]
        if row["method"] == "nnpu" and row["dataset"] == "spambase"
    )
    assert checkpoint_epochs(protocol["budgets"][row["budget"]]) == 200
    assert unit_checkpoint_bytes(protocol, row, input_dim=3) > 0


def test_edge_zero_warmup_keeps_only_pu_snapshots_without_phantom_cost(tmp_path):
    protocol = matrix("pulda")
    profile = protocol["method_profiles"]["pulda_recipe"]
    contract = protocol["budgets"]["explicit_stages"][FIELD]
    profile["params"]["warmup_epochs"] = 0
    contract["parameter_values"]["warmup_epochs"] = 0
    contract["checkpoint_epoch_count"] = 2
    contract["eligible_stages"] = ["pu_mixup"]
    assert load_protocol(write_matrix(tmp_path, protocol)) == protocol
    model = assemble_model(
        protocol,
        protocol["execution_units"][0],
        3,
        seed=0,
        params={},
        class_prior=0.4,
        device="cpu",
    )
    features = np.random.RandomState(0).normal(size=(18, 3)).astype("float32")
    labels = np.r_[np.ones(3, int), np.zeros(15, int)]
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        model, features, labels, class_prior=0.4, os_or_ts="ts"
    )
    receipt = verify_trajectory(model, trajectory, contract, features, labels, "ts")
    assert receipt["checkpoint_epoch_count"] == 2 and receipt["optimizer_steps"] == 4
    assert receipt["stage_optimizer_steps"] == {"warmup": 0, "pu_mixup": 4}


def _model_with_encoder(algorithm):
    from torch import nn

    protocol = matrix(algorithm)
    row = protocol["execution_units"][0]
    model = assemble_model(
        protocol, row, 4, seed=0, params={}, class_prior=0.4, device="cpu", encoder=None
    )
    model.set_params(encoder=nn.Identity())
    return model, protocol["budgets"]["explicit_stages"][FIELD]


@pytest.mark.parametrize("algorithm", ["genpu", "pulda"])
def test_edge_native_2d_still_rejects_an_injected_encoder(algorithm):
    model, contract = _model_with_encoder(algorithm)

    with pytest.raises(ValueError, match="without injected modules"):
        validate_estimator(model, contract)


@pytest.mark.parametrize("algorithm", ["genpu", "pulda"])
def test_basic_native_cnn_accepts_its_bound_encoder(algorithm):
    model, contract = _model_with_encoder(algorithm)

    validate_estimator(model, contract, architecture="cnn")


@pytest.mark.parametrize("algorithm", ["genpu", "pulda"])
def test_edge_native_cnn_without_the_encoder_is_refused(algorithm):
    model, contract = _model_with_encoder(algorithm)
    model.set_params(encoder=None)

    with pytest.raises(ValueError, match="bound encoder"):
        validate_estimator(model, contract, architecture="cnn")


def test_edge_native_cnn_still_rejects_any_other_injected_module():
    from torch import nn

    model, contract = _model_with_encoder("pulda")
    model.set_params(train_transform=nn.Identity())

    with pytest.raises(ValueError, match="without injected modules"):
        validate_estimator(model, contract, architecture="cnn")
