# ruff: noqa: N803, F811
"""Real ResNet CPU/GPU smoke and encoder provenance gates (not paper evidence)."""

import json

import numpy as np
import pytest
import torch
from _survey_script_helpers import survey_script  # noqa: F401

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.survey_execution import assemble_model, prepare_image_bundle
from pu_toolbox.experiment.survey_protocol import ROLES, load_protocol, resolve_unit

pytestmark = pytest.mark.unit


def _bound(tmp_path, device="cpu"):
    rng = np.random.RandomState(8)
    source = DatasetBundle(
        **{
            role: DatasetPart(
                X=rng.randint(0, 256, (8, 3, 32, 32), dtype=np.uint8),
                labels=np.array([1, 0] * 4),
                indices=np.arange(index * 8, (index + 1) * 8),
                view="clean",
                for_selection=role != "test",
            )
            for index, role in enumerate(ROLES)
        }
    )
    protocol = load_protocol()
    # Reduce only the technical fixture budget; this must mark a protocol deviation.
    protocol["method_profiles"]["nnpu"]["params"]["max_epochs"] = 1
    protocol["budgets"]["minibatch"]["epochs"] = 1
    path = tmp_path / "smoke_protocol.json"
    path.write_text(json.dumps(protocol))
    prepared, encoder, image = prepare_image_bundle(source, protocol, 0)
    row = resolve_unit(protocol, "cifar10", "nnpu")
    model = assemble_model(
        protocol, row, 3, seed=0, params={}, class_prior=0.3, device=device, encoder=encoder
    )
    config = {
        "architecture": "cnn",
        "c": 0.5,
        "c_requested_token": "0.5",
        "image_manifest": image,
        "split_ref": {"dataset": "cifar10", "seed": 0},
        "survey_protocol": {
            "path": str(path),
            "dataset": "cifar10",
            "method": "nnpu",
            "training_path": "native_cnn",
            "mechanism": "scar",
            "seeds": [0],
            "c_tokens": ["0.5"],
        },
    }
    return model, prepared, config


def test_basic_real_resnet_native_cnn_cpu_smoke(tmp_path):
    model, bundle, config = _bound(tmp_path)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = ExperimentRunner(
            class_prior=0.3, config=config, manifest_path=str(tmp_path / "manifest.json")
        ).fit(model, *(getattr(bundle, role) for role in ROLES))
    finally:
        torch.set_num_threads(before)
    assert result.manifest["training_path"] == "native_cnn"
    assert set(result.test_metrics) == {"PA", "OA"}
    assert result.manifest["budget"]["epochs"] == 1
    assert result.manifest["formal_eligible"] is False
    assert "noncanonical_protocol_matrix" in result.manifest["protocol_deviation"]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_basic_real_resnet_native_cnn_gpu_smoke(tmp_path, monkeypatch):
    model, bundle, config = _bound(tmp_path, device="cuda")
    fitted_devices = []
    original_fit = type(model).fit

    def checked_fit(estimator, *args, **kwargs):
        fitted = original_fit(estimator, *args, **kwargs)
        fitted_devices.extend(parameter.device.type for parameter in fitted.model_.parameters())
        assert len(fitted.history_["epoch"]) == 1
        assert np.isfinite(fitted.decision_function(bundle.test.X)).all()
        return fitted

    monkeypatch.setattr(type(model), "fit", checked_fit)
    result = ExperimentRunner(
        class_prior=0.3, config=config, manifest_path=str(tmp_path / "gpu_manifest.json")
    ).fit(model, *(getattr(bundle, role) for role in ROLES))
    assert fitted_devices and set(fitted_devices) == {"cuda"}
    assert result.manifest["training_path"] == "native_cnn"
    assert result.manifest["budget"]["epochs"] == 1
    assert set(result.test_metrics) == {"PA", "OA"}
    assert result.manifest["formal_eligible"] is False
    assert "noncanonical_protocol_matrix" in result.manifest["protocol_deviation"]
    assert json.loads((tmp_path / "gpu_manifest.json").read_text())["formal_eligible"] is False


def test_param_mutated_encoder_rejected_before_training(tmp_path):
    model, bundle, config = _bound(tmp_path)
    with torch.no_grad():
        next(model.encoder.parameters()).add_(1)
    with pytest.raises(ValueError, match="encoder state"):
        ExperimentRunner(class_prior=0.3, config=config).fit(
            model, *(getattr(bundle, role) for role in ROLES)
        )


def test_edge_missing_image_manifest_never_acquires_cnn_claim(tmp_path):
    model, bundle, config = _bound(tmp_path)
    config.pop("image_manifest")
    with pytest.raises(ValueError, match="preprocessing provenance"):
        ExperimentRunner(class_prior=0.3, config=config).fit(
            model, *(getattr(bundle, role) for role in ROLES)
        )


def test_determ_same_seed_has_same_bound_initial_encoder(tmp_path):
    first, _, left = _bound(tmp_path)
    second, _, right = _bound(tmp_path)
    assert left["image_manifest"] == right["image_manifest"]
    assert type(first.encoder) is type(second.encoder)


def test_param_declared_budget_cannot_misreport_actual_epochs(tmp_path):
    model, bundle, config = _bound(tmp_path)
    path = config["survey_protocol"]["path"]
    protocol = load_protocol(path)
    protocol["budgets"]["minibatch"]["epochs"] = 2
    from pathlib import Path

    Path(path).write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="epoch budget"):
        ExperimentRunner(class_prior=0.3, config=config).fit(
            model, *(getattr(bundle, role) for role in ROLES)
        )


def test_edge_all_candidate_failures_keep_protocol_context(tmp_path):
    model, bundle, config = _bound(tmp_path)
    config["candidates"] = [{"unknown_parameter": 1}]
    path = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="all candidate runs failed"):
        ExperimentRunner(class_prior=0.3, config=config, manifest_path=str(path)).fit(
            model, *(getattr(bundle, role) for role in ROLES)
        )
    payload = json.loads(path.read_text())
    assert payload["protocol_version"] == "survey-v1.1"
    assert payload["budget"]["epochs"] == 1
    assert payload["candidate_runs"][0]["attempts"] == 2
    assert payload["formal_eligible"] is False
    assert "estimator_parameters" in payload
