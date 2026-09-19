# ruff: noqa: N803, N806, F811
"""Pre-run disk capacity guard for checkpoint-capturing runs.

A formal pilot writes per-epoch checkpoints for every candidate and does not
delete earlier artifacts, so a full run needs tens of gigabytes before it
starts.  Nothing measured disk usage and nothing refused a run that could not
fit, so the failure mode was a full disk partway through a batch.

The guard is graded by execution mode: a versioned pilot is refused outright,
while a technical smoke run is only warned about.  The mode already says
whether the result counts, so no extra override flag is introduced.

Everything here is a pure function over injected facts; no test depends on how
much space the host happens to have.
"""

import json

import numpy as np
import pytest
import torch
from _survey_script_helpers import make_splits, survey_script  # noqa: F401
from sklearn.base import BaseEstimator
from torch import nn

from pu_toolbox.experiment import resources
from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.survey_execution import assemble_model
from pu_toolbox.experiment.survey_protocol import (
    PROTOCOL_PATH,
    RESNET18_COMPONENT_BYTES,
    load_protocol,
    resolve_unit,
    unit_checkpoint_bytes,
)

pytestmark = pytest.mark.unit


class _PerEpochNetwork(BaseEstimator):
    """A minimal estimator whose fit advertises per-epoch checkpoints."""

    epoch_components = ("model",)

    def __init__(self, max_epochs=2):
        self.max_epochs = max_epochs

    def fit(self, X, y, *, epoch_callback=None):
        self.model_ = nn.Linear(np.shape(X)[1], 1)
        for epoch in range(self.max_epochs):
            if epoch_callback is not None:
                epoch_callback(epoch, self)
        return self

    def decision_function(self, X):
        with torch.no_grad():
            return self.model_(torch.as_tensor(X, dtype=torch.float32)).reshape(-1).numpy()

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


def _bundle(n=8):
    rng = np.random.RandomState(0)
    return DatasetBundle(
        **{
            role: DatasetPart(
                X=rng.normal(size=(n, 3)).astype(np.float32),
                labels=np.array([1, 1] + [0] * (n - 2)),
                indices=np.arange(index * n, index * n + n),
                view="clean",
                for_selection=role != "test",
            )
            for index, role in enumerate(("train", "pu_val", "clean_val", "test"))
        }
    )


def _smoke_run(tmp_path, model, *, config):
    runner = ExperimentRunner(config=config, manifest_path=str(tmp_path / "manifest.json"))
    bundle = _bundle()
    runner.fit(model, bundle.train, bundle.pu_val, bundle.clean_val, bundle.test)
    return json.loads((tmp_path / "manifest.json").read_text())


def _bound_self_pu(survey_script, tmp_path):  # noqa: F811
    make_splits(tmp_path)
    parts = survey_script.load_split_parts(tmp_path)
    protocol = load_protocol()
    row = resolve_unit(protocol, "spambase", "self_pu")
    model = assemble_model(protocol, row, 3, seed=0, params={}, class_prior=0.3, device="cpu")
    config = {
        "architecture": "mlp",
        "c": 0.5,
        "c_requested_token": "0.5",
        "split_ref": {"dataset": "spambase", "seed": 0},
        "checkpoint_bytes_per_component": 1024,
        "survey_protocol": {
            "path": str(PROTOCOL_PATH),
            "dataset": "spambase",
            "method": "self_pu",
            "mechanism": "scar",
            "seeds": [0],
            "c_tokens": ["0.5"],
        },
    }
    return model, parts, config


# --- the arithmetic and the payload ----------------------------------------


def test_param_disk_requirement_arithmetic_and_input_validation():
    assert (
        resources.checkpoint_disk_requirement(
            bytes_per_component=100, epochs=3, components=2, candidates=4, attempts=1
        )
        == 2400
    )
    assert (
        resources.checkpoint_disk_requirement(
            bytes_per_component=100, epochs=3, components=2, candidates=1, attempts=2
        )
        == 1200
    )
    fields = {
        "bytes_per_component": 1,
        "epochs": 1,
        "components": 1,
        "candidates": 1,
        "attempts": 1,
    }
    for field in fields:
        for bad in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match="positive integer"):
                resources.checkpoint_disk_requirement(**{**fields, field: bad})


def test_edge_disk_preflight_ready_boundary_at_exact_free_space(tmp_path):
    exact = resources.disk_space_preflight(required_bytes=100, directory=tmp_path, free_bytes=100)
    assert exact["ready"] is True
    assert exact["missing_bytes"] == 0
    assert exact["free_bytes"] == 100

    short = resources.disk_space_preflight(required_bytes=100, directory=tmp_path, free_bytes=99)
    assert short["ready"] is False
    assert short["missing_bytes"] == 1
    assert short["required_bytes"] == 100
    assert short["directory"] == str(tmp_path)


def test_edge_disk_free_bytes_walks_to_an_existing_ancestor(tmp_path):
    """The checkpoint directory does not exist yet when the guard runs."""
    assert resources.disk_free_bytes(tmp_path / "not" / "created" / "yet") > 0


def test_determ_disk_preflight_payload_is_host_independent(tmp_path):
    first = resources.disk_space_preflight(required_bytes=5, directory=tmp_path, free_bytes=10)
    second = resources.disk_space_preflight(required_bytes=5, directory=tmp_path, free_bytes=10)
    assert first == second


def test_edge_unit_checkpoint_bytes_per_row_family():
    protocol = load_protocol()

    mlp = resolve_unit(protocol, "spambase", "self_pu")
    assert unit_checkpoint_bytes(protocol, mlp, input_dim=57) == 4 * (128 * (57 + 2) + 1)

    image = resolve_unit(protocol, "cifar10", "self_pu")
    assert unit_checkpoint_bytes(protocol, image, input_dim=3) == RESNET18_COMPONENT_BYTES

    closed_form = resolve_unit(protocol, "spambase", "upu")
    assert unit_checkpoint_bytes(protocol, closed_form, input_dim=57) is None


# --- the graded guard -------------------------------------------------------


def test_param_versioned_pilot_insufficient_disk_writes_rejection_manifest(
    tmp_path, monkeypatch, survey_script
):  # noqa: F811
    model, parts, config = _bound_self_pu(survey_script, tmp_path)
    manifest_path = tmp_path / "manifest.json"
    runner = ExperimentRunner(config=config, manifest_path=str(manifest_path))
    monkeypatch.setattr(resources, "disk_free_bytes", lambda directory: 0)

    with pytest.raises(ValueError, match="disk"):
        runner.fit(model, *parts)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["execution_mode"] == "rejected_versioned_pilot"
    assert manifest["failures"][0]["stage"] == "protocol_preflight"
    payload = manifest["resources"]["checkpoint_disk_preflight"]
    assert payload["ready"] is False
    assert payload["free_bytes"] == 0
    assert payload["required_bytes"] > 0
    assert not (tmp_path / "checkpoints").exists()


def test_basic_technical_smoke_insufficient_disk_warns_and_trains(tmp_path, monkeypatch):
    """The same shortage only warns when the run is not a formal pilot."""
    monkeypatch.setattr(resources, "disk_free_bytes", lambda directory: 0)
    with pytest.warns(UserWarning, match="disk"):
        manifest = _smoke_run(
            tmp_path, _PerEpochNetwork(), config={"c": 0.5, "checkpoint_bytes_per_component": 1024}
        )
    assert manifest["resources"]["checkpoint_disk_preflight"]["ready"] is False
    assert manifest["failures"] == []


def test_edge_unknown_component_size_warns_instead_of_refusing(tmp_path, monkeypatch):
    """An unsizable run is not a run that cannot fit.

    The guard protects the formal pilot path, which always knows the size from
    the protocol.  Refusing when the size is merely unknown would block custom
    estimators for no safety gain.
    """
    monkeypatch.setattr(resources, "disk_free_bytes", lambda directory: 0)
    with pytest.warns(UserWarning, match="disk"):
        manifest = _smoke_run(tmp_path, _PerEpochNetwork(), config={"c": 0.5})
    assert "checkpoint_disk_preflight" not in manifest["resources"]
    assert manifest["failures"] == []
