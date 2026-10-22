"""Refuse ambiguous protocols and protect support/checkpoint identities."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from _support_checkpoint_helpers import adapter, inputs, model, run  # noqa: E402

from pu_toolbox.experiment.bundle import DatasetBundle  # noqa: E402
from pu_toolbox.experiment.strategies import (  # noqa: E402
    ProtocolOA,
    ProtocolPA,
    ProtocolPACleanSupport,
)
from pu_toolbox.experiment.tracking import RunTrajectory  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
@pytest.mark.parametrize(
    "protocols",
    [
        [ProtocolPA()],
        [ProtocolPA(), ProtocolOA()],
        [ProtocolPACleanSupport(), ProtocolPACleanSupport()],
    ],
)
def test_param_plain_pa_or_duplicate_protocol_names_refused(method, protocols, tmp_path):
    with pytest.raises(ValueError, match="unique OA / PA"):
        run(method, tmp_path, protocols=protocols)
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
@pytest.mark.parametrize("role", [0, 1, 2, 3])
def test_edge_role_overlap_remains_refused_before_writing(method, role, tmp_path):
    parts, support = inputs()
    support.indices[0] = parts[role].indices[0]
    with pytest.raises(ValueError, match="overlap"):
        run(method, tmp_path, parts=parts, support=support)
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize(
    "override", [{"require_validation": False}, {"distillation_scope": "untrusted_u"}]
)
def test_param_full_meta_candidate_cannot_fall_back_to_ablation(override, tmp_path):
    with pytest.raises(ValueError, match="full-meta"):
        run("self_pu", tmp_path, candidates=[override])


def test_basic_named_pa_requires_support_disclosure_and_keeps_clean_view_guard():
    parts, _ = inputs()
    with pytest.raises(ValueError, match="audited support trajectory"):
        ProtocolPACleanSupport().select([], parts[1], class_prior=0.5)
    untagged = RunTrajectory([], SimpleNamespace())
    with pytest.raises(ValueError, match="audited support trajectory"):
        ProtocolPACleanSupport().select([untagged], parts[1], class_prior=0.5)
    tagged = RunTrajectory(
        [], SimpleNamespace(training_provenance_={"pa_clean_support_eligible": True})
    )
    with pytest.raises(ValueError, match="never clean labels"):
        ProtocolPACleanSupport().select([tagged], parts[2], class_prior=0.5)


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
def test_edge_support_mutation_after_preflight_revokes_fit(method, tmp_path):
    parts, support = inputs()
    trainer = adapter(method, support)
    fitted = model(method)
    trainer.validate_experiment(
        fitted, DatasetBundle(*parts), {"checkpoint_bytes_per_component": 65536}, [ProtocolOA()]
    )
    trainer.bind_checkpoint_directory(tmp_path / "checkpoints")
    support.labels[:] = 1 - support.labels
    with pytest.raises(ValueError, match="changed after"):
        trainer.fit(fitted, parts[0].X, parts[0].labels, class_prior=0.5)
    assert not (tmp_path / "checkpoints").exists()


def test_determ_larger_candidate_budget_is_used_for_disk_estimate(tmp_path):
    result = run("lagam", tmp_path, candidates=[{"max_epochs": 2}, {"max_epochs": 4}])
    disk = result.manifest["resources"]["checkpoint_disk_preflight"]
    assert disk["required_bytes"] == 65536 * 4 * 1 * 2 * 2
    assert result.manifest["candidate_runs"][1]["training_provenance"]["captured_meta_epochs"] == [
        1,
        2,
        3,
    ]


def test_edge_no_runner_preflight_or_no_meta_epochs_cannot_fall_back_to_terminal():
    parts, support = inputs()
    trainer = adapter("lagam", support)
    fitted = model("lagam")
    trainer.validate_experiment(
        fitted, DatasetBundle(*parts), {"checkpoint_bytes_per_component": 65536}, [ProtocolOA()]
    )
    with pytest.raises(ValueError, match="runner checkpoint preflight"):
        trainer.fit(fitted, parts[0].X, parts[0].labels)
    with pytest.raises(ValueError, match="meta-active"):
        trainer.validate_experiment(
            fitted,
            DatasetBundle(*parts),
            {"checkpoint_bytes_per_component": 65536, "candidates": [{"warmup_epochs": 3}]},
            [ProtocolOA()],
        )
    assert not trainer._prepared


def test_edge_actual_checkpoint_larger_than_declared_budget_is_not_accepted(tmp_path):
    with pytest.raises(RuntimeError, match="all candidate runs failed"):
        run("lagam", tmp_path, config={"checkpoint_bytes_per_component": 1})
    import json

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert not manifest["selection"]
    assert len(manifest["failures"]) == 1
    failure = manifest["failures"][0]
    assert failure["attempts"] == 2
    assert len(failure["errors"]) == 2
    assert all("actual checkpoint exceeds" in item["message"] for item in failure["errors"])
    assert list((tmp_path / "checkpoints").rglob("*.pt"))
