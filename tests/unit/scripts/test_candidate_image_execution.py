"""Pinned image CLI plans and actual dual-protocol replay; no formal admission."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_candidate_image_experiment as script  # noqa: E402

pytestmark = pytest.mark.unit


def product(tmp_path):
    directory = tmp_path / "input"
    directory.mkdir()
    for i, role in enumerate(script.ROLES):
        np.savez(
            directory / f"{role}.npz",
            X=np.random.RandomState(i).randint(0, 256, (12, 3, 8, 8)).astype(np.uint8),
            y=np.arange(12) % 2,
            indices=np.arange(12) + i * 20,
        )
    return directory


def plan(directory, method="pulda", **kwargs):
    parameters = (
        dict(warmup_epochs=1, pu_epochs=2, positive_batch_size=2, unlabeled_batch_size=4)
        if method == "pulda"
        else dict(max_epochs=3, batch_size=4, lr_schedule="constant")
    ) | {"device": "cpu"}
    return script.build_plan(
        directory,
        **(
            dict(
                method=method,
                dataset="cifar10",
                prior=0.4,
                prior_source="synthetic population",
                seed=3,
                parameters=parameters,
                checkpoint_bytes=64 * 2**20,
                c_token="0.5",
            )
            | kwargs
        ),
    )


@pytest.mark.parametrize("method", ["pulda", "gradpu"])
def test_basic_actual_image_cli_retention_dual_models_and_replay(method, tmp_path):
    planned = plan(product(tmp_path), method)
    output = tmp_path / "results"
    summary = script.execute(planned, output)
    assert summary["accepted_engineering_run"] and summary["selected_models_replayed"]
    assert summary["warning_count"] == summary["retry_count"] == 0
    assert summary["checkpoint_peak_payload_bytes"] > summary["retained_checkpoint_bytes"] > 0
    assert 1 <= summary["retained_checkpoint_count"] <= 2
    assert not summary["formal_admission"] and not summary["paper_numerical_reproduction"]
    assert summary["source_default_snapshot_projection"]["epochs"] == (
        120 if method == "pulda" else 200
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["run_view"] == "ts-compatible" and not manifest["formal_eligible"]
    assert set(summary["selection"]) == {"PA", "OA"}
    assert all((output / f"selected_{name}.pkl").is_file() for name in ("PA", "OA"))
    with pytest.raises(FileExistsError):
        script.execute(planned, output)


def test_edge_tampered_plan_file_or_source_refused_before_output(tmp_path, monkeypatch):
    directory = product(tmp_path)
    planned = plan(directory)
    tampered = copy.deepcopy(planned)
    tampered["parameters_requested"]["pu_epochs"] += 1
    with pytest.raises(ValueError, match="plan changed"):
        script.execute(tampered, tmp_path / "bad")
    original = script.file_hash
    monkeypatch.setattr(
        script,
        "file_hash",
        lambda path: "changed" if str(path).endswith("uv.lock") else original(path),
    )
    with pytest.raises(ValueError, match="identity changed"):
        script.execute(planned, tmp_path / "bad")
    monkeypatch.setattr(script, "file_hash", original)
    with (directory / "test.npz").open("ab") as stream:
        stream.write(b"changed bytes")
    with pytest.raises(ValueError, match="identity changed"):
        script.execute(planned, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_determ_fixed_plan_and_groups_do_not_depend_on_replanning(tmp_path):
    directory = product(tmp_path)
    assert plan(directory) == plan(directory)
    assert plan(directory)["comparison_group"] != plan(directory, "gradpu")["comparison_group"]
    assert plan(directory, seed=7)["label_view_sha256"] != plan(directory)["label_view_sha256"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"checkpoint_bytes": True},
        {"c_token": "0.50"},
        {"prior": True},
        {"prior_source": ""},
        {"dataset": "unknown_dataset"},
        {"parameters": {"device": "cpu"}},
    ],
)
def test_param_invalid_or_ambiguous_plan_refused(tmp_path, kwargs):
    with pytest.raises(ValueError):
        plan(product(tmp_path), **kwargs)


def test_edge_output_cannot_overlap_any_input_ancestor(tmp_path):
    directory = product(tmp_path)
    planned = plan(directory)
    for output in (directory, tmp_path, directory / "nested"):
        with pytest.raises(ValueError, match="overlap"):
            script.execute(planned, output)


def test_edge_requested_cuda_unavailable_is_not_a_cpu_success(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    parameters = dict(device="cuda", max_epochs=1, batch_size=4, lr_schedule="constant")
    with pytest.raises(ValueError, match="no CPU fallback"):
        plan(product(tmp_path), "gradpu", parameters=parameters)
