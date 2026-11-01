"""Offload is explicit execution metadata, not a frozen recipe override."""

# ruff: noqa: F811

import json

import pytest
from _survey_script_helpers import make_sar_splits, survey_script  # noqa: F401
from test_survey_backlog import small_protocol

pytestmark = pytest.mark.unit


def test_determ_recipe_binding_keeps_execution_policy_separate_and_rejects_hidden_override(
    tmp_path,
):
    from pu_toolbox import GradPUClassifier
    from pu_toolbox.experiment.runner import _recipe_context
    from pu_toolbox.experiment.survey_protocol import digest, load_protocol

    source, registry = small_protocol(tmp_path, "gradpu")
    protocol = load_protocol(source)
    model = GradPUClassifier(**protocol["method_profiles"]["gradpu"]["params"])
    context = dict(
        execution_mode="versioned_pilot",
        protocol_sha256=digest(protocol),
        training_path="native_2d",
        runtime_parameters={},
    )
    config = dict(
        recipe_registry=json.loads(registry.read_text()),
        survey_protocol=dict(
            path=str(source), method="gradpu", dataset="spambase", constructor_overrides={}
        ),
    )
    baseline = _recipe_context(model, config, context)
    model.set_params(activation_memory="cpu_offload")
    with pytest.raises(ValueError, match="outside the resolved"):
        _recipe_context(model, config, context)
    config["gradpu_activation_memory"] = "device"
    with pytest.raises(ValueError, match="activation memory"):
        _recipe_context(model, config, context)
    config["gradpu_activation_memory"] = "cpu_offload"
    assert _recipe_context(model, config, context) == baseline
    assert _recipe_context(model, config, context) == baseline


@pytest.mark.parametrize("memory", ["device", "cpu_offload"])
def test_basic_bound_cli_records_memory_policy_and_keeps_full_budget(
    tmp_path, survey_script, memory, capsys
):
    data = tmp_path / "data"
    data.mkdir()
    make_sar_splits(data)
    (data / "split_manifest.json").write_text(
        json.dumps({"dataset": "spambase", "seed": 0, "class_prior": {"population": 0.4}})
    )
    source, registry = small_protocol(tmp_path, "gradpu")
    output = tmp_path / "out"
    code = survey_script.main(
        [
            str(data),
            "--method",
            "gradpu",
            "--protocol",
            str(source),
            "--dataset",
            "spambase",
            "--recipe-registry",
            str(registry),
            "--device",
            "cpu",
            "--c",
            "0.5",
            "--class-prior",
            "0.4",
            "--out-dir",
            str(output),
            "--gradpu-activation-memory",
            memory,
            "--reclaim-unselected-checkpoints",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err + captured.out
    paths = list(output.rglob("manifest.json"))
    assert len(paths) == 1
    assert ("execution_cpu_offload" in paths[0].parts) == (memory == "cpu_offload")
    manifest = json.loads(paths[0].read_text())
    assert not manifest["failures"] and not manifest["formal_eligible"]
    assert manifest["recipe_binding_status"] == "registry_bound"
    assert set(manifest["test_results"]) == {"PA", "OA"}
    candidate = manifest["candidate_runs"][0]
    assert len(candidate["epoch_checkpoints"]) == 2
    facts = candidate["training_provenance"]
    assert facts["activation_memory"]["policy"] == memory
    assert not facts["activation_memory"]["effective_batch_changed"]
    assert facts["batch"]["configured_per_branch_size"] == 8
    assert facts["batch"]["epochs"] == 2


@pytest.mark.parametrize(
    "extra", [[], ["--protocol", "survey-v3.1", "--method", "vpu", "--dataset", "adni"]]
)
def test_param_edge_offload_wrong_scope_rejected_before_loading_data(
    tmp_path, survey_script, extra, capsys
):
    output = tmp_path / "out"
    code = survey_script.main(
        [
            str(tmp_path / "missing"),
            "--gradpu-activation-memory",
            "cpu_offload",
            "--out-dir",
            str(output),
            *extra,
        ]
    )
    assert code == 1
    assert "offload requires" in capsys.readouterr().err
    assert not output.exists()
