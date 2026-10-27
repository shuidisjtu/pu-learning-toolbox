"""Pool bindings through real runner, opt-in CLI, pilot planning and resume."""

# ruff: noqa: F811

import json

import pytest
from _survey_pilot_helpers import driver, splits_tree, unit_calls  # noqa: F401
from _survey_recipe_registry_helpers import locked_registry, pool_registry
from _survey_script_helpers import make_splits, survey_script  # noqa: F401
from _survey_summary_helpers import config_for, summary_cli  # noqa: F401

from pu_toolbox.experiment.pilot_plan import completed_runs, manifest_resume_key
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA
from pu_toolbox.experiment.survey_execution import assemble_model
from pu_toolbox.experiment.survey_protocol import digest, load_protocol, resolve_unit
from pu_toolbox.experiment.survey_recipe_binding import (
    SCHEMA_VERSION,
    bind_recipe_pool,
    matches_recipe_binding,
)
from pu_toolbox.experiment.survey_recipe_registry import canonical_digest
from pu_toolbox.experiment.tracking import SelectionArtifact

pytestmark = pytest.mark.unit


def _request(tmp_path, survey_script):
    make_splits(tmp_path)
    protocol, registry = pool_registry()
    path = tmp_path / "synthetic-protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    row = resolve_unit(protocol, "spambase", "upu", "native_2d")
    model = assemble_model(protocol, row, 3, seed=0, params={}, class_prior=0.3, device="cpu")
    config = {
        "architecture": "mlp",
        "c": 0.5,
        "c_requested_token": "0.5",
        "split_ref": {"dataset": "spambase", "seed": 0},
        "candidates": protocol["candidate_pool"],
        "recipe_registry": registry,
        "recipe_binding_schema_version": SCHEMA_VERSION,
        "survey_protocol": {
            "path": str(path),
            "dataset": "spambase",
            "method": "upu",
            "training_path": "native_2d",
            "mechanism": "scar",
            "seeds": [0],
            "c_tokens": ["0.5"],
        },
    }
    expected = bind_recipe_pool(
        registry, protocol, method="upu", dataset="spambase", training_path="native_2d"
    )
    expected.update(protocol_version=protocol["protocol_version"], protocol_sha256=digest(protocol))
    return model, survey_script.load_split_parts(tmp_path), config, expected


def test_basic_runner_tracks_distinct_pa_oa_after_exclusion_and_retry(
    survey_script, tmp_path, monkeypatch
):
    model, parts, config, expected = _request(tmp_path, survey_script)
    pa, oa = ProtocolPA(), ProtocolOA()
    monkeypatch.setattr(
        pa, "select", lambda *args, **kwargs: SelectionArtifact("PA", 0, None, None, {})
    )
    monkeypatch.setattr(
        oa, "select", lambda *args, **kwargs: SelectionArtifact("OA", 1, None, None, {})
    )
    runner = ExperimentRunner(class_prior=0.3, config=config, protocols=[pa, oa])
    train = runner._train
    calls = 0

    def controlled_train(*args):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise ValueError("synthetic failure to exercise compressed trajectory indices")
        return train(*args)

    monkeypatch.setattr(runner, "_train", controlled_train)
    payload = runner.fit(model, *parts).manifest
    assert [record["status"] for record in payload["candidate_runs"]] == [
        "excluded",
        "recovered",
        "succeeded",
    ]
    assert payload["selection"]["PA"]["candidate_index"] == 1
    assert payload["selection"]["OA"]["candidate_index"] == 2
    assert payload["selection"]["PA"]["run_index"] == 0
    assert payload["selection"]["OA"]["run_index"] == 1
    assert matches_recipe_binding(payload, expected)
    # A synthetic wider protocol is not promoted to the frozen Pilot leaderboard.
    assert not payload["formal_eligible"]


@pytest.mark.parametrize("defect", ["draft", "schema", "scalar", "no_registry"])
def test_param_bad_pool_preflight_precedes_label_generation(
    survey_script, tmp_path, monkeypatch, defect
):
    model, parts, config, _ = _request(tmp_path, survey_script)
    if defect == "draft":
        config["recipe_registry"]["recipe_lifecycle"] = "draft"
    elif defect == "schema":
        config["recipe_binding_schema_version"] = "unknown"
    elif defect == "scalar":
        config.pop("recipe_binding_schema_version")
    else:
        config.pop("recipe_registry")
    runner = ExperimentRunner(class_prior=0.3, config=config)

    def forbidden(*args, **kwargs):
        pytest.fail("label generation must follow successful pool binding")

    monkeypatch.setattr(runner.generator, "generate", forbidden)
    with pytest.raises(ValueError):
        runner.fit(model, *parts)


def test_edge_completed_runs_requires_exact_pool_and_trajectory_joins(survey_script, tmp_path):
    model, parts, config, expected = _request(tmp_path, survey_script)
    payload = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    # The CLI stamps the run token; the programmatic runner is not the CLI.
    payload["c_requested_token"] = "0.5"
    path = tmp_path / "result/manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps(payload), encoding="utf-8")
    identity = manifest_resume_key(payload)
    assert identity is not None
    # Resume expectations use the no-view identity; the view remains the result key.
    key = identity[:6]
    assert len(completed_runs(tmp_path / "result", expected_recipe_bindings={key: expected})) == 1
    payload["selection"]["OA"]["recipe_candidate_id"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert completed_runs(tmp_path / "result", expected_recipe_bindings={key: expected}) == {}


def test_basic_cli_single_pool_has_a_separate_namespace(survey_script, summary_cli, tmp_path):
    source, out = tmp_path / "splits", tmp_path / "results"
    source.mkdir()
    make_splits(source)
    registry = locked_registry()
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    args = [
        str(source),
        "--protocol",
        "survey-v2.0",
        "--dataset",
        "spambase",
        "--method",
        "upu",
        "--c",
        "0.5",
        "--seeds",
        "0",
        "--class-prior",
        "0.3",
        "--out-dir",
        str(out),
        "--recipe-registry",
        str(path),
    ]
    assert survey_script.main(args) == 0
    scalar_path = next((out / f"recipe_{canonical_digest(registry)}").rglob("manifest.json"))
    before = scalar_path.read_bytes()
    assert survey_script.main([*args, "--recipe-binding-schema-version", SCHEMA_VERSION]) == 0
    pool_path = next((out / f"recipe_pool_{canonical_digest(registry)}").rglob("manifest.json"))
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    expected = bind_recipe_pool(
        registry, load_protocol(), method="upu", dataset="spambase", training_path="native_2d"
    )
    assert matches_recipe_binding(payload, expected)
    assert scalar_path.read_bytes() == before
    config = config_for({"B1": (out / f"recipe_pool_{canonical_digest(registry)}", 1)})
    config["schema_version"] = "survey-batch-roots-2"
    config["batches"][0].update(
        recipe_policy="registry_bound",
        recipe_registry=str(path),
        recipe_registry_sha256=canonical_digest(registry),
        recipe_binding_schema_version=SCHEMA_VERSION,
    )
    report = summary_cli.build_summary(config)
    assert report["recipe_audit"]["overall"] == "pass"
    # 17 analytic checks plus E01, the batch-evidence citation check (D52).
    assert len(report["checks"]) == 18
    assert len(report["rows"]) == 2
    assert all(row["recipe_selection_bindings"] for row in report["rows"])


def test_edge_schema_flag_without_registry_is_rejected_before_data_reads(
    survey_script, driver, tmp_path
):
    assert (
        survey_script.main(
            [str(tmp_path / "absent"), "--recipe-binding-schema-version", SCHEMA_VERSION]
        )
        == 1
    )
    assert driver.main(["--recipe-binding-schema-version", SCHEMA_VERSION]) == 1


def test_basic_determ_pilot_plan_and_child_flags_keep_pool_schema(driver, unit_calls, tmp_path):
    splits_tree(tmp_path / "splits", datasets=("spambase",), prior=0.3)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(locked_registry()), encoding="utf-8")
    plan = tmp_path / "plan.json"
    args = [
        "--splits",
        str(tmp_path / "splits"),
        "--results",
        str(tmp_path / "results"),
        "--datasets",
        "spambase",
        "--methods",
        "upu",
        "--training-paths",
        "native_2d",
        "--recipe-registry",
        str(registry),
        "--recipe-binding-schema-version",
        SCHEMA_VERSION,
    ]
    assert driver.main([*args, "--dry-run", "--plan-json", str(plan)]) == 0
    before = plan.read_bytes()
    assert driver.main([*args, "--dry-run", "--plan-json", str(plan)]) == 0
    assert plan.read_bytes() == before
    payload = json.loads(before)
    assert all(
        binding["recipe_binding_schema_version"] == SCHEMA_VERSION
        for binding in payload["recipe_bindings"]
    )
    assert driver.main(args) == 1  # Stub children do not deliver results.
    assert unit_calls
    for argv in unit_calls:
        assert argv[argv.index("--recipe-binding-schema-version") + 1] == SCHEMA_VERSION
