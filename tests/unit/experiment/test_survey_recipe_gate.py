"""The lower-level fairness gate cannot bypass external recipe obligations."""

# ruff: noqa: F811

import copy
import json

import pytest
from _survey_recipe_registry_helpers import locked_registry
from _survey_script_helpers import make_splits, survey_script  # noqa: F401
from _survey_summary_helpers import (  # noqa: F401  # noqa: F401
    audit_cli,
    config_for,
    manifest,
    summary_cli,
    write_tree,
)

from pu_toolbox.experiment.survey_protocol import digest, load_protocol
from pu_toolbox.experiment.survey_recipe_audit import validate_consumed_bindings
from pu_toolbox.experiment.survey_recipe_registry import bind_recipe_candidate, canonical_digest

pytestmark = pytest.mark.unit


def _payload_and_binding():
    protocol = load_protocol()
    binding = bind_recipe_candidate(
        locked_registry("nnpu"),
        protocol,
        method="nnpu",
        dataset="spambase",
        training_path="native_2d",
    )
    binding.update(protocol_version=protocol["protocol_version"], protocol_sha256=digest(protocol))
    payload = manifest()
    payload.update(binding)
    payload["candidate_runs"][0]["params"] = protocol["candidate_pool"][0]
    for child in [*payload["candidate_runs"], *payload["selection"].values()]:
        child.update(binding)
    return payload, {"nnpu": binding}


def test_basic_audited_identities_pass_and_legacy_stays_compatible():
    payload, expected = _payload_and_binding()
    validate_consumed_bindings([payload], expected, protocol=load_protocol())
    validate_consumed_bindings([manifest()], None, protocol=load_protocol())


@pytest.mark.parametrize(
    "change", ["no_expectation", "empty", "partial", "erased", "mixed", "params"]
)
def test_param_unverified_or_changed_binding_rejected(change):
    payload, expected = _payload_and_binding()
    if change == "no_expectation":
        expected = None
    elif change == "empty":
        expected = {}
    elif change == "partial":
        expected["nnpu"].pop("protocol_sha256")
    elif change == "erased":
        payload = manifest()
    elif change == "mixed":
        expected["upu"] = dict(expected["nnpu"], recipe_registry_sha256="a" * 64)
    else:
        payload["candidate_runs"][0]["params"] = {"invented": True}
    with pytest.raises(ValueError):
        validate_consumed_bindings([payload], expected, protocol=load_protocol())


@pytest.mark.parametrize("diagnostic", [False, True])
def test_edge_standalone_cli_refuses_bound_artifacts_without_batch_contract(
    audit_cli, tmp_path, diagnostic
):
    payload, _ = _payload_and_binding()
    root = write_tree(tmp_path / "bound", {"run": payload})
    args = [str(root), "--json"] + (["--diagnostic"] if diagnostic else [])
    assert audit_cli.gate.main(args) == 1


def test_edge_aggregate_checks_before_fairness(audit_cli, tmp_path, monkeypatch):
    payload, expected = _payload_and_binding()
    payload["selection"]["OA"].pop("inner_search_id")
    root = write_tree(tmp_path / "bound", {"run": payload})

    def forbidden(*args, **kwargs):
        raise AssertionError("must refuse before fairness or metric consumption")

    monkeypatch.setattr(audit_cli.gate, "_group_report", forbidden)
    with pytest.raises(ValueError, match="externally audited"):
        audit_cli.gate.aggregate(
            [root / "run/manifest.json"],
            protocol=load_protocol(),
            expected_recipe_bindings=expected,
        )


def test_determ_inputs_remain_unchanged():
    payload, expected = _payload_and_binding()
    before = copy.deepcopy((payload, expected))
    validate_consumed_bindings([payload], expected, protocol=load_protocol())
    validate_consumed_bindings([payload], expected, protocol=load_protocol())
    assert (payload, expected) == before


def test_basic_real_runner_manifest_passes_recipe_consumers(survey_script, summary_cli, tmp_path):
    source, out = tmp_path / "splits", tmp_path / "results"
    source.mkdir()
    make_splits(source)
    registry = locked_registry("upu")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    assert (
        survey_script.main(
            [
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
                str(registry_path),
            ]
        )
        == 0
    )
    config = config_for({"B1": (out, 1)})
    config["schema_version"] = "survey-batch-roots-2"
    config["batches"][0].update(
        recipe_policy="registry_bound",
        recipe_registry=str(registry_path),
        recipe_registry_sha256=canonical_digest(registry),
    )
    report = summary_cli.build_summary(config)
    assert report["recipe_audit"]["overall"] == "pass"
    assert report["rows"]
    assert all(
        row["recipe_binding"]["recipe_registry_sha256"] == canonical_digest(registry)
        for row in report["rows"]
    )
