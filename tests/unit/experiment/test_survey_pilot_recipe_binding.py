"""Pilot recipe preflight pins all children before any data or batch is read."""

# ruff: noqa: F811

import json

import pytest
from _survey_pilot_helpers import driver, splits_tree, unit_calls  # noqa: F401
from _survey_recipe_registry_helpers import locked_registry

from pu_toolbox.experiment.survey_recipe_registry import canonical_digest

pytestmark = pytest.mark.unit


def command(tmp_path, registry):
    return [
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
    ]


def save_registry(tmp_path, value=None):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(value or locked_registry()))
    return path


def test_basic_determ_dry_run_records_bound_identity_for_each_planned_run(driver, tmp_path):
    splits_tree(tmp_path / "splits", datasets=("spambase",), prior=0.3)
    registry = save_registry(tmp_path)
    plan = tmp_path / "plan.json"
    args = [*command(tmp_path, registry), "--dry-run", "--plan-json", str(plan)]
    assert driver.main(args) == 0
    first = json.loads(plan.read_text())
    assert len(first["recipe_bindings"]) == first["totals"]["planned"] == 35
    assert {b["recipe_registry_sha256"] for b in first["recipe_bindings"]} == {
        canonical_digest(locked_registry())
    }
    assert driver.main(args) == 0
    assert json.loads(plan.read_text()) == first


@pytest.mark.parametrize("defect", ["draft", "scope", "basis"])
def test_param_registry_rejected_before_split_reads_or_batch_launch(
    driver, unit_calls, tmp_path, monkeypatch, defect
):
    value = locked_registry()
    if defect == "draft":
        value["recipe_lifecycle"] = "draft"
    elif defect == "scope":
        value["profiles"]["upu"]["scope"]["datasets"] = ["imdb"]
    else:
        value["protocol_basis"]["protocol_sha256"] = "0" * 64
    registry = save_registry(tmp_path, value)

    def forbidden(*args, **kwargs):
        pytest.fail("split metadata read before recipe validation")

    monkeypatch.setattr(driver, "split_digests", forbidden)
    assert driver.main(command(tmp_path, registry)) == 1
    assert unit_calls == []


def test_edge_pin_is_passed_to_every_child_not_recomputed_from_later_files(
    driver, unit_calls, tmp_path
):
    splits_tree(tmp_path / "splits", datasets=("spambase",), prior=0.3)
    registry = save_registry(tmp_path)
    # Stub children do not create manifests, so the final rescan reports pending.
    assert driver.main(command(tmp_path, registry)) == 1
    assert len(unit_calls) == 3
    expected = canonical_digest(locked_registry())
    for argv in unit_calls:
        assert argv[argv.index("--recipe-registry") + 1] == str(registry)
        assert argv[argv.index("--recipe-registry-sha256") + 1] == expected
