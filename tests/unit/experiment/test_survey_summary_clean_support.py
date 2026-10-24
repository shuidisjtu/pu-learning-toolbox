# ruff: noqa: F811
"""Extra-supervised selection stays separate from pure PA and formal ranking."""

import copy
import json

import pytest
from _survey_summary_helpers import (  # noqa: F401
    config_for,
    entry,
    manifest,
    summary_cli,
    write_config,
    write_tree,
)

from pu_toolbox.experiment.survey_summary import SummaryError

pytestmark = pytest.mark.unit


def _support(seed=0):
    payload = manifest(seed=seed)
    for key in ("selection", "test_results"):
        payload[key]["PA+clean-support"] = payload[key].pop("PA")
    payload["extra_supervision"] = {
        "extra_supervision": True,
        "pa_eligible": False,
        "pa_clean_support_eligible": True,
        "support_samples": 100,
        "support_positive_labels": 40 + seed,
        "support_negative_labels": 60 - seed,
        "support_source_verified": False,
        "selection_scope": "completed_meta_epochs",
    }
    return payload


def _rows(script, payloads):
    rows, runs, _ = script.collect_observations(
        [entry(payload, name=f"seed_{payload['seed']}.json") for payload in payloads],
        metrics=script.CARRIED_METRICS,
    )
    return script.build_rows(
        rows, expected_seeds=[0, 1], gate_reasons={}, unrecorded_by_run={}
    ), runs


def test_basic_support_protocol_is_not_dropped_or_renamed_and_cost_is_once(summary_cli):
    rows, runs = _rows(summary_cli, [_support(0), _support(1)])

    assert {row["result_identity"]["selection_protocol"] for row in rows} == {
        "pa+clean-support",
        "oa",
    }
    assert {row["row_key"]["selection_protocol"] for row in rows} == {"PA+clean-support", "OA"}
    assert len(runs) == 1
    assert len(next(iter(runs.values()))) == 2
    for row in rows:
        assert row["status"] == "partial"
        assert "extra_clean_support" in row["reasons"]
        assert row["pure_pu_comparison_eligible"] is False
        assert row["metric"]["n_observed"] == 2
        assert row["extra_supervision_by_seed"]["1"]["support_positive_labels"] == 41
        assert set(row["additional_metrics"]) == {"auc", "f1", "precision", "recall"}


def test_edge_missing_support_seed_keeps_missing_and_unavailable_metrics_separate(summary_cli):
    rows, _ = _rows(summary_cli, [_support()])
    for row in rows:
        assert row["metric"]["missing_seeds"] == [1]
        assert row["additional_metrics"]["f1"]["unavailable_seeds"] == [0]


@pytest.mark.parametrize("keep_metadata", [True, False])
def test_param_support_protocol_cannot_be_ranked_even_if_manifest_claims_formal(
    summary_cli, keep_metadata
):
    payload = _support()
    if not keep_metadata:
        del payload["extra_supervision"]
    rows, _ = _rows(summary_cli, [payload])
    assert all(row["status"] != "formal" for row in rows)


def test_edge_pure_pa_with_explicit_extra_support_is_rejected(summary_cli):
    payload = _support()
    payload["test_results"]["PA"] = {"accuracy": 0.9}
    with pytest.raises(SummaryError, match="pure PA"):
        _rows(summary_cli, [payload])


def test_edge_support_and_pure_results_cannot_merge_even_with_same_group_key(summary_cli):
    with pytest.raises(SummaryError, match="supervision"):
        _rows(summary_cli, [_support(0), manifest(seed=1)])


def test_basic_legacy_pa_oa_remain_formal_without_new_metadata(summary_cli):
    rows, _ = _rows(summary_cli, [manifest(seed=0), manifest(seed=1)])
    assert {row["status"] for row in rows} == {"formal"}
    assert all("extra_supervision_by_seed" not in row for row in rows)


def test_basic_cli_renders_support_separately_without_formal_admission(summary_cli, tmp_path):
    root = write_tree(tmp_path / "input", {"seed0": _support(0), "seed1": _support(1)})
    config = write_config(tmp_path / "roots.json", config_for({"support": (root, 2)}))
    output = tmp_path / "out"
    summary_cli.main(["--config", str(config), "--out-dir", str(output)])
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(report["rows"]) == 2
    assert all(row["status"] != "formal" for row in report["rows"])
    assert "PA+clean-support" in (output / "summary.md").read_text(encoding="utf-8")


def test_determ_support_summary_is_repeatable_and_does_not_mutate_input(summary_cli):
    payloads = [_support(0), _support(1)]
    before = copy.deepcopy(payloads)
    assert _rows(summary_cli, payloads) == _rows(summary_cli, payloads)
    assert payloads == before


def test_basic_independent_support_manifests_use_explicit_diagnostic_mode(summary_cli, tmp_path):
    payloads = []
    for seed in (0, 1):
        payload = _support(seed)
        for field in (
            "execution_unit",
            "training_path",
            "representation",
            "budget",
            "protocol_version",
            "protocol_sha256",
        ):
            del payload[field]
        payload["execution_mode"] = "technical_smoke"
        payload["formal_eligible"] = False
        payload["formal_blockers"] = ["no_versioned_survey_protocol"]
        payload["extra_supervision"].update(method="lagam", dataset="spambase")
        payload["estimator_parameters"] = {"max_epochs": 3, "random_state": seed}
        payloads.append(payload)
    root = write_tree(tmp_path / "input", {str(p["seed"]): p for p in payloads})
    config = write_config(
        tmp_path / "roots.json",
        config_for({"support": (root, 2)}, expected_seeds=[0, 1, 2]),
    )
    output = tmp_path / "out"
    assert (
        summary_cli.main(["--config", str(config), "--out-dir", str(output), "--clean-support"])
        == 0
    )
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["formal_readiness_assessed"] is False
    assert len(report["rows"]) == 2
    assert all(row["metric"]["missing_seeds"] == [2] for row in report["rows"])
    assert all(row["status"] == "technical_probe" for row in report["rows"])
    assert len(report["run_costs"]) == 1
    assert report["rows"][0]["extra_supervision_by_seed"]["0"]["support_samples"] == 100
    assert "PA+clean-support" in (output / "summary.md").read_text(encoding="utf-8")
    assert len(report["input_manifest_sha256"]) == 2
    payloads[1]["estimator_parameters"]["max_epochs"] = 4
    write_tree(root, {str(p["seed"]): p for p in payloads})
    split_report = summary_cli.build_clean_support_summary(
        config_for({"support": (root, 2)}, expected_seeds=[0, 1, 2])
    )
    assert len(split_report["rows"]) == 4
    assert len(split_report["run_costs"]) == 2


@pytest.mark.parametrize("seeds", [None, [], [0, 0], [True], [-1]])
def test_param_diagnostic_mode_requires_explicit_expected_seeds(summary_cli, seeds):
    with pytest.raises(SummaryError, match="expected_seeds"):
        summary_cli.build_clean_support_summary({"expected_seeds": seeds})


def test_edge_diagnostic_mode_rejects_versioned_inputs(summary_cli, tmp_path):
    root = write_tree(tmp_path / "input", {"seed0": _support()})
    config = config_for({"support": (root, 1)}, expected_seeds=[0])
    with pytest.raises(SummaryError, match="independent technical_smoke"):
        summary_cli.build_clean_support_summary(config)
