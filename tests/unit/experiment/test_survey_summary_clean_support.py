# ruff: noqa: F811
"""Support results retain their supervision budget and separate PA identity."""

import json

import pytest
from _survey_summary_helpers import manifest, summary_cli, write_tree  # noqa: F401

from pu_toolbox.experiment.results import build_summary
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


def test_basic_support_identity_and_per_seed_label_budget_are_preserved(tmp_path):
    root = write_tree(tmp_path, {str(seed): _support(seed) for seed in (0, 1)})
    report = build_summary([root], expected_seeds=[0, 1])
    assert {row["row_key"]["selection_protocol"] for row in report["rows"]} == {
        "PA+clean-support",
        "OA",
    }
    assert len(report["run_costs"]) == 1
    for row in report["rows"]:
        assert row["pure_pu_comparison_eligible"] is False
        assert row["metric"]["n_observed"] == 2
        assert row["extra_supervision_by_seed"]["1"]["support_positive_labels"] == 41
        assert row["status"] == "partial"
    assert report["run_costs"][0]["costs"]["tuning_cost_seconds"]["sum"] == 400


def test_edge_missing_support_seed_and_unavailable_metric_are_distinct(tmp_path):
    tree = write_tree(tmp_path, {"run": _support()})
    report = build_summary([tree], expected_seeds=[0, 1])
    for row in report["rows"]:
        assert row["metric"]["missing_seeds"] == [1]
        assert row["additional_metrics"]["f1"]["unavailable_seeds"] == [0]


@pytest.mark.parametrize("keep_metadata", [True, False])
def test_param_support_selection_cannot_gain_pure_pu_or_formal_status(tmp_path, keep_metadata):
    payload = _support()
    if not keep_metadata:
        del payload["extra_supervision"]
    tree = write_tree(tmp_path, {"run": payload})
    report = build_summary([tree], expected_seeds=[0])
    assert all(row["pure_pu_comparison_eligible"] is False for row in report["rows"])
    assert all(row["status"] != "formal" for row in report["rows"])


def test_edge_support_blockers_remain_visible_even_when_metadata_claims_approval(tmp_path):
    payload = _support()
    payload["extra_supervision"].update(
        support_source_verified=True,
        all_role_isolation_verified=True,
        support_removed_from_train=True,
    )
    payload["formal_blockers"] = ["protocol_deviation"]
    tree = write_tree(tmp_path, {"run": payload})
    report = build_summary([tree], expected_seeds=[0])
    assert all(row["status"] != "formal" for row in report["rows"])
    assert all(row["formal_blockers"] == ["protocol_deviation"] for row in report["rows"])


def test_edge_support_and_pure_results_cannot_merge_in_one_method_row(tmp_path):
    tree = write_tree(tmp_path, {"support": _support(0), "pure": manifest(seed=1)})
    with pytest.raises(SummaryError, match="configurations"):
        build_summary([tree], expected_seeds=[0, 1])


def test_basic_determ_cli_reports_support_without_mutating_inputs(summary_cli, tmp_path):
    root = write_tree(tmp_path / "input", {"run": _support()})
    path = root / "run/manifest.json"
    original = path.read_bytes()
    outputs = []
    for name in ("one", "two"):
        out = tmp_path / name
        assert (
            summary_cli.main(
                ["--manifests", str(root), "--expected-seeds", "0", "--out-dir", str(out)]
            )
            == 0
        )
        output = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        output.pop("generated_at")
        outputs.append(output)
    assert outputs[0] == outputs[1]
    assert path.read_bytes() == original
    report = outputs[0]
    assert all(row["status"] != "formal" for row in report["rows"])
    assert "PA+clean-support" in (tmp_path / "one/summary.md").read_text(encoding="utf-8")
