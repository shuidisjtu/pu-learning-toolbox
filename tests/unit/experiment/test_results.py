"""Read completed v3 results without replaying the experiment process."""

import copy
import json
import sys

import pytest
from _survey_summary_helpers import manifest, write_tree

from pu_toolbox.experiment.results import build_summary
from pu_toolbox.experiment.survey_summary import SummaryError

pytestmark = pytest.mark.unit


def _grid(root, **kwargs):
    return write_tree(
        root,
        {
            f"seed_{seed}": manifest(seed=seed, accuracy=0.8 + seed * 0.01, **kwargs)
            for seed in range(5)
        },
    )


def test_basic_complete_results_keep_statistics_and_count_costs_once(tmp_path):
    tree = _grid(tmp_path / "tree", blockers=["review_pending"])
    before = {path: path.read_bytes() for path in tree.rglob("manifest.json")}

    report = build_summary([tree, tree / "seed_0/manifest.json"])

    assert len(report["rows"]) == 2
    assert all(row["metric"]["mean"] == pytest.approx(0.82) for row in report["rows"])
    assert report["rows"][0]["metric"]["std"] == pytest.approx(0.0158113883)
    assert report["admission_status"] == "partial"
    assert all(row["status"] == "partial" for row in report["rows"])
    assert all("review_pending" in row["formal_blockers"] for row in report["rows"])
    assert len(report["run_costs"]) == 1
    assert report["run_costs"][0]["costs"]["tuning_cost_seconds"]["sum"] == 1000.0
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("mode", ["pending", "excluded", "failed", "missing_accuracy", "risk"])
def test_param_unusable_results_remain_visible_without_entering_means(tmp_path, mode):
    tree = _grid(tmp_path / "tree")
    path = tree / "seed_4/manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    kwargs = {}
    if mode in {"pending", "excluded"}:
        kwargs[f"{mode}_paths"] = [path]
    elif mode == "failed":
        payload["selection"] = {}
        payload["test_results"] = {}
        payload["failures"] = [{"reason": "fit failed"}]
    elif mode == "risk":
        payload["numerical_risk"] = "constant predictions"
    else:
        payload["test_results"]["PA"].pop("accuracy")
        payload["test_results"]["OA"]["accuracy"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_summary([tree], **kwargs)

    assert all(row["metric"]["missing_seeds"] == [4] for row in report["rows"])
    assert all(row["status"] == "incomplete" for row in report["rows"])
    assert report["coverage"][mode] == 1
    assert any(item["path"] == str(path.resolve()) for item in report["sources"])


@pytest.mark.parametrize(
    "fault", ["duplicate", "budget", "split", "backbone", "selection", "view", "old"]
)
def test_param_conflicting_identities_or_evidence_are_refused(tmp_path, fault):
    tree = _grid(tmp_path / "tree")
    path = tree / ("seed_0/manifest.json" if fault == "duplicate" else "seed_4/manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if fault == "duplicate":
        path = tree / "duplicate/manifest.json"
        path.parent.mkdir()
    elif fault == "budget":
        payload["budget"]["batch_size"] = 128
    elif fault in {"split", "backbone"}:
        other = copy.deepcopy(payload)
        other["execution_unit"]["method"] = "pulda"
        if fault == "split":
            other["representation"]["split_sha256"] = "different"
        else:
            other["backbone"] = "different"
        write_tree(tree, {"other": other})
    elif fault == "selection":
        payload["selection"]["OA"]["candidate_index"] = 99
    elif fault == "view":
        payload["calibration_applied"] = False
    else:
        payload["protocol_version"] = "survey-v2.0"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SummaryError):
        build_summary([tree])


def test_edge_support_sar_and_oracle_keep_distinct_selection_roles(tmp_path):
    support = manifest(method="lagam")
    support["extra_supervision"] = {"extra_supervision": True, "support_size": 100}
    support["selection"]["PA+clean-support"] = support["selection"].pop("PA")
    support["test_results"]["PA+clean-support"] = support["test_results"].pop("PA")
    sar = manifest(method="lbe", mechanism="sar_lbe_a")
    oracle = manifest(method="pn_oracle", c_independent=True)
    tree = write_tree(tmp_path / "tree", {"support": support, "sar": sar, "oracle": oracle})

    report = build_summary([tree], expected_seeds=[0])

    assert len(report["rows"]) == 4
    assert all(
        row["row_key"]["selection_protocol"] == "OA"
        for row in report["rows"]
        if row["row_key"]["mechanism"] != "scar"
    )
    assert any(row["row_key"]["c_token"] == "c_independent" for row in report["rows"])
    assert report["coverage"]["diagnostic_results"] == 2
    support_rows = [row for row in report["rows"] if row["row_key"]["method"] == "lagam"]
    assert all(row["pure_pu_comparison_eligible"] is False for row in support_rows)


def test_edge_clean_support_cannot_be_reported_as_pure_pa(tmp_path):
    payload = manifest(method="lagam")
    payload["extra_supervision"] = {"extra_supervision": True}
    tree = write_tree(tmp_path / "tree", {"run": payload})

    with pytest.raises(SummaryError, match="support"):
        build_summary([tree])


def test_determ_missing_path_and_pending_unreceived_path_are_explicit(tmp_path):
    missing = tmp_path / "unreceived"
    with pytest.raises(SummaryError, match="missing"):
        build_summary([missing])

    first = build_summary([], pending_paths=[missing])
    second = build_summary([], pending_paths=[missing])

    assert first == second
    assert first["coverage"]["pending"] == 1
    assert first["rows"] == []


def test_basic_read_only_report_reaches_workbook_diagnostics_and_conclusions(tmp_path, monkeypatch):
    pytest.importorskip("openpyxl")
    from pu_toolbox.experiment import survey_protocol
    from pu_toolbox.experiment.survey_conclusions import build_conclusions
    from pu_toolbox.experiment.survey_diagnostics import build_cards
    from pu_toolbox.experiment.survey_workbook import build_workbook

    tree = _grid(tmp_path / "tree")
    for path in tree.rglob("manifest.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["test_results"]["OA"]["accuracy"] += 0.1
        path.write_text(json.dumps(payload), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("result readers must not replay protocol/comparison loading")

    monkeypatch.setattr(survey_protocol, "load_protocol", forbidden)
    monkeypatch.setitem(sys.modules, "pu_toolbox.experiment.survey_comparison", None)
    monkeypatch.setitem(sys.modules, "pu_toolbox.experiment.survey_profile_map", None)
    report = build_summary(tree)
    cards = build_cards(report, native_view={"nnpu": "ts"})
    conclusions = build_conclusions(report, cards)
    workbook = build_workbook(report)

    assert cards["nnpu"]["rows_used"] == 2
    assert conclusions["conclusions"]
    for sheet, expected in (("SCAR-PA", "82.00"), ("SCAR-OA", "92.00")):
        values = [cell for row in workbook[sheet].iter_rows(values_only=True) for cell in row]
        assert "nnpu" in values
        assert any(isinstance(cell, str) and expected in cell and "±" in cell for cell in values)


@pytest.mark.parametrize("fault", ["protocol", "checkpoint"])
def test_param_selection_evidence_cannot_be_relabelled(tmp_path, fault):
    payload = manifest()
    if fault == "protocol":
        payload["selection"]["PA"]["protocol"] = "OA"
    else:
        payload["selection"]["PA"]["checkpoint"] = {"path": "other.pt", "sha256": "other"}
    tree = write_tree(tmp_path / "tree", {"run": payload})

    with pytest.raises(SummaryError, match="selection"):
        build_summary([tree])
