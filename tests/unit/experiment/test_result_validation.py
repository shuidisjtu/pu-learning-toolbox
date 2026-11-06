"""Read-only summaries retain selection evidence and reject inconsistent observations."""

import json

import pytest
from _survey_summary_helpers import manifest, write_tree

from pu_toolbox.experiment.results import build_summary
from pu_toolbox.experiment.survey_summary import SummaryError

pytestmark = pytest.mark.unit


def test_basic_determ_selected_candidates_remain_distinct_and_traceable(tmp_path):
    payload = manifest(candidates=2)
    payload["selection"]["OA"]["candidate_index"] = 1
    tree = write_tree(tmp_path, {"run": payload})
    first = build_summary([tree], expected_seeds=[0])

    assert first == build_summary([tree], expected_seeds=[0])
    source = first["sources"][0]
    assert source["selection"]["PA"]["candidate_index"] == 0
    assert source["selection"]["OA"]["candidate_index"] == 1
    assert source["budget"] == payload["budget"]
    assert source["reproducibility"]["state"] == "unavailable"


@pytest.mark.parametrize("fault", ["excluded_candidate", "wrong_protocol", "staged_budget"])
def test_param_edge_inconsistent_selection_or_paid_budget_is_rejected(tmp_path, fault):
    payload = manifest()
    if fault == "excluded_candidate":
        payload["candidate_runs"][0]["status"] = "excluded"
    elif fault == "wrong_protocol":
        payload["selection"]["OA"]["protocol"] = "PA"
    else:
        other = manifest(method="pulda")
        other["budget"]["staged_budget"] = {"algorithm_id": "pulda"}
        write_tree(tmp_path, {"other": other})
    write_tree(tmp_path, {"run": payload})

    with pytest.raises(SummaryError):
        build_summary([tmp_path], expected_seeds=[0])


def test_edge_missing_reproducibility_is_disclosed_without_inventing_identity(tmp_path):
    tree = write_tree(tmp_path, {"run": manifest(environment=False)})

    report = build_summary([tree], expected_seeds=[0])

    assert all("missing_reproducibility" in row["reasons"] for row in report["rows"])
    assert report["sources"][0]["reproducibility"]["missing"]
    assert all(row["status"] != "formal" for row in report["rows"])


def test_edge_missing_shared_packet_preserves_original_reference(tmp_path):
    payload = manifest()
    payload["batch"] = {
        "batch_id": "a" * 64,
        "evidence_file": "missing.json",
        "evidence_sha256": "b" * 64,
    }
    tree = write_tree(tmp_path, {"run": payload})
    path = tree / "run/manifest.json"
    before = path.read_bytes()

    report = build_summary([tree], expected_seeds=[0])

    identity = report["sources"][0]["reproducibility"]
    assert identity["state"] == "unavailable"
    assert identity["batch"] == payload["batch"]
    assert path.read_bytes() == before
    assert json.loads(path.read_text(encoding="utf-8"))["batch"] == payload["batch"]


def test_edge_single_seed_smoke_is_not_a_five_seed_conclusion(tmp_path):
    pytest.importorskip("openpyxl")
    from pu_toolbox.experiment.survey_diagnostics import build_cards
    from pu_toolbox.experiment.survey_workbook import build_workbook

    tree = write_tree(tmp_path, {"run": manifest()})
    report = build_summary([tree], expected_seeds=[0])
    assert report["rows"][0]["metric"]["n_observed"] == 1
    workbook = build_workbook(report)
    assert not any("nnpu" in row for row in workbook["SCAR-PA"].iter_rows(values_only=True))
    assert any("nnpu" in row for row in workbook["coverage"].iter_rows(values_only=True))
    assert build_cards(report, native_view={})["nnpu"]["rows_used"] == 0


def _checkpoint_manifest():
    payload = manifest()
    checkpoint = {
        "path": "epoch_0001_model.pt",
        "sha256": "a" * 64,
        "epoch_position": 1,
        "epoch_label": 0,
        "component": "model",
    }
    payload["candidate_runs"][0].update({"trajectory_index": 0, "epoch_checkpoints": [checkpoint]})
    for selected in payload["selection"].values():
        selected.update(
            {"run_index": 0, "epoch": 1, "checkpoint_index": 0, "checkpoint": dict(checkpoint)}
        )
    return payload


@pytest.mark.parametrize("fault", ["epoch", "index", "position", "label", "component"])
def test_param_edge_selected_checkpoint_metadata_must_match_its_trajectory(tmp_path, fault):
    payload = _checkpoint_manifest()
    selected = payload["selection"]["OA"]
    if fault == "epoch":
        selected["epoch"] = 999
    elif fault == "index":
        selected["checkpoint_index"] = 99
    else:
        field = {"position": "epoch_position", "label": "epoch_label", "component": "component"}[
            fault
        ]
        selected["checkpoint"][field] = "wrong"

    tree = write_tree(tmp_path, {"run": payload})

    with pytest.raises(SummaryError, match="selection"):
        build_summary([tree], expected_seeds=[0])


def test_basic_legacy_checkpoint_fields_are_not_backfilled_or_required(tmp_path):
    payload = _checkpoint_manifest()
    for selected in payload["selection"].values():
        del selected["epoch"]
        del selected["checkpoint_index"]
        selected["checkpoint"] = {
            key: value for key, value in selected["checkpoint"].items() if key in {"path", "sha256"}
        }
    tree = write_tree(tmp_path, {"run": payload})

    assert build_summary([tree], expected_seeds=[0])["coverage"]["available"] == 1


def test_basic_checkpoint_index_tracks_components_independently_of_epoch_position(tmp_path):
    payload = _checkpoint_manifest()
    teacher = dict(
        payload["candidate_runs"][0]["epoch_checkpoints"][0],
        component="teacher",
        path="epoch_0001_teacher.pt",
    )
    payload["candidate_runs"][0]["epoch_checkpoints"].append(teacher)
    payload["selection"]["OA"].update({"checkpoint_index": 1, "checkpoint": dict(teacher)})
    tree = write_tree(tmp_path, {"run": payload})

    assert build_summary([tree], expected_seeds=[0])["coverage"]["available"] == 1
