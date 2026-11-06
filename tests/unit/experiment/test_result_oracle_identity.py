"""Oracle diagnostics require the actual per-seed inputs and comparable paid budgets."""

import copy

import pytest
from _survey_summary_helpers import manifest, write_tree

from pu_toolbox.experiment.results import build_summary
from pu_toolbox.experiment.survey_diagnostics import build_cards, render_cards

pytestmark = pytest.mark.unit

_ROLES = ("train", "pu_val", "clean_val", "test")


def _paired_tree(tmp_path, fault=None):
    payloads = {}
    for seed in range(5):
        for oracle in (False, True):
            payload = manifest(
                method="pn_oracle" if oracle else "nnpu",
                c_independent=oracle,
                seed=seed,
            )
            payload["representation"]["feature_sha256"] = {
                role: f"{role}-{seed}" for role in _ROLES
            }
            # PN uses clean labels and OS, while this PU run is calibrated TS.
            if oracle:
                payload["generation"]["train"]["label_view_sha256"] = "pn-labels"
                payload["generation"]["pu_val"]["label_view_sha256"] = "pn-labels"
                if seed == 4:
                    if fault == "split":
                        payload["representation"]["split_sha256"] = "another-split"
                    elif fault == "features":
                        payload["representation"]["feature_sha256"]["test"] = "another-test"
                    elif fault == "missing_features":
                        del payload["representation"]["feature_sha256"]["clean_val"]
                if fault == "budget":
                    payload["budget"]["epochs"] += 1
                elif fault == "missing_budget":
                    del payload["budget"]["actual_outer_candidates"]
            payloads[f"{'pn' if oracle else 'pu'}-{seed}"] = payload
    return write_tree(tmp_path, payloads)


def test_basic_determ_matching_pn_os_and_pu_ts_use_actual_inputs_not_label_hashes(tmp_path):
    report = build_summary([_paired_tree(tmp_path)])
    card = build_cards(report, native_view={})["nnpu"]

    assert card["signals"]["pn_oracle_alignment"][0]["gap_points"] == pytest.approx(0)
    assert report["rows"][0]["comparison_by_seed"]["4"]["feature_sha256"]["test"] == "test-4"
    assert build_cards(copy.deepcopy(report), native_view={}) == build_cards(report, native_view={})


@pytest.mark.parametrize(
    "fault", ["split", "features", "budget", "missing_features", "missing_budget"]
)
def test_param_edge_oracle_identity_mismatch_or_missing_role_never_generates_gap(tmp_path, fault):
    report = build_summary([_paired_tree(tmp_path, fault)])
    cards = build_cards(report, native_view={})
    card = cards["nnpu"]

    assert card["signals"]["pn_oracle_alignment"] == []
    omissions = [item for item in card["not_computed"] if item["signal"] == "pn_oracle_alignment"]
    assert omissions and omissions[0]["reason"]
    assert "pn_oracle_alignment" in render_cards(cards)
    assert omissions[0]["reason"] in render_cards(cards)


def test_edge_old_summary_without_per_seed_identity_does_not_infer_oracle_alignment(tmp_path):
    report = build_summary([_paired_tree(tmp_path)])
    for row in report["rows"]:
        row.pop("comparison_by_seed", None)

    card = build_cards(report, native_view={})["nnpu"]

    assert card["signals"]["pn_oracle_alignment"] == []
    assert any(item["signal"] == "pn_oracle_alignment" for item in card["not_computed"])
