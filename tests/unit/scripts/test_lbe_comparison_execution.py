"""Actual tiny fits, paired summaries and test/selection isolation."""

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from _lbe_comparison_helpers import comparison, plan_for, write_split

pytest.importorskip("torch")
pytestmark = pytest.mark.unit


def test_basic_terminal_pair_uses_same_marking_and_persists_selected_models(tmp_path):
    plan = plan_for(write_split(tmp_path))
    records = comparison.run_pair(plan, plan["pairs"][0], tmp_path / "output")
    assert records[0]["label_view_sha256"] == records[1]["label_view_sha256"]
    summary = comparison.summarize(plan, records)
    assert len(summary["rows"]) == 4 and len(summary["costs"]) == 2
    assert all(row["lbe"]["std"] is None for row in summary["rows"])
    for row in records:
        manifest = json.loads(Path(row["manifest"]).read_text(encoding="utf-8"))
        assert manifest["execution_mode"] == "technical_smoke"
        assert not manifest["formal_eligible"]
        assert not manifest["candidate_runs"][0]["epoch_checkpoints"]
        assert set(row["diagnostics"]["selected_models_sha256"]) == {
            "selected_PA.pkl",
            "selected_OA.pkl",
        }
        test = comparison.load_split(plan["split_root"], "spambase", 0)[0][-1]
        for protocol in ("PA", "OA"):
            with (Path(row["manifest"]).parent / f"selected_{protocol}.pkl").open("rb") as stream:
                saved = pickle.load(stream)
            predicted = comparison.predict_saved_selection(saved, test.X)
            assert np.mean(predicted == test.labels) == row["test_results"][protocol]["accuracy"]


def test_basic_em_round_scope_keeps_checkpoint_selection_explicit(tmp_path):
    plan = plan_for(write_split(tmp_path), selection_scope="em_rounds")
    records = comparison.run_pair(plan, plan["pairs"][0], tmp_path / "output")
    assert all("checkpoint" in item for item in records[1]["selection"].values())
    assert all("checkpoint" not in item for item in records[0]["selection"].values())
    assert records[1]["diagnostics"]["fit"]["optimizer_steps_"] == 8
    assert comparison.summarize(plan, records)["selection_scope"] == "em_rounds"


def test_determ_test_changes_do_not_change_fits_or_selection(tmp_path):
    root = write_split(tmp_path)
    plan = plan_for(root)
    first = comparison.run_pair(plan, plan["pairs"][0], tmp_path / "first")
    source = root / "split_0/test.npz"
    with np.load(source) as data:
        features, labels, indices = (data[key].copy() for key in ("X", "y", "indices"))
    np.savez(source, X=features + 20, y=1 - labels, indices=indices)
    changed = plan_for(root)
    second = comparison.run_pair(changed, changed["pairs"][0], tmp_path / "second")
    for left, right in zip(first, second, strict=True):
        assert left["selection"] == right["selection"]
        for protocol in ("PA", "OA"):
            models = []
            for row in (left, right):
                path = Path(row["manifest"]).parent / f"selected_{protocol}.pkl"
                with path.open("rb") as stream:
                    models.append(pickle.load(stream)["model"])
            np.testing.assert_array_equal(
                models[0].decision_function(features), models[1].decision_function(features)
            )


@pytest.mark.parametrize("mode", ["missing", "duplicate", "marking"])
def test_param_incomplete_or_unpaired_records_cannot_be_summarized(tmp_path, mode):
    plan = plan_for(write_split(tmp_path))
    records = comparison.run_pair(plan, plan["pairs"][0], tmp_path / "output")
    if mode == "missing":
        records.pop()
    elif mode == "duplicate":
        records.append(copy.deepcopy(records[0]))
    else:
        records[1]["label_view_sha256"]["train"] = "wrong"
    with pytest.raises(ValueError):
        comparison.summarize(plan, records)


def test_edge_changed_source_after_planning_is_rejected_before_fit(tmp_path):
    root = write_split(tmp_path)
    plan = plan_for(root)
    path = root / "split_0/split_manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after planning"):
        comparison.run_pair(plan, plan["pairs"][0], tmp_path / "output")


def test_basic_paired_sd_uses_seed_differences_not_independent_variances():
    plan = {
        "dataset": "synthetic",
        "smoke": False,
        "selection_scope": "terminal",
        "pairs": [{"seed": seed, "c_token": "0.5"} for seed in (0, 1)],
    }
    records = []
    for seed in (0, 1):
        for method in comparison.METHODS:
            value = 0.5 + seed * 0.2 + (0.1 if method == "lbe_neural" else 0)
            records.append(
                {
                    "seed": seed,
                    "method": method,
                    "c_token": "0.5",
                    "label_view_sha256": {"train": str(seed)},
                    "test_results": {
                        p: {m: value for m in ("accuracy", "auc")} for p in ("PA", "OA")
                    },
                    "elapsed_seconds": 1.0,
                    "peak_gpu_memory_bytes": None,
                }
            )
    result = comparison.summarize(plan, records)["rows"][0]
    assert result["lbe"]["std"] == pytest.approx(np.sqrt(0.02))
    assert result["paired_neural_minus_linear"]["std"] == pytest.approx(0, abs=1e-14)
