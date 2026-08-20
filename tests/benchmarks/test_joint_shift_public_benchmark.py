# ruff: noqa: N803, N806

"""Tests for the public-data multi-seed joint-shift benchmark."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from benchmarks.joint_shift.runner import (  # noqa: E402
    run_joint_shift_benchmark,
    summarize_trials,
)

pytestmark = pytest.mark.integration


def test_basic_public_benchmark_writes_trials_summary_and_split_audit(tmp_path):
    report = run_joint_shift_benchmark(
        seeds=(3, 4),
        methods=("dynamic", "trpu"),
        target_pu_size=80,
        max_epochs=1,
        hidden_dim=6,
        feature_dim=4,
    )
    report.save(tmp_path)
    assert len(report.trials) == 4
    assert len(report.summary) == 4
    assert all(row["source_target_overlap"] == 0 for row in report.split_audit)
    assert all(row["train_test_overlap"] == 0 for row in report.split_audit)
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["paper_claim"] is False
    assert (tmp_path / "trials.csv").is_file()
    assert (tmp_path / "summary.md").is_file()


def test_param_student_t_summary_matches_known_two_seed_values():
    trials = [
        {"method": "a", "seed": 1, "accuracy": 0.6, "roc_auc": 0.7},
        {"method": "a", "seed": 2, "accuracy": 0.8, "roc_auc": 0.9},
    ]
    summary = summarize_trials(trials, confidence=0.95)
    accuracy = next(row for row in summary if row["metric"] == "accuracy")
    assert accuracy["mean"] == pytest.approx(0.7)
    assert accuracy["std"] == pytest.approx(2**0.5 / 10)
    assert accuracy["ci_low"] < accuracy["mean"] < accuracy["ci_high"]


def test_edge_seed_method_and_trial_contracts_fail_early():
    with pytest.raises(ValueError, match="at least two unique"):
        run_joint_shift_benchmark(seeds=(1,), methods=("trpu",), max_epochs=1)
    with pytest.raises(ValueError, match="Unknown methods"):
        run_joint_shift_benchmark(seeds=(1, 2), methods=("bad",), max_epochs=1)
    with pytest.raises(ValueError, match="duplicate seeds"):
        summarize_trials(
            [
                {"method": "a", "seed": 1, "accuracy": 0.5, "roc_auc": 0.5},
                {"method": "a", "seed": 1, "accuracy": 0.6, "roc_auc": 0.6},
            ]
        )


def test_determ_same_protocol_produces_identical_serialized_results():
    options = dict(
        seeds=(5, 6),
        methods=("tepu",),
        target_pu_size=80,
        max_epochs=1,
        hidden_dim=6,
        feature_dim=4,
    )
    first = run_joint_shift_benchmark(**options)
    second = run_joint_shift_benchmark(**options)
    assert first.to_dict() == second.to_dict()
    assert first.trials == second.trials
