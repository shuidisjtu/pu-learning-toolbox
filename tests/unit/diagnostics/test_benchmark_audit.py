"""Tests for persisted benchmark artifact auditing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pu_toolbox.diagnostics import BenchmarkAuditReport, audit_benchmark_results
from pu_toolbox.utils.serialization import canonical_hash

pytestmark = pytest.mark.unit


@pytest.fixture
def result_dir(tmp_path):
    config = {"protocol": "official_data", "seeds": [0, 1]}
    manifest = {
        "protocol": "official_data",
        "paper_claim": False,
        "n_trials": 2,
        "config_sha256": canonical_hash(config),
        "git_worktree_dirty": False,
        "dataset_splits": {
            "0": {
                "labeled_unlabeled_overlap": 0,
                "train_validation_overlap": 0,
                "target_unlabeled_class_prior": 0.3,
                "hidden_unlabeled_positive_rate": 0.3,
            },
            "1": {
                "labeled_unlabeled_overlap": 0,
                "train_validation_overlap": 0,
                "target_unlabeled_class_prior": 0.3,
                "hidden_unlabeled_positive_rate": 0.3,
            },
        },
    }
    (tmp_path / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "trials.csv").write_text(
        "method,seed,accuracy,roc_auc\ninfomax_pu,0,0.7,0.8\ninfomax_pu,1,0.6,0.75\n",
        encoding="utf-8",
    )
    (tmp_path / "summary.csv").write_text(
        "method,metric,mean\ninfomax_pu,roc_auc,0.775\n", encoding="utf-8"
    )
    return tmp_path


def _update_json(path: Path, update) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    update(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_basic_valid_directory_passes_and_serializes(result_dir, tmp_path):
    report = audit_benchmark_results(result_dir)
    assert isinstance(report, BenchmarkAuditReport)
    assert report.passed
    assert (report.n_trials, report.n_methods, report.n_seeds) == (2, 1, 2)
    assert all(report.checks.values())
    assert any("paper_claim=false" in item for item in report.warnings)
    output = tmp_path / "audit" / "report.json"
    report.save(output)
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_edge_missing_directory_and_artifacts_fail(tmp_path):
    report = audit_benchmark_results(tmp_path / "missing")
    assert not report.passed
    assert not report.checks["result_directory"]
    assert "missing required artifacts" in " ".join(report.errors)


def test_edge_config_tampering_and_trial_count_fail(result_dir):
    _update_json(result_dir / "resolved_config.json", lambda value: value.update(seeds=[0]))
    _update_json(result_dir / "run_manifest.json", lambda value: value.update(n_trials=3))
    report = audit_benchmark_results(result_dir)
    assert not report.checks["config_hash"]
    assert not report.checks["trial_count"]
    assert not report.passed


@pytest.mark.parametrize("bad_value", ["nan", "inf", "not-a-number"])
def test_param_nonfinite_or_invalid_metric_fails(result_dir, bad_value):
    rows = list(csv.reader((result_dir / "trials.csv").read_text().splitlines()))
    rows[1][2] = bad_value
    with (result_dir / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    report = audit_benchmark_results(result_dir)
    assert not report.checks["finite_metrics"]
    assert "not finite" in " ".join(report.errors)


def test_edge_empty_inapplicable_metric_is_allowed(result_dir):
    text = (result_dir / "trials.csv").read_text(encoding="utf-8").replace("0.7,0.8", ",0.8")
    (result_dir / "trials.csv").write_text(text, encoding="utf-8")
    assert audit_benchmark_results(result_dir).checks["finite_metrics"]


def test_edge_duplicate_trials_and_missing_seed_fail(result_dir):
    rows = (result_dir / "trials.csv").read_text(encoding="utf-8").splitlines()
    (result_dir / "trials.csv").write_text(
        "\n".join([rows[0], rows[1], rows[1]]) + "\n", encoding="utf-8"
    )
    report = audit_benchmark_results(result_dir)
    assert not report.checks["no_duplicate_trials"]
    assert not report.checks["seed_coverage"]


def test_edge_split_overlap_and_prior_mismatch_fail(result_dir):
    def corrupt(manifest):
        manifest["dataset_splits"]["0"]["labeled_unlabeled_overlap"] = 1
        manifest["dataset_splits"]["1"]["hidden_unlabeled_positive_rate"] = 0.4

    _update_json(result_dir / "run_manifest.json", corrupt)
    report = audit_benchmark_results(result_dir)
    assert not report.checks["pu_split_integrity"]
    assert any("overlap" in item for item in report.errors)
    assert any("does not match target" in item for item in report.errors)


def test_determ_json_output_is_stable_and_dirty_worktree_warns(result_dir):
    _update_json(
        result_dir / "run_manifest.json",
        lambda value: value.update(git_worktree_dirty=True),
    )
    first = audit_benchmark_results(result_dir)
    second = audit_benchmark_results(result_dir)
    assert first.to_json() == second.to_json()
    assert any("dirty Git worktree" in item for item in first.warnings)


def test_basic_tracked_infomax_matrix_passes():
    root = Path(__file__).resolve().parents[3]
    for suffix in ("03", "05", "07"):
        result = root / "benchmarks" / "deep_pu" / "results"
        report = audit_benchmark_results(result / f"infomax_fashion_protocol_pi{suffix}_full")
        assert report.passed
        assert report.n_trials == 20
        assert report.n_seeds == 20
