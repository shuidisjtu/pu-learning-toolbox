"""Contract §6 paired-verdict tool tests: pairing, direction, budget, alignment."""

from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.traditional_pu.compare import (
    alignment_is_valid,
    pair_cells,
    verdicts,
)

pytestmark = pytest.mark.unit

SCENARIO = "scar-pi0.1-scalemid"
SEEDS = [100, 101, 102, 103, 104]


def _trials(metric_values: dict[int, float], *, status: str = "success") -> pd.DataFrame:
    rows = []
    for seed, value in metric_values.items():
        rows.append(
            {
                "algorithm": "ldce",
                "scenario": SCENARIO,
                "seed": seed,
                "status": status,
                "elapsed_seconds": 5.0,
                "pu_zero_one_risk": value,
            }
        )
    return pd.DataFrame(rows)


class TestPairing:
    def test_param_same_seed_success_pairs_only(self):
        baseline = _trials({100: 1.0, 101: 1.2})
        candidate = _trials({100: 0.8, 101: 1.4, 102: 0.5})
        table = pair_cells(baseline, candidate, "pu_zero_one_risk")
        row = table.iloc[0]
        assert row["n_paired"] == 2  # seed 102 has no baseline success → unpaired
        assert row["baseline_mean"] == pytest.approx(1.1)
        assert row["candidate_mean"] == pytest.approx(1.1)
        # paired diffs: (0.8-1.0, 1.4-1.2) = (-0.2, +0.2)
        assert row["diff_mean"] == pytest.approx(0.0)
        assert bool(row["paired_available"]) is True

    def test_param_no_baseline_values_reports_unpaired(self):
        baseline = _trials({100: 1.0}, status="nonconverged")
        candidate = _trials({100: 0.5})
        table = pair_cells(baseline, candidate, "pu_zero_one_risk")
        row = table.iloc[0]
        assert int(row["n_paired"]) == 0
        assert bool(row["paired_available"]) is False
        assert row["baseline_mean"] != row["baseline_mean"]  # NaN
        assert row["candidate_success_rate"] == 1.0

    def test_param_metric_direction_drives_cond1(self):
        # lower-is-better metric: negative paired CI ⇒ improvement
        improved = pair_cells(
            _trials({100: 2.0, 101: 2.0}), _trials({100: 1.0, 101: 1.0}), "pu_zero_one_risk"
        )
        assert bool(improved.iloc[0]["cond1_improves"]) is True
        worsened = pair_cells(
            _trials({100: 1.0, 101: 1.0}), _trials({100: 2.0, 101: 2.0}), "pu_zero_one_risk"
        )
        assert bool(worsened.iloc[0]["cond1_improves"]) is False


def _trials_with_oracle(risks: dict[int, float], aucs: dict[int, float]) -> pd.DataFrame:
    """Trials carrying both the PU selection metric and an oracle metric."""
    rows = []
    for seed, risk in risks.items():
        rows.append(
            {
                "algorithm": "ldce",
                "scenario": SCENARIO,
                "seed": seed,
                "status": "success",
                "elapsed_seconds": 5.0,
                "pu_zero_one_risk": risk,
                "pu_auc_roc": aucs[seed],
            }
        )
    return pd.DataFrame(rows)


class TestOracleClassification:
    def test_param_oracle_only_improvement_labelled(self):
        """Optimization plan §5: oracle improves while the PU metric does not."""
        baseline = _trials_with_oracle({100: 1.0, 101: 1.0}, {100: 0.5, 101: 0.5})
        candidate = _trials_with_oracle({100: 1.0, 101: 1.0}, {100: 0.9, 101: 0.9})
        pu = pair_cells(baseline, candidate, "pu_zero_one_risk")
        oracle = pair_cells(baseline, candidate, "pu_auc_roc")
        table = pu.merge(
            oracle[["algorithm", "scenario", "cond1_improves", "paired_available"]].rename(
                columns={
                    "cond1_improves": "oracle_cond1_improves",
                    "paired_available": "oracle_paired_available",
                }
            ),
            on=["algorithm", "scenario"],
        )
        out = verdicts(table, budget_seconds=120.0)
        assert bool(out.iloc[0]["oracle_cond1_improves"]) is True
        assert bool(out.iloc[0]["oracle_only_improvement"]) is True
        assert bool(out.iloc[0]["confirmed_improvement"]) is False

    def test_param_pu_improvement_is_not_oracle_only(self):
        """A PU-metric improvement stays confirmed, never oracle-only."""
        baseline = _trials_with_oracle({100: 1.0, 101: 1.0}, {100: 0.5, 101: 0.5})
        candidate = _trials_with_oracle({100: 0.5, 101: 0.5}, {100: 0.9, 101: 0.9})
        pu = pair_cells(baseline, candidate, "pu_zero_one_risk")
        oracle = pair_cells(baseline, candidate, "pu_auc_roc")
        table = pu.merge(
            oracle[["algorithm", "scenario", "cond1_improves", "paired_available"]].rename(
                columns={
                    "cond1_improves": "oracle_cond1_improves",
                    "paired_available": "oracle_paired_available",
                }
            ),
            on=["algorithm", "scenario"],
        )
        out = verdicts(table, budget_seconds=120.0)
        assert bool(out.iloc[0]["oracle_only_improvement"]) is False
        assert bool(out.iloc[0]["confirmed_improvement"]) is True

    def test_edge_verdicts_without_oracle_columns_yields_false(self):
        """verdicts stays usable when oracle columns are absent (old tables)."""
        table = pd.DataFrame(
            [
                {
                    "algorithm": "ldce",
                    "scenario": SCENARIO,
                    "candidate_p95_elapsed_seconds": 10.0,
                    "candidate_success_rate": 1.0,
                    "baseline_success_rate": 1.0,
                    "cond1_improves": True,
                    "paired_available": True,
                }
            ]
        )
        out = verdicts(table, budget_seconds=120.0)
        assert bool(out.iloc[0]["oracle_only_improvement"]) is False
        assert bool(out.iloc[0]["confirmed_improvement"]) is True


class TestVerdicts:
    def test_param_budget_gates_confirmation(self):
        table = pd.DataFrame(
            [
                {
                    "algorithm": "ldce",
                    "scenario": SCENARIO,
                    "candidate_p95_elapsed_seconds": 130.0,
                    "candidate_success_rate": 1.0,
                    "baseline_success_rate": 0.8,
                    "cond1_improves": True,
                    "paired_available": True,
                }
            ]
        )
        out = verdicts(table, budget_seconds=120.0)
        assert bool(out.iloc[0]["cond3_budget_ok"]) is False
        assert bool(out.iloc[0]["cond2_success_ok"]) is True
        assert bool(out.iloc[0]["confirmed_improvement"]) is False

    def test_param_success_rate_tolerance_recorded(self):
        table = pd.DataFrame(
            [
                {
                    "algorithm": "ldce",
                    "scenario": SCENARIO,
                    "candidate_p95_elapsed_seconds": 10.0,
                    "candidate_success_rate": 0.80,
                    "baseline_success_rate": 0.84,  # only 4pp below → tolerated
                    "cond1_improves": True,
                    "paired_available": True,
                }
            ]
        )
        out = verdicts(table, budget_seconds=120.0)
        assert bool(out.iloc[0]["cond2_success_ok"]) is True

    def test_edge_alignment_detects_protocol_drift(self):
        baseline = _trials({100: 1.0})
        candidate = _trials({200: 1.0})
        assert alignment_is_valid(baseline, candidate) is False


class TestCli:
    def test_basic_main_writes_verdict_csv(self, tmp_path):
        from benchmarks.traditional_pu.compare import main as compare_main

        base_dir = tmp_path / "base"
        cand_dir = tmp_path / "cand"
        base_dir.mkdir()
        cand_dir.mkdir()
        _trials({100: 1.0, 101: 1.0}).to_csv(base_dir / "trials.csv", index=False)
        _trials({100: 0.5, 101: 0.5}).to_csv(cand_dir / "trials.csv", index=False)
        out = tmp_path / "verdict.csv"
        assert (
            compare_main(
                [
                    "--baseline-dir",
                    str(base_dir),
                    "--candidate-dir",
                    str(cand_dir),
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        assert out.exists()

    def test_basic_main_classifies_oracle_only(self, tmp_path):
        """CLI emits the oracle_only_improvement column when trials carry it.

        No PU-metric improvement → exit code 1 (the gate semantics), but the
        verdict table still classifies the cell as oracle-only.
        """
        from benchmarks.traditional_pu.compare import main as compare_main

        base_dir = tmp_path / "base"
        cand_dir = tmp_path / "cand"
        base_dir.mkdir()
        cand_dir.mkdir()
        _trials_with_oracle({100: 1.0, 101: 1.0}, {100: 0.5, 101: 0.5}).to_csv(
            base_dir / "trials.csv", index=False
        )
        _trials_with_oracle({100: 1.0, 101: 1.0}, {100: 0.9, 101: 0.9}).to_csv(
            cand_dir / "trials.csv", index=False
        )
        out = tmp_path / "verdict.csv"
        assert (
            compare_main(
                [
                    "--baseline-dir",
                    str(base_dir),
                    "--candidate-dir",
                    str(cand_dir),
                    "--out",
                    str(out),
                ]
            )
            == 1
        )
        table = pd.read_csv(out)
        assert "oracle_only_improvement" in table.columns
        assert bool(table.iloc[0]["oracle_only_improvement"]) is True
        assert bool(table.iloc[0]["confirmed_improvement"]) is False
