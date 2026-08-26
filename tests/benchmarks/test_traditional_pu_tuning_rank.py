"""Ranking/elimination tests for the tuning-round pipeline (plan §4).

Config generation and degenerate-rate tests live in
test_traditional_pu_tuning_round.py; shared builders in tuning_helpers.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.traditional_pu.tuning_round import rank_candidates
from tests.benchmarks.tuning_helpers import (
    SCENARIOS,
    full_grid,
    make_row,
    write_candidate_trials,
)

pytestmark = pytest.mark.unit


class TestRankCandidates:
    def test_basic_rank_orders_survivors_by_risk_mean(self, tmp_path):
        for name, risk in [("a", 0.3), ("b", 0.1), ("c", 0.2)]:
            write_candidate_trials(tmp_path, name, full_grid({s: risk for s in SCENARIOS}))
        table = rank_candidates(tmp_path, ["a", "b", "c"])
        assert list(table["name"]) == ["a", "b", "c"]  # input order kept
        ranked = table[table["selected"]].sort_values("rank")
        assert list(ranked["name"]) == ["b", "c", "a"]
        assert (table["eliminated_reason"].isna()).all()
        assert (table["n_total"] == 30).all() and (table["n_success"] == 30).all()

    def test_param_rank_eliminates_degenerate_before_ranking(self, tmp_path):
        # Lowest risk but degenerate → eliminated despite ranking first.
        write_candidate_trials(
            tmp_path,
            "degenerate",
            full_grid(
                {s: 0.01 for s in SCENARIOS}, degenerate_scenarios=("scar-pi0.1-scalesmall",)
            ),
        )
        write_candidate_trials(tmp_path, "clean", full_grid({s: 0.5 for s in SCENARIOS}))
        table = rank_candidates(tmp_path, ["degenerate", "clean"])
        row = table[table["name"] == "degenerate"].iloc[0]
        assert row["eliminated_reason"] == "degenerate_prediction"
        assert "scar-pi0.1-scalesmall" in row["degenerate_cells"]
        assert not row["selected"]

    def test_param_rank_eliminates_sub100_success(self, tmp_path):
        write_candidate_trials(
            tmp_path, "flaky", full_grid(failed_cells={("scar-pi0.5-scalemid", 4)})
        )
        table = rank_candidates(tmp_path, ["flaky"])
        row = table.iloc[0]
        assert row["eliminated_reason"] == "success_rate < 100%"
        assert row["n_success"] == 29
        assert not row["selected"]

    def test_param_rank_n_select_takes_top_k(self, tmp_path):
        for name, risk in [("a", 0.4), ("b", 0.3), ("c", 0.2), ("d", 0.1)]:
            write_candidate_trials(tmp_path, name, full_grid({s: risk for s in SCENARIOS}))
        table = rank_candidates(tmp_path, ["a", "b", "c", "d"], n_select=2)
        ranked = table[table["selected"]].sort_values("rank")
        assert list(ranked["name"]) == ["d", "c"]

    def test_edge_rank_missing_trials_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ghost"):
            rank_candidates(tmp_path, ["ghost"])

    def test_edge_rank_rejects_scar_row_deficit_and_excludes_non_success_risk(self, tmp_path):
        # 29 scar rows + 1 sar row → scar-count violation (sar rows are
        # only legal as a complete 30-row diagnostic block).
        sar_row = make_row("sar-pi0.5-scalesmall", 0)
        write_candidate_trials(tmp_path, "mixed", [sar_row] + full_grid()[:29])
        with pytest.raises(ValueError, match="scar dev rows"):
            rank_candidates(tmp_path, ["mixed"])
        # non-success rows never feed the risk mean
        rows = full_grid({s: 0.7 for s in SCENARIOS}, failed_cells={("scar-pi0.1-scalemid", 2)})
        rows_with_nan = [
            {**r, "pu_zero_one_risk": np.nan if r["status"] == "failed" else r["pu_zero_one_risk"]}
            for r in rows
        ]
        write_candidate_trials(tmp_path, "nanrisk", rows_with_nan)
        table = rank_candidates(tmp_path, ["nanrisk"])
        assert table.iloc[0]["risk_mean"] == pytest.approx(0.7)

    def test_param_rank_mixed_scar_sar_filters_on_scar_rows_only(self, tmp_path):
        # nnPU-class methods run the 6-cell SAR diagnostic line too, so dev
        # trials carry 60 rows.  The elimination chain and the risk mean
        # must use the 30 scar rows only; sar flaws are diagnostics.
        sar_rows = []
        for pi in ("0.1", "0.3", "0.5"):
            for scale in ("small", "mid"):
                for seed in range(5):
                    sar_rows.append(make_row(f"sar-pi{pi}-scale{scale}", seed, risk=0.01))
        for i, row in enumerate(sar_rows):
            if i < 3:
                row["status"] = "failed"
            elif i < 6:
                row["degenerate_prediction"] = True
        scar_rows = full_grid({s: 0.5 for s in SCENARIOS})
        write_candidate_trials(tmp_path, "mixed", scar_rows + sar_rows)
        table = rank_candidates(tmp_path, ["mixed"])
        row = table.iloc[0]
        assert row["n_total"] == 60
        assert row["n_success"] == 30  # scar rows only
        assert row["sar_n_total"] == 30 and row["sar_n_success"] == 27
        assert row["risk_mean"] == pytest.approx(0.5)  # sar 0.01 risk excluded
        assert row["eliminated_reason"] is None
        assert row["selected"]

    def test_edge_rank_rejects_unknown_scenario_prefix(self, tmp_path):
        bogus_row = make_row("pnu-1-1-4-scalesmall", 0)
        write_candidate_trials(tmp_path, "pnu", full_grid()[:29] + [bogus_row])
        with pytest.raises(ValueError, match="unknown scenario prefix"):
            rank_candidates(tmp_path, ["pnu"])

    def test_determ_rank_same_input_same_output(self, tmp_path):
        for name, risk in [("a", 0.4), ("b", 0.3), ("c", 0.2)]:
            write_candidate_trials(tmp_path, name, full_grid({s: risk for s in SCENARIOS}))
        first = rank_candidates(tmp_path, ["a", "b", "c"])
        second = rank_candidates(tmp_path, ["a", "b", "c"])
        pd.testing.assert_frame_equal(first, second)
