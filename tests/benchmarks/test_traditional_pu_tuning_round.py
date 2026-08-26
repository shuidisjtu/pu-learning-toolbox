"""Tuning-round plumbing tests (optimization plan §8 step 5).

Covers config generation for one-parameter-cluster candidate rounds, the
dev-stage ranking/elimination chain (plan §4) and the degenerate-rate
comparison (plan §5 condition 3).  Verdicts (paired CIs, budget) stay in
compare.py and are not re-implemented here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.traditional_pu.tuning_round import (
    compare_degenerate_condition,
    degenerate_rates,
    generate_round_configs,
    rank_candidates,
)

pytestmark = pytest.mark.unit

SCENARIOS = [
    "scar-pi0.1-scalesmall",
    "scar-pi0.1-scalemid",
    "scar-pi0.3-scalesmall",
    "scar-pi0.3-scalemid",
    "scar-pi0.5-scalesmall",
    "scar-pi0.5-scalemid",
]


def _base_config_path(tmp_path, *, methods=None):
    """Persist a baseline-like config (locked kldce defaults) and return its path."""
    cfg = {
        "schema_version": 1,
        "protocol": "traditional_pu_baseline",
        "description": "base",
        "seed_set_development": [0, 1],
        "seed_set_confirmation": [2, 3],
        "data": {
            "n_samples_small": 50,
            "n_samples_mid": 50,
            "n_features": 5,
            "class_priors": [0.5],
            "separation": 2.0,
            "label_frequency": 0.5,
        },
        "methods": methods
        or {
            "kldce": {
                "sigma": "scale",
                "reg_strength": 1.0,
                "centroid_radius": 1.0,
                "mom_groups": 10,
                "covariance_ridge": 0.0,
                "max_acs_iter": 50,
                "max_inner_iter": 2000,
                "inner_tol": 1e-6,
                "tol": 1e-6,
            }
        },
        "timeouts": {"kldce": 120},
        "limitations": [],
        "locks_source_defaults": True,
    }
    path = tmp_path / "base.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _make_row(scenario, seed, *, status="success", degenerate=False, risk=0.5, elapsed=1.0):
    return {
        "algorithm": "kldce",
        "scenario": scenario,
        "seed": seed,
        "status": status,
        "degenerate_prediction": degenerate,
        "pu_zero_one_risk": risk,
        "elapsed_seconds": elapsed,
    }


def _write_candidate_trials(root, name, rows):
    """Persist a candidate's trials.csv under <root>/<name>/."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "trials.csv", index=False)


def _full_grid(risk_per_scenario=None, *, degenerate_scenarios=(), failed_cells=()):
    """30 rows (6 scenarios × 5 seeds) with per-scenario risk and planted flaws."""
    risk_per_scenario = risk_per_scenario or {}
    rows = []
    for scenario in SCENARIOS:
        for seed in range(5):
            status = "failed" if (scenario, seed) in failed_cells else "success"
            rows.append(
                _make_row(
                    scenario,
                    seed,
                    status=status,
                    degenerate=scenario in degenerate_scenarios and status == "success",
                    risk=risk_per_scenario.get(scenario, 0.5),
                )
            )
    return rows


class TestGenerateConfigs:
    def test_basic_generate_produces_kldce_only_config(self, tmp_path):
        base = _base_config_path(tmp_path)
        out = tmp_path / "round"
        paths = generate_round_configs(base, out, [("c1", {"reg_strength": 0.1})])
        assert paths == [out / "c1.json"]
        produced = json.loads(paths[0].read_text(encoding="utf-8"))
        assert set(produced["methods"]) == {"kldce"}
        assert produced["data"] is not None and produced["timeouts"] == {"kldce": 120}
        assert produced["limitations"] == []

    def test_param_generate_merges_overrides_and_records_description(self, tmp_path):
        base = _base_config_path(tmp_path)
        out = tmp_path / "round"
        paths = generate_round_configs(
            base, out, [("c1", {"reg_strength": 0.1, "centroid_radius": 10.0})]
        )
        produced = json.loads(paths[0].read_text(encoding="utf-8"))
        params = produced["methods"]["kldce"]
        assert params["reg_strength"] == 0.1
        assert params["centroid_radius"] == 10.0
        assert params["covariance_ridge"] == 0.0  # untouched defaults kept
        assert "c1" in produced["description"]
        assert "reg_strength" in produced["description"]
        assert "locks_source_defaults" not in produced

    def test_param_generate_covers_full_constructor_defaults(self, tmp_path):
        # A base whose method params miss a constructor default must be
        # rejected — merged configs must never be under-specified.
        methods = {
            "kldce": {
                "sigma": "scale",
                "reg_strength": 1.0,
                # centroid_radius / mom_groups / ... missing
            }
        }
        base = _base_config_path(tmp_path, methods=methods)
        with pytest.raises(ValueError, match="constructor default"):
            generate_round_configs(base, tmp_path / "round", [("c1", {})])

    def test_edge_generate_rejects_runner_injected_and_unknown_keys(self, tmp_path):
        base = _base_config_path(tmp_path)
        with pytest.raises(ValueError, match="runner-injected"):
            generate_round_configs(base, tmp_path / "round", [("c1", {"flip_probability": 0.3})])
        with pytest.raises(ValueError, match="not a constructor parameter"):
            generate_round_configs(base, tmp_path / "round", [("c1", {"bogus_param": 1})])

    def test_edge_generate_rejects_duplicate_names(self, tmp_path):
        base = _base_config_path(tmp_path)
        with pytest.raises(ValueError, match="duplicate"):
            generate_round_configs(
                base,
                tmp_path / "round",
                [("c1", {}), ("c1", {"reg_strength": 0.1})],
            )


class TestRankCandidates:
    def test_basic_rank_orders_survivors_by_risk_mean(self, tmp_path):
        for name, risk in [("a", 0.3), ("b", 0.1), ("c", 0.2)]:
            _write_candidate_trials(tmp_path, name, _full_grid({s: risk for s in SCENARIOS}))
        table = rank_candidates(tmp_path, ["a", "b", "c"])
        assert list(table["name"]) == ["a", "b", "c"]  # input order kept
        ranked = table[table["selected"]].sort_values("rank")
        assert list(ranked["name"]) == ["b", "c", "a"]
        assert (table["eliminated_reason"].isna()).all()
        assert (table["n_total"] == 30).all() and (table["n_success"] == 30).all()

    def test_param_rank_eliminates_degenerate_before_ranking(self, tmp_path):
        # Lowest risk but degenerate → eliminated despite ranking first.
        _write_candidate_trials(
            tmp_path,
            "degenerate",
            _full_grid(
                {s: 0.01 for s in SCENARIOS}, degenerate_scenarios=("scar-pi0.1-scalesmall",)
            ),
        )
        _write_candidate_trials(tmp_path, "clean", _full_grid({s: 0.5 for s in SCENARIOS}))
        table = rank_candidates(tmp_path, ["degenerate", "clean"])
        row = table[table["name"] == "degenerate"].iloc[0]
        assert row["eliminated_reason"] == "degenerate_prediction"
        assert "scar-pi0.1-scalesmall" in row["degenerate_cells"]
        assert not row["selected"]

    def test_param_rank_eliminates_sub100_success(self, tmp_path):
        _write_candidate_trials(
            tmp_path, "flaky", _full_grid(failed_cells={("scar-pi0.5-scalemid", 4)})
        )
        table = rank_candidates(tmp_path, ["flaky"])
        row = table.iloc[0]
        assert row["eliminated_reason"] == "success_rate < 100%"
        assert row["n_success"] == 29
        assert not row["selected"]

    def test_param_rank_n_select_takes_top_k(self, tmp_path):
        for name, risk in [("a", 0.4), ("b", 0.3), ("c", 0.2), ("d", 0.1)]:
            _write_candidate_trials(tmp_path, name, _full_grid({s: risk for s in SCENARIOS}))
        table = rank_candidates(tmp_path, ["a", "b", "c", "d"], n_select=2)
        ranked = table[table["selected"]].sort_values("rank")
        assert list(ranked["name"]) == ["d", "c"]

    def test_edge_rank_missing_trials_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ghost"):
            rank_candidates(tmp_path, ["ghost"])

    def test_edge_rank_rejects_sar_scenario_and_excludes_non_success_risk(self, tmp_path):
        sar_row = _make_row("sar-pi0.5-scalesmall", 0)
        _write_candidate_trials(tmp_path, "sarred", [sar_row] + _full_grid()[:29])
        with pytest.raises(ValueError, match="scar"):
            rank_candidates(tmp_path, ["sarred"])
        # non-success rows never feed the risk mean
        rows = _full_grid({s: 0.7 for s in SCENARIOS}, failed_cells={("scar-pi0.1-scalemid", 2)})
        rows_with_nan = [
            {**r, "pu_zero_one_risk": np.nan if r["status"] == "failed" else r["pu_zero_one_risk"]}
            for r in rows
        ]
        _write_candidate_trials(tmp_path, "nanrisk", rows_with_nan)
        table = rank_candidates(tmp_path, ["nanrisk"])
        assert table.iloc[0]["risk_mean"] == pytest.approx(0.7)

    def test_determ_rank_same_input_same_output(self, tmp_path):
        for name, risk in [("a", 0.4), ("b", 0.3), ("c", 0.2)]:
            _write_candidate_trials(tmp_path, name, _full_grid({s: risk for s in SCENARIOS}))
        first = rank_candidates(tmp_path, ["a", "b", "c"])
        second = rank_candidates(tmp_path, ["a", "b", "c"])
        pd.testing.assert_frame_equal(first, second)


class TestDegenerateRates:
    def test_basic_degenerate_rates_counts_success_only(self, tmp_path):
        rows = _full_grid(
            degenerate_scenarios=("scar-pi0.1-scalesmall",),
            failed_cells={("scar-pi0.5-scalemid", 4)},
        )
        _write_candidate_trials(tmp_path, "cand", rows)
        rates = degenerate_rates(tmp_path / "cand")
        cell = rates[rates["scenario"] == "scar-pi0.1-scalesmall"].iloc[0]
        assert cell["n_success"] == 5 and cell["n_degenerate"] == 5
        assert cell["degenerate_rate"] == 1.0
        failed_cell = rates[rates["scenario"] == "scar-pi0.5-scalemid"].iloc[0]
        assert failed_cell["n_success"] == 4  # denominator excludes the failed row
        assert failed_cell["n_degenerate"] == 0

    def test_edge_degenerate_rates_missing_column_marks_unknown(self, tmp_path):
        rows = [
            {k: v for k, v in _make_row(scenario, seed).items() if k != "degenerate_prediction"}
            for scenario in SCENARIOS[:1]
            for seed in range(2)
        ]
        _write_candidate_trials(tmp_path, "old", rows)
        rates = degenerate_rates(tmp_path / "old")
        assert rates["degenerate_rate"].isna().all()
        assert (rates["degenerate_column"] == False).all()  # noqa: E712 - explicit unknown

    def test_param_degenerate_condition_semantics(self, tmp_path):
        for name, degenerate in [("base", False), ("cand", False)]:
            _write_candidate_trials(
                tmp_path,
                name,
                _full_grid(degenerate_scenarios=("scar-pi0.1-scalesmall",) if degenerate else ()),
            )
        # Both known and candidate is no worse → passes.
        result = compare_degenerate_condition(tmp_path / "base", tmp_path / "cand")
        assert result["passes"] is True
        # Baseline column missing (pre-step-4 artifact) → unknown, not False.
        old_rows = [
            {k: v for k, v in _make_row(s, seed).items() if k != "degenerate_prediction"}
            for s in SCENARIOS[:1]
            for seed in range(2)
        ]
        _write_candidate_trials(tmp_path, "oldbase", old_rows)
        result = compare_degenerate_condition(tmp_path / "oldbase", tmp_path / "cand")
        assert result["passes"] is None
        assert "companion" in result["reason"]
