"""Tuning-round plumbing tests (optimization plan §8 step 5).

Covers config generation for one-parameter-cluster candidate rounds and the
degenerate-rate comparison (plan §5 condition 3).  Ranking/elimination tests
live in test_traditional_pu_tuning_rank.py.  Verdicts (paired CIs, budget)
stay in compare.py and are not re-implemented here.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.traditional_pu.tuning_round import (
    compare_degenerate_condition,
    degenerate_rates,
    generate_round_configs,
)
from tests.benchmarks.tuning_helpers import (
    NNPU_BASE_PARAMS,
    SCENARIOS,
    UPU_BASE_PARAMS,
    base_config_path,
    full_grid,
    make_row,
    write_candidate_trials,
)

pytestmark = pytest.mark.unit


class TestGenerateConfigs:
    def test_basic_generate_produces_kldce_only_config(self, tmp_path):
        base = base_config_path(tmp_path)
        out = tmp_path / "round"
        paths = generate_round_configs(base, out, [("c1", {"reg_strength": 0.1})])
        assert paths == [out / "c1.json"]
        produced = json.loads(paths[0].read_text(encoding="utf-8"))
        assert set(produced["methods"]) == {"kldce"}
        assert produced["data"] is not None and produced["timeouts"] == {"kldce": 120}
        assert produced["limitations"] == []

    def test_param_generate_merges_overrides_and_records_description(self, tmp_path):
        base = base_config_path(tmp_path)
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
        base = base_config_path(tmp_path, methods=methods)
        with pytest.raises(ValueError, match="constructor default"):
            generate_round_configs(base, tmp_path / "round", [("c1", {})])

    def test_edge_generate_rejects_runner_injected_and_unknown_keys(self, tmp_path):
        base = base_config_path(tmp_path)
        with pytest.raises(ValueError, match="runner-injected"):
            generate_round_configs(base, tmp_path / "round", [("c1", {"flip_probability": 0.3})])
        with pytest.raises(ValueError, match="not a constructor parameter"):
            generate_round_configs(base, tmp_path / "round", [("c1", {"bogus_param": 1})])

    def test_edge_generate_rejects_duplicate_names(self, tmp_path):
        base = base_config_path(tmp_path)
        with pytest.raises(ValueError, match="duplicate"):
            generate_round_configs(
                base,
                tmp_path / "round",
                [("c1", {}), ("c1", {"reg_strength": 0.1})],
            )

    def test_determ_generate_same_input_same_output(self, tmp_path):
        base = base_config_path(tmp_path)
        first = generate_round_configs(base, tmp_path / "a", [("c1", {"reg_strength": 0.1})])
        second = generate_round_configs(base, tmp_path / "b", [("c1", {"reg_strength": 0.1})])
        first_cfg = json.loads(first[0].read_text(encoding="utf-8"))
        second_cfg = json.loads(second[0].read_text(encoding="utf-8"))
        assert first_cfg == second_cfg

    def test_basic_upu_generate_with_rbf_overrides(self, tmp_path):
        base = base_config_path(
            tmp_path,
            methods={"upu": dict(UPU_BASE_PARAMS)},
            timeouts={"upu": 120},
        )
        paths = generate_round_configs(
            base,
            tmp_path / "round",
            [("c1", {"basis": "rbf", "kernel_width": 0.5})],
            method="upu",
        )
        produced = json.loads(paths[0].read_text(encoding="utf-8"))
        assert set(produced["methods"]) == {"upu"}
        params = produced["methods"]["upu"]
        assert params["basis"] == "rbf" and params["kernel_width"] == 0.5
        assert params["loss"] == "double_hinge"  # untouched default kept
        assert "locks_source_defaults" not in produced

    def test_basic_nnpu_generate_keeps_non_tunable_nulls(self, tmp_path):
        base = base_config_path(
            tmp_path,
            methods={"nnpu": dict(NNPU_BASE_PARAMS)},
            timeouts={"nnpu": 120},
        )
        paths = generate_round_configs(
            base, tmp_path / "round", [("c1", {"beta": 0.5, "batch_size": 64})], method="nnpu"
        )
        produced = json.loads(paths[0].read_text(encoding="utf-8"))
        assert set(produced["methods"]) == {"nnpu"}
        params = produced["methods"]["nnpu"]
        assert params["beta"] == 0.5 and params["batch_size"] == 64
        assert params["gamma"] == 1.0  # untouched default kept
        # JSON-inexpressible params stay at their base nulls, not overridden.
        assert params["model"] is None and params["optimizer"] is None
        assert params["device"] is None
        assert "locks_source_defaults" not in produced

    def test_edge_nnpu_rejects_non_tunable_overrides(self, tmp_path):
        base = base_config_path(tmp_path, methods={"nnpu": dict(NNPU_BASE_PARAMS)})
        with pytest.raises(ValueError, match="not tunable"):
            generate_round_configs(
                base, tmp_path / "round", [("c1", {"optimizer": "adam"})], method="nnpu"
            )
        with pytest.raises(ValueError, match="not tunable"):
            generate_round_configs(
                base, tmp_path / "round", [("c1", {"device": "cuda"})], method="nnpu"
            )

    def test_edge_nnpu_missing_non_tunable_keys_pass_coverage_check(self, tmp_path):
        # model/optimizer/device are excluded from the constructor-defaults
        # coverage check — a base missing them still generates (their
        # constructor defaults of None are the correct values).
        methods = {
            "nnpu": {
                k: v
                for k, v in NNPU_BASE_PARAMS.items()
                if k not in ("model", "optimizer", "device")
            }
        }
        base = base_config_path(tmp_path, methods=methods)
        paths = generate_round_configs(base, tmp_path / "round", [("c1", {})], method="nnpu")
        produced = json.loads(paths[0].read_text(encoding="utf-8"))
        assert "model" not in produced["methods"]["nnpu"]


class TestDegenerateRates:
    def test_basic_degenerate_rates_counts_success_only(self, tmp_path):
        rows = full_grid(
            degenerate_scenarios=("scar-pi0.1-scalesmall",),
            failed_cells={("scar-pi0.5-scalemid", 4)},
        )
        write_candidate_trials(tmp_path, "cand", rows)
        rates = degenerate_rates(tmp_path / "cand")
        cell = rates[rates["scenario"] == "scar-pi0.1-scalesmall"].iloc[0]
        assert cell["n_success"] == 5 and cell["n_degenerate"] == 5
        assert cell["degenerate_rate"] == 1.0
        failed_cell = rates[rates["scenario"] == "scar-pi0.5-scalemid"].iloc[0]
        assert failed_cell["n_success"] == 4  # denominator excludes the failed row
        assert failed_cell["n_degenerate"] == 0

    def test_edge_degenerate_rates_missing_column_marks_unknown(self, tmp_path):
        rows = [
            {k: v for k, v in make_row(scenario, seed).items() if k != "degenerate_prediction"}
            for scenario in SCENARIOS[:1]
            for seed in range(2)
        ]
        write_candidate_trials(tmp_path, "old", rows)
        rates = degenerate_rates(tmp_path / "old")
        assert rates["degenerate_rate"].isna().all()
        assert (rates["degenerate_column"] == False).all()  # noqa: E712 - explicit unknown

    def test_param_degenerate_condition_semantics(self, tmp_path):
        for name, degenerate in [("base", False), ("cand", False)]:
            write_candidate_trials(
                tmp_path,
                name,
                full_grid(degenerate_scenarios=("scar-pi0.1-scalesmall",) if degenerate else ()),
            )
        # Both known and candidate is no worse → passes.
        result = compare_degenerate_condition(tmp_path / "base", tmp_path / "cand")
        assert result["passes"] is True
        # Baseline column missing (pre-step-4 artifact) → unknown, not False.
        old_rows = [
            {k: v for k, v in make_row(s, seed).items() if k != "degenerate_prediction"}
            for s in SCENARIOS[:1]
            for seed in range(2)
        ]
        write_candidate_trials(tmp_path, "oldbase", old_rows)
        result = compare_degenerate_condition(tmp_path / "oldbase", tmp_path / "cand")
        assert result["passes"] is None
        assert "companion" in result["reason"]
