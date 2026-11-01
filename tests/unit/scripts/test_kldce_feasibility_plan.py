"""Torch-free KLDCE applicability preview agrees with core Phase-I feasibility."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit
PATH = Path(__file__).resolve().parents[3] / "scripts/prepare_kldce_resource_plan.py"
SPEC = importlib.util.spec_from_file_location("kldce_feasibility_plan_test", PATH)
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def plan(**overrides):
    return planner.build_plan(
        **(
            dict(
                n_positive=6,
                n_unlabeled=18,
                feature_count=3,
                gram_itemsize=8,
                prior_policy="derived",
                flip_probability=0.2,
            )
            | overrides
        )
    )


def test_basic_determ_prior_and_scalar_resource_preview_do_not_approve_fit():
    planned = plan()
    assert planned == plan()
    screen = planned["qp_feasibility_screen"]
    assert screen["effective_training_prior"] == 6 / (24 * 0.8)
    assert screen["feasible"] and screen["reason"] == "equality_box_feasible"
    assert not screen["SCAR_generator_binding_verified"]
    assert not planned["recipe_validation_performed"] and not planned["ready_for_formal_run"]


@pytest.mark.parametrize(
    "flip,prior,feasible",
    [(0.2, 0.45, True), (0.5, 0.5, True), (0.8, 0.5, False), (0.9, 0.9, True), (0.9, 0.8, False)],
)
def test_param_actual_phase_one_lp_matches_preview(flip, prior, feasible):
    from pu_toolbox.estimators.risk.kldce import _find_feasible_init

    screen = plan(
        prior_policy="population_override", flip_probability=flip, population_prior=prior
    )["qp_feasibility_screen"]
    assert screen["feasible"] is feasible
    coefficients = np.r_[np.ones(6), -np.ones(18), np.ones(18)].reshape(1, -1)
    lower, upper = np.zeros(42), np.r_[np.full(24, 1 / 24), np.full(18, 1 / 48)]
    if feasible:
        point = _find_feasible_init(coefficients, screen["rhs"], lower, upper)
        assert float((coefficients @ point).item()) == pytest.approx(screen["rhs"])
    else:
        with pytest.raises(RuntimeError, match="infeasible"):
            _find_feasible_init(coefficients, screen["rhs"], lower, upper)


def test_edge_near_zero_denominator_has_no_fabricated_finite_rhs():
    screen = plan(prior_policy="population_override", flip_probability=0.8, population_prior=0.625)[
        "qp_feasibility_screen"
    ]
    assert screen["rhs"] is None and not screen["feasible"]
    assert screen["reason"] == "near_zero_centroid_denominator"


@pytest.mark.parametrize(
    "overrides",
    [
        dict(prior_policy=None),
        dict(flip_probability=True),
        dict(flip_probability=0),
        dict(flip_probability=0.9),
        dict(population_prior=0.4),
        dict(prior_policy="population_override"),
        dict(prior_policy="population_override", population_prior=True),
    ],
)
def test_param_invalid_or_ambiguous_prior_assumptions_refused(overrides):
    with pytest.raises(ValueError):
        plan(**overrides)


def test_edge_cli_infeasible_exit_code_uses_only_stdlib_not_model_or_numerical_imports():
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'scipy', 'sklearn', 'pu_toolbox'}:
        raise AssertionError('planner imported numerical/model code: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--n-positive', '6', '--n-unlabeled', '18', '--feature-count', '3',
            '--gram-itemsize', '8', '--prior-policy', 'population_override',
            '--flip-probability', '0.7', '--population-prior', '0.5']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code, str(PATH)], capture_output=True, text=True)
    assert result.returncode == 3
    planned = json.loads(result.stdout)
    assert planned["qp_feasibility_screen"]["reason"] == "equality_outside_box"
    assert not planned["estimator_constructed"] and not planned["cuda_initialized"]
