"""Read-only KLDCE dense-array payload screening, not a peak-RAM or fit approval."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "pu_toolbox/estimators/risk/kldce.py"


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _defaults(root):
    tree = ast.parse((root / SOURCE).read_text(encoding="utf-8"))
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KLDCEClassifier"
    )
    init = next(
        node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    return {
        arg.arg: ast.literal_eval(value)
        for arg, value in zip(init.args.kwonlyargs, init.args.kw_defaults, strict=True)
        if value is not None
    }


def _feasibility(n_positive, n_unlabeled, policy, flip, population_prior):
    """Scalar counterpart of the core Phase-I LP; all inputs are caller declarations."""
    if policy not in ("derived", "population_override"):
        raise ValueError("explicit prior_policy required for the feasibility screen")
    if (
        isinstance(flip, bool)
        or not isinstance(flip, int | float)
        or not math.isfinite(flip)
        or not 0 < flip < 1
    ):
        raise ValueError("flip_probability must be a finite real in (0,1)")
    n = n_positive + n_unlabeled
    if policy == "derived":
        if population_prior is not None:
            raise ValueError("derived policy cannot silently ignore population_prior")
        prior = n_positive / (n * (1 - flip))
        if not 0 < prior <= 1:
            raise ValueError("derived training prior is outside (0,1]")
    else:
        if (
            isinstance(population_prior, bool)
            or not isinstance(population_prior, int | float)
            or not math.isfinite(population_prior)
            or not 0 < population_prior < 1
        ):
            raise ValueError("population_override requires population_prior in (0,1)")
        prior = float(population_prior)
    denominator = 1 - 2 * prior * flip
    lower, upper = -n_unlabeled / n, n_positive / n + n_unlabeled / (2 * n)
    rhs = None if abs(denominator) < 1e-12 else -n_unlabeled / (2 * n * denominator)
    feasible = rhs is not None and lower - 1e-12 <= rhs <= upper + 1e-12
    return {
        "prior_policy": policy,
        "effective_training_prior": prior,
        "flip_probability": flip,
        "centroid_denominator": denominator,
        "rhs": rhs,
        "reachable_min": lower,
        "reachable_max": upper,
        "absolute_tolerance": 1e-12,
        "feasible": feasible,
        "reason": "near_zero_centroid_denominator"
        if rhs is None
        else "equality_box_feasible"
        if feasible
        else "equality_outside_box",
        "flip_probability_source_verified": False,
        "population_prior_source_verified": False,
        "SCAR_generator_binding_verified": False,
        "feasible_is_not_full_fit_or_resource_approval": True,
    }


def build_plan(
    *,
    n_positive,
    n_unlabeled,
    feature_count,
    gram_itemsize,
    prediction_rows=0,
    max_acs_iter=None,
    max_inner_iter=None,
    memory_cap_bytes=None,
    root=ROOT,
    prior_policy=None,
    flip_probability=None,
    population_prior=None,
):
    """Account for named live QP arrays; omitted temporaries make this a lower bound.

    Gram dtype is caller-declared, not measured. The float64 QP blocks derive
    from the core's float64 y_tilde and Q allocation. A below-cap payload is NOT
    evidence that the fit is safe: previous QP matrices, solver/eigh copies,
    inputs, Python/BLAS state and kernel temporaries are deliberately excluded.
    """
    for name, value in (
        ("n_positive", n_positive),
        ("n_unlabeled", n_unlabeled),
        ("feature_count", feature_count),
    ):
        _positive_integer(value, name)
    if type(gram_itemsize) is not int or gram_itemsize not in (4, 8):
        raise ValueError("explicit Gram itemsize must be 4 or 8 bytes")
    if type(prediction_rows) is not int or prediction_rows < 0:
        raise ValueError("prediction_rows must be a nonnegative integer")
    if memory_cap_bytes is not None:
        _positive_integer(memory_cap_bytes, "memory_cap_bytes")
    feasibility = None
    if any(value is not None for value in (prior_policy, flip_probability, population_prior)):
        feasibility = _feasibility(
            n_positive, n_unlabeled, prior_policy, flip_probability, population_prior
        )
    root = Path(root)
    defaults = _defaults(root)
    outer = defaults["max_acs_iter"] if max_acs_iter is None else max_acs_iter
    inner = defaults["max_inner_iter"] if max_inner_iter is None else max_inner_iter
    _positive_integer(outer, "max_acs_iter")
    _positive_integer(inner, "max_inner_iter")
    n, u = n_positive + n_unlabeled, n_unlabeled
    dual = n + u
    blocks = {
        "yy": {"shape": [n, n], "bytes": 8 * n * n},
        "Q_alpha_alpha": {"shape": [n, n], "bytes": 8 * n * n},
        "Q_alpha_gamma": {"shape": [n, u], "bytes": 8 * n * u},
        "Q_gamma_gamma": {"shape": [u, u], "bytes": 8 * u * u},
        "Q": {"shape": [dual, dual], "bytes": 8 * dual * dual},
    }
    named_qp = sum(block["bytes"] for block in blocks.values())
    gram = gram_itemsize * n * n
    known = named_qp + gram
    references = (
        "scripts/prepare_kldce_resource_plan.py",
        SOURCE,
        "pu_toolbox/estimators/risk/kldce_smo.py",
        "pu_toolbox/utils/basis.py",
        "pu_toolbox/estimators/risk/_class_prior.py",
    )
    plan = {
        "schema_version": "kldce-dense-resource-plan-1",
        "method": "kldce",
        "execution_device": "cpu",
        "training_view": "os",
        "observed_rows": {"P": n_positive, "U": u},
        "feature_count": feature_count,
        "gram_itemsize_assumption": gram_itemsize,
        "gram_dtype_verified": False,
        "training_gram_shape": [n, n],
        "training_gram_payload_bytes": gram,
        "qp_float64_named_blocks": blocks,
        "qp_named_array_payload_bytes": named_qp,
        "known_simultaneous_gram_and_qp_payload_bytes": known,
        "covariance_itemsize_assumption": gram_itemsize,
        "covariance_single_matrix_payload_bytes": gram_itemsize * feature_count * feature_count,
        "prediction_rows": prediction_rows,
        "prediction_gram_shape": [prediction_rows, n],
        "prediction_gram_payload_bytes": gram_itemsize * prediction_rows * n,
        "max_acs_iter": outer,
        "max_inner_iter": inner,
        "qp_solves_upper_bound": 2 * outer - 1,
        "smo_pair_updates_upper_bound": (2 * outer - 1) * inner,
        "qp_bound_reason": (
            "one primary solve per ACS iteration plus at most one rollback; "
            "first iteration has no previous objective"
        ),
        "max_acs_iter_is_not_neural_epochs": True,
        "memory_cap_bytes": memory_cap_bytes,
        "known_payload_exceeds_cap": memory_cap_bytes is not None and known > memory_cap_bytes,
        "below_cap_is_not_fit_approval": True,
        "memory_scope": (
            "named live dense QP array payload lower bound under declared Gram dtype; "
            "not full peak RSS"
        ),
        "excluded_memory": [
            "input_and_P_U_copies",
            "covariance_matrices_and_decomposition",
            "previous_iteration_Q_while_building_next_Q",
            "RBF_temporaries",
            "solver_and_eigh_copies",
            "extra_covariance_ridge_copy",
            "vectors_and_centroid_kernels",
            "Python_and_BLAS_state",
        ],
        "budget_scope": "constructor_or_explicit_upper_bounds_not_frozen_recipe_or_runtime_ETA",
        "recipe_validation_performed": False,
        "resource_profile_completed": False,
        "data_loaded": False,
        "estimator_constructed": False,
        "cuda_initialized": False,
        "formal_admission": False,
        "ready_for_formal_run": False,
        "source_files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in references
        },
    }
    if feasibility is not None:
        plan["qp_feasibility_screen"] = feasibility
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-positive", type=int, required=True)
    parser.add_argument("--n-unlabeled", type=int, required=True)
    parser.add_argument("--feature-count", type=int, required=True)
    parser.add_argument("--gram-itemsize", type=int, choices=(4, 8), required=True)
    parser.add_argument("--prediction-rows", type=int, default=0)
    parser.add_argument("--max-acs-iter", type=int)
    parser.add_argument("--max-inner-iter", type=int)
    parser.add_argument("--memory-cap-bytes", type=int)
    parser.add_argument("--prior-policy", choices=("derived", "population_override"))
    parser.add_argument(
        "--flip-probability", type=float, help="caller-declared h, not a verified SCAR binding"
    )
    parser.add_argument("--population-prior", type=float)
    args = parser.parse_args(argv)
    try:
        plan = build_plan(**vars(args))
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        if plan.get("qp_feasibility_screen", {}).get("feasible") is False:
            return 3
        return 2 if plan["known_payload_exceeds_cap"] else 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
