#!/usr/bin/env python3
# ruff: noqa: N802, N803, N806, E402, E501
"""pu-workflow step 2: recommend PU methods and estimate the class prior.

Reads the ``profile.json`` written by the profile step (Task 2) and writes
``recommendation.json``.  Exit codes: 0 success; 1 user/input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_toolbox.advisor.recommender import recommend_from_profile
from pu_toolbox.cli.run import _load_features, _load_label_column
from pu_toolbox.preprocessing.data_profiler import ProfileIssue, PUDataProfile


def _profile_from_dict(payload: dict) -> PUDataProfile:
    """Rebuild a PUDataProfile from the strict-JSON profile file."""
    return PUDataProfile(
        summary=payload["summary"],
        feature_statistics=payload["feature_statistics"],
        selection_diagnostic=payload["selection_diagnostic"],
        issues=tuple(
            ProfileIssue(
                code=i["code"], severity=i["severity"], message=i["message"], action=i["action"]
            )
            for i in payload["issues"]
        ),
        assumption_hints=tuple(payload["assumption_hints"]),
    )


def _estimate_prior(name: str, X, y_pu) -> float:
    """Estimate the class prior with a registered estimator or km1/km2."""
    if name in {"km1", "km2"}:
        from pu_toolbox.prior.kernel_mean import KernelMeanPriorEstimator

        est = KernelMeanPriorEstimator(variant=name)
    else:
        from pu_toolbox.core.base import BasePriorEstimator
        from pu_toolbox.core.exceptions import PULearningError
        from pu_toolbox.registry import get_algorithm
        from pu_toolbox.registry.builtin_methods import register_all_builtin_methods

        register_all_builtin_methods()
        try:
            est_cls = get_algorithm(name)
        except PULearningError as exc:
            raise ValueError(
                f"Unknown prior estimator '{name}'. Use 'recpe', 'pen_l1', 'km1', 'km2', "
                "or a registered prior estimator."
            ) from exc
        if not issubclass(est_cls, BasePriorEstimator):
            raise ValueError(
                f"Algorithm '{name}' is not a prior estimator. "
                "Use 'recpe', 'pen_l1', 'km1', 'km2', or a registered prior estimator."
            )
        est = est_cls()
    est.fit(X, y_pu)
    return est.estimate()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pu-workflow-recommend")
    parser.add_argument("--profile", required=True, help="profile.json from the profile step")
    parser.add_argument(
        "--data", default=None, help="feature matrix CSV (required for prior estimation)"
    )
    parser.add_argument(
        "--labels", default=None, help="PU labels CSV (required for prior estimation)"
    )
    parser.add_argument(
        "--class-prior", type=float, default=None, help="explicit class prior (0, 1)"
    )
    parser.add_argument(
        "--prior-estimator",
        default="none",
        help="prior estimator: none | recpe | pen_l1 | km1 | km2 (default: none)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="candidates to return (default: 5)")
    parser.add_argument("--has-gpu", action="store_true", help="declare a GPU device")
    parser.add_argument("--out-dir", default=".", help="output directory (default: current)")
    args = parser.parse_args(argv)

    try:
        payload = json.loads(Path(args.profile).read_text(encoding="utf-8"))
        profile = _profile_from_dict(payload)
        class_prior = args.class_prior
        class_prior_source: str | None = None
        if args.class_prior is not None:
            class_prior_source = "user"
        if args.prior_estimator != "none":
            if args.data is None or args.labels is None:
                raise ValueError(
                    "--prior-estimator requires --data and --labels "
                    "(estimation runs on the raw data)"
                )
            X = _load_features(Path(args.data))
            y_pu = _load_label_column(Path(args.labels), "labels")
            class_prior = _estimate_prior(args.prior_estimator, X, y_pu)
            class_prior_source = "estimated"
        result = recommend_from_profile(
            profile,
            class_prior=class_prior,
            class_prior_source=class_prior_source,
            has_gpu=args.has_gpu,
            top_k=args.top_k,
        )
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "recommendation.json").write_text(result.to_json() + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
