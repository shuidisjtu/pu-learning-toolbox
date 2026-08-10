# ruff: noqa: N802, N803, N806, E501

"""The ``recommend`` subcommand: recommend PU methods and estimate the prior.

pu-workflow step 2: reads the ``profile.json`` written by the profile step
and writes ``recommendation.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..advisor.recommender import recommend_from_profile
from ..preprocessing.data_profiler import ProfileIssue, PUDataProfile
from .run import _load_features, _load_label_column

__all__ = ["build_recommend_parser", "run_recommend"]


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
        from ..prior.kernel_mean import KernelMeanPriorEstimator

        est = KernelMeanPriorEstimator(variant=name)
    else:
        from ..core.base import BasePriorEstimator
        from ..core.exceptions import PULearningError
        from ..registry import get_algorithm
        from ..registry.builtin_methods import register_all_builtin_methods

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


def build_recommend_parser(sub: argparse._SubParsersAction) -> None:
    """Attach the ``recommend`` subcommand to *sub* (side-effect only)."""
    parser = sub.add_parser(
        "recommend",
        help="recommend PU methods and estimate the class prior (pu-workflow step 2)",
    )
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
    parser.set_defaults(func=run_recommend)


def run_recommend(args: argparse.Namespace) -> None:
    """Write ``recommendation.json`` into ``--out-dir``; raise on user errors."""
    payload = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    profile = _profile_from_dict(payload)
    class_prior = args.class_prior
    class_prior_source: str | None = None
    if args.class_prior is not None:
        class_prior_source = "user"
    if args.prior_estimator != "none":
        if args.data is None or args.labels is None:
            raise ValueError(
                "--prior-estimator requires --data and --labels (estimation runs on the raw data)"
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
