# ruff: noqa: N803, N806

"""Cross-domain class-prior and positive-labeling mechanism analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pu_toolbox.core.base import BasePriorEstimator
from pu_toolbox.core.labels import normalize_pu_labels
from pu_toolbox.core.validation import check_scalar_in_range
from pu_toolbox.preprocessing.data_profiler import ProfileIssue
from pu_toolbox.utils.serialization import format_from_suffix, json_safe

MechanismConclusion = Literal[
    "stable", "class_prior_shift", "labeling_mechanism_shift", "both_shift", "inconclusive"
]

__all__ = ["DomainAssumptionReport", "analyze_domain_assumptions"]


@dataclass(frozen=True)
class DomainAssumptionReport:
    """Separable evidence about prevalence and positive-label propensity."""

    source: dict[str, Any]
    target: dict[str, Any]
    differences: dict[str, float | None]
    conclusion: MechanismConclusion
    sensitivity: tuple[dict[str, float | bool], ...]
    uncertainty: dict[str, Any] | None
    issues: tuple[ProfileIssue, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "schema_version": "1.0",
                "analysis_type": "cross_domain_pu_assumption_analysis",
                "source": self.source,
                "target": self.target,
                "differences": self.differences,
                "conclusion": self.conclusion,
                "sensitivity": list(self.sensitivity),
                "uncertainty": self.uncertainty,
                "issues": [item.to_dict() for item in self.issues],
                "provenance": self.provenance,
            }
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)

    def to_markdown(self) -> str:
        lines = [
            "# Cross-domain PU Assumption Report",
            "",
            f"- Conclusion: `{self.conclusion}`",
            f"- Source class prior: `{self.source['class_prior']:.6f}`",
            f"- Target class prior: `{self.target['class_prior']:.6f}`",
            f"- Source mean label propensity: `{self.source['mean_label_propensity']:.6f}`",
            f"- Target mean label propensity: `{self.target['mean_label_propensity']:.6f}`",
            "",
            "## Differences (target - source)",
            "",
            f"- Class prior: `{self.differences['class_prior']:.6f}`",
            f"- Observed label rate: `{self.differences['observed_label_rate']:.6f}`",
            f"- Mean label propensity: `{self.differences['mean_label_propensity']:.6f}`",
        ]
        if self.uncertainty is not None:
            lines.extend(
                [
                    "",
                    "## Bootstrap Uncertainty",
                    "",
                    f"- Valid replicates: `{self.uncertainty['n_valid']}` / "
                    f"`{self.uncertainty['n_requested']}`",
                    f"- Confidence level: `{self.uncertainty['confidence']:.3f}`",
                    f"- Class-prior difference CI: "
                    f"`[{self.uncertainty['differences']['class_prior']['low']:.6f}, "
                    f"{self.uncertainty['differences']['class_prior']['high']:.6f}]`",
                    f"- Propensity difference CI: "
                    f"`[{self.uncertainty['differences']['mean_label_propensity']['low']:.6f}, "
                    f"{self.uncertainty['differences']['mean_label_propensity']['high']:.6f}]`",
                ]
            )
        lines.extend(
            [
                "",
                "## Interpretation Boundary",
                "",
                "- The mean propensity follows `P(S=1) / P(Y=1)`; it does not prove SCAR.",
                "- Feature-dependent SAR is not identifiable from aggregate rates alone.",
                "- Percentile bootstrap captures sampling/estimator variation, not "
                "assumption bias.",
                "- Estimated class priors inherit the selected estimator's assumptions and bias.",
            ]
        )
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path, *, format: Literal["json", "markdown"] | None = None) -> Path:
        destination = Path(path)
        output_format = format or format_from_suffix(destination)
        if output_format == "csv":
            raise ValueError("DomainAssumptionReport supports JSON or Markdown, not CSV.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            self.to_json() + "\n" if output_format == "json" else self.to_markdown(),
            encoding="utf-8",
        )
        return destination


def _prior_estimator(name: str | BasePriorEstimator) -> BasePriorEstimator:
    if isinstance(name, BasePriorEstimator):
        from sklearn.base import clone

        return clone(name)
    if name in {"km1", "km2"}:
        from pu_toolbox.prior import KernelMeanPriorEstimator

        return KernelMeanPriorEstimator(variant=name)
    from pu_toolbox.core.exceptions import PULearningError
    from pu_toolbox.registry import get_algorithm
    from pu_toolbox.registry.builtin_methods import register_all_builtin_methods

    register_all_builtin_methods()
    try:
        cls = get_algorithm(name)
    except PULearningError as exc:
        raise ValueError(f"Unknown prior estimator {name!r}.") from exc
    if not isinstance(cls, type) or not issubclass(cls, BasePriorEstimator):
        raise ValueError(f"Algorithm {name!r} is not a prior estimator.")
    return cls()


def _resolve_prior(
    X: Any, y_pu: np.ndarray, value: float | None, estimator: str | BasePriorEstimator
) -> tuple[float, str]:
    if value is not None:
        check_scalar_in_range(value, 0.0, 1.0, "class_prior", inclusive=False)
        return float(value), "user"
    fitted = _prior_estimator(estimator).fit(X, y_pu)
    estimate = float(fitted.estimate())
    check_scalar_in_range(estimate, 0.0, 1.0, "estimated class_prior", inclusive=False)
    return estimate, "estimated"


def analyze_domain_assumptions(
    X_source: Any,
    y_source_pu: Any,
    X_target: Any,
    y_target_pu: Any,
    *,
    source_class_prior: float | None = None,
    target_class_prior: float | None = None,
    prior_estimator: str | BasePriorEstimator = "pen_l1",
    prior_shift_threshold: float = 0.05,
    propensity_shift_threshold: float = 0.05,
    sensitivity_radius: float = 0.05,
    bootstrap_replicates: int = 0,
    confidence: float = 0.95,
    random_state: int | None = 42,
) -> DomainAssumptionReport:
    """Contrast prevalence and aggregate positive-labeling propensity by domain."""
    for name, value in {
        "prior_shift_threshold": prior_shift_threshold,
        "propensity_shift_threshold": propensity_shift_threshold,
        "sensitivity_radius": sensitivity_radius,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number.")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 0
        or 0 < bootstrap_replicates < 2
    ):
        raise ValueError("bootstrap_replicates must be 0 or an integer >= 2.")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1).")
    source_labels = normalize_pu_labels(y_source_pu)
    target_labels = normalize_pu_labels(y_target_pu)
    if len(source_labels) != len(X_source) or len(target_labels) != len(X_target):
        raise ValueError("Each PU label vector must match its domain feature rows.")
    source_prior, source_origin = _resolve_prior(
        X_source, source_labels, source_class_prior, prior_estimator
    )
    target_prior, target_origin = _resolve_prior(
        X_target, target_labels, target_class_prior, prior_estimator
    )
    source_rate = float(np.mean(source_labels == 1))
    target_rate = float(np.mean(target_labels == 1))
    source_propensity = source_rate / source_prior
    target_propensity = target_rate / target_prior
    prior_delta = target_prior - source_prior
    rate_delta = target_rate - source_rate
    propensity_delta = target_propensity - source_propensity
    issues: list[ProfileIssue] = []
    consistent = source_propensity <= 1.0 and target_propensity <= 1.0
    if not consistent:
        issues.append(
            ProfileIssue(
                code="prior_below_observed_label_rate",
                severity="error",
                message="At least one class prior is below its observed labeled-positive rate.",
                action="Use a feasible prior or re-check the prior estimator and labels.",
            )
        )
        conclusion: MechanismConclusion = "inconclusive"
    else:
        prior_changed = abs(prior_delta) >= prior_shift_threshold
        propensity_changed = abs(propensity_delta) >= propensity_shift_threshold
        conclusion = (
            "both_shift"
            if prior_changed and propensity_changed
            else "class_prior_shift"
            if prior_changed
            else "labeling_mechanism_shift"
            if propensity_changed
            else "stable"
        )
    sensitivity: list[dict[str, float | bool]] = []
    for source_candidate in _candidate_priors(source_prior, sensitivity_radius):
        for target_candidate in _candidate_priors(target_prior, sensitivity_radius):
            source_c = source_rate / source_candidate
            target_c = target_rate / target_candidate
            sensitivity.append(
                {
                    "source_class_prior": source_candidate,
                    "target_class_prior": target_candidate,
                    "prior_difference": target_candidate - source_candidate,
                    "propensity_difference": target_c - source_c,
                    "is_feasible": source_c <= 1.0 and target_c <= 1.0,
                }
            )
    uncertainty = None
    if bootstrap_replicates:
        uncertainty = _bootstrap_domain_uncertainty(
            X_source,
            source_labels,
            X_target,
            target_labels,
            source_class_prior=source_class_prior,
            target_class_prior=target_class_prior,
            prior_estimator=prior_estimator,
            n_replicates=bootstrap_replicates,
            confidence=confidence,
            random_state=random_state,
        )
        if uncertainty["n_valid"] / uncertainty["n_requested"] < 0.8:
            issues.append(
                ProfileIssue(
                    code="bootstrap_instability",
                    severity="warning",
                    message="More than 20% of bootstrap prior fits were infeasible or failed.",
                    action="Increase domain sample sizes or use a more stable prior estimator.",
                )
            )
    return DomainAssumptionReport(
        source={
            "n_samples": len(source_labels),
            "observed_label_rate": source_rate,
            "class_prior": source_prior,
            "class_prior_source": source_origin,
            "mean_label_propensity": source_propensity,
        },
        target={
            "n_samples": len(target_labels),
            "observed_label_rate": target_rate,
            "class_prior": target_prior,
            "class_prior_source": target_origin,
            "mean_label_propensity": target_propensity,
        },
        differences={
            "class_prior": prior_delta,
            "observed_label_rate": rate_delta,
            "mean_label_propensity": propensity_delta,
        },
        conclusion=conclusion,
        sensitivity=tuple(sensitivity),
        uncertainty=uncertainty,
        issues=tuple(issues),
        provenance={
            "identity": "P(S=1)=P(Y=1)*E[P(S=1|Y=1,X)|Y=1]",
            "prior_estimator": (
                prior_estimator
                if isinstance(prior_estimator, str)
                else type(prior_estimator).__name__
            ),
            "aggregate_propensity_does_not_identify_scar": True,
            "bootstrap_random_state": random_state if bootstrap_replicates else None,
        },
    )


def _candidate_priors(center: float, radius: float) -> tuple[float, ...]:
    return tuple(sorted({max(1e-6, center - radius), center, min(1.0 - 1e-6, center + radius)}))


def _take_rows(X: Any, indices: np.ndarray) -> Any:
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    try:
        return X[indices]
    except (TypeError, IndexError):
        return np.asarray(X)[indices]


def _interval(values: list[float], confidence: float) -> dict[str, float]:
    lower = (1.0 - confidence) / 2.0
    array = np.asarray(values, dtype=float)
    return {
        "estimate": float(np.mean(array)),
        "low": float(np.quantile(array, lower)),
        "high": float(np.quantile(array, 1.0 - lower)),
    }


def _bootstrap_domain_uncertainty(
    X_source: Any,
    source_labels: np.ndarray,
    X_target: Any,
    target_labels: np.ndarray,
    *,
    source_class_prior: float | None,
    target_class_prior: float | None,
    prior_estimator: str | BasePriorEstimator,
    n_replicates: int,
    confidence: float,
    random_state: int | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_state)
    values: dict[str, list[float]] = {
        "source_prior": [],
        "target_prior": [],
        "source_rate": [],
        "target_rate": [],
        "source_propensity": [],
        "target_propensity": [],
        "prior_difference": [],
        "rate_difference": [],
        "propensity_difference": [],
    }
    failed = 0
    for _ in range(n_replicates):
        source_indices = rng.integers(0, len(source_labels), size=len(source_labels))
        target_indices = rng.integers(0, len(target_labels), size=len(target_labels))
        source_boot_labels = source_labels[source_indices]
        target_boot_labels = target_labels[target_indices]
        try:
            source_prior, _ = _resolve_prior(
                _take_rows(X_source, source_indices),
                source_boot_labels,
                source_class_prior,
                prior_estimator,
            )
            target_prior, _ = _resolve_prior(
                _take_rows(X_target, target_indices),
                target_boot_labels,
                target_class_prior,
                prior_estimator,
            )
            source_rate = float(np.mean(source_boot_labels == 1))
            target_rate = float(np.mean(target_boot_labels == 1))
            source_propensity = source_rate / source_prior
            target_propensity = target_rate / target_prior
            if source_propensity > 1.0 or target_propensity > 1.0:
                raise ValueError("bootstrap prior is below the observed label rate")
        except Exception:  # noqa: BLE001 - failures are counted in the uncertainty report
            failed += 1
            continue
        values["source_prior"].append(source_prior)
        values["target_prior"].append(target_prior)
        values["source_rate"].append(source_rate)
        values["target_rate"].append(target_rate)
        values["source_propensity"].append(source_propensity)
        values["target_propensity"].append(target_propensity)
        values["prior_difference"].append(target_prior - source_prior)
        values["rate_difference"].append(target_rate - source_rate)
        values["propensity_difference"].append(target_propensity - source_propensity)
    n_valid = len(values["source_prior"])
    if n_valid < 2:
        raise ValueError(
            "Fewer than two valid bootstrap replicates remained after prior estimation."
        )
    return {
        "method": "independent_nonparametric_percentile_bootstrap",
        "confidence": float(confidence),
        "n_requested": n_replicates,
        "n_valid": n_valid,
        "n_failed": failed,
        "source": {
            "class_prior": _interval(values["source_prior"], confidence),
            "observed_label_rate": _interval(values["source_rate"], confidence),
            "mean_label_propensity": _interval(values["source_propensity"], confidence),
        },
        "target": {
            "class_prior": _interval(values["target_prior"], confidence),
            "observed_label_rate": _interval(values["target_rate"], confidence),
            "mean_label_propensity": _interval(values["target_propensity"], confidence),
        },
        "differences": {
            "class_prior": _interval(values["prior_difference"], confidence),
            "observed_label_rate": _interval(values["rate_difference"], confidence),
            "mean_label_propensity": _interval(values["propensity_difference"], confidence),
        },
    }
