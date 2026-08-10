# ruff: noqa: N803, N806

"""Structured, actionable profiling for positive-unlabeled datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse
from sklearn.utils.sparsefuncs import mean_variance_axis

from pu_toolbox.core.labels import normalize_pu_labels
from pu_toolbox.core.validation import check_scalar_in_range, validate_true_binary_labels
from pu_toolbox.preprocessing.profiling import pu_data_summary, scar_diagnostic
from pu_toolbox.utils.serialization import json_safe

IssueSeverity = Literal["info", "warning", "error"]

__all__ = [
    "IssueSeverity",
    "PUDataProfile",
    "ProfileIssue",
    "profile_pu_data",
]


@dataclass(frozen=True)
class ProfileIssue:
    """One data or assumption issue with a concrete next action."""

    code: str
    severity: IssueSeverity
    message: str
    action: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "action": self.action,
        }


@dataclass(frozen=True)
class PUDataProfile:
    """Structured report returned by :func:`profile_pu_data`."""

    summary: dict[str, Any]
    feature_statistics: dict[str, Any]
    selection_diagnostic: dict[str, Any]
    issues: tuple[ProfileIssue, ...]
    assumption_hints: tuple[str, ...]

    @property
    def has_errors(self) -> bool:
        """Whether any issue prevents reliable downstream fitting."""
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Whether any issue merits user review before model fitting."""
        return any(issue.severity == "warning" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-serializable nested dictionary.

        Undefined diagnostics and infinite ratios become ``None`` instead of
        non-standard JSON ``NaN`` or ``Infinity`` tokens.
        """
        payload = {
            "summary": dict(self.summary),
            "feature_statistics": dict(self.feature_statistics),
            "selection_diagnostic": dict(self.selection_diagnostic),
            "issues": [issue.to_dict() for issue in self.issues],
            "assumption_hints": list(self.assumption_hints),
        }
        return json_safe(payload)

    def format_text(self) -> str:
        """Render a compact report suitable for terminals and logs."""
        summary = self.summary
        selection = self.selection_diagnostic
        auc = selection["separability_auc"]
        auc_text = "not available" if not np.isfinite(auc) else f"{auc:.3f}"
        lines = [
            "PU data profile",
            (
                f"Samples: {summary['n_samples']} | Features: "
                f"{summary['n_features']} | Labeled positives: "
                f"{summary['n_positives']} | Unlabeled: {summary['n_unlabeled']}"
            ),
            (
                f"Selection diagnostic: {selection['status']} "
                f"(AUC={auc_text}, evidence={selection['evidence']}, "
                f"identifying={selection['is_identifying']})"
            ),
        ]
        if self.issues:
            lines.append("Issues:")
            lines.extend(
                f"- [{issue.severity.upper()}] {issue.message} Action: {issue.action}"
                for issue in self.issues
            )
        else:
            lines.append("Issues: none detected by configured checks.")
        lines.append("Assumption notes:")
        lines.extend(f"- {hint}" for hint in self.assumption_hints)
        return "\n".join(lines)


def _validate_thresholds(
    min_labeled_positives: int,
    max_unlabeled_to_positive: float,
    low_variance_threshold: float,
) -> None:
    if (
        isinstance(min_labeled_positives, bool)
        or not isinstance(min_labeled_positives, int | np.integer)
        or min_labeled_positives < 1
    ):
        raise ValueError("min_labeled_positives must be an integer >= 1.")
    if not np.isfinite(max_unlabeled_to_positive) or max_unlabeled_to_positive <= 0:
        raise ValueError("max_unlabeled_to_positive must be finite and > 0.")
    if not np.isfinite(low_variance_threshold) or low_variance_threshold < 0:
        raise ValueError("low_variance_threshold must be finite and >= 0.")


def _validate_x(X: Any) -> np.ndarray | sparse.csr_matrix:
    if sparse.issparse(X):
        matrix = sparse.csr_matrix(X, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"X must be 2-D; got ndim={matrix.ndim}.")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature.")
        return matrix
    try:
        array = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain numeric feature values.") from exc
    if array.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={array.ndim}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must contain at least one sample and one feature.")
    return array


def _validate_class_prior(class_prior: float | None) -> float | None:
    if class_prior is None:
        return None
    if isinstance(class_prior, bool) or not np.isscalar(class_prior):
        raise TypeError("class_prior must be a real scalar or None.")
    value = float(class_prior)
    check_scalar_in_range(value, 0.0, 1.0, "class_prior", inclusive=False)
    return value


def _validate_audit_labels(
    y_true: np.ndarray | None,
    y_pu: np.ndarray,
) -> np.ndarray | None:
    if y_true is None:
        return None
    true = np.asarray(y_true)
    if true.ndim != 1:
        raise ValueError(f"y_true must be 1-D; got ndim={true.ndim}.")
    if len(true) != len(y_pu):
        raise ValueError(f"X has {len(y_pu)} samples but y_true has {len(true)}.")
    validate_true_binary_labels(true, estimator_name="y_true")
    if np.any((y_pu == 1) & (true != 1)):
        raise ValueError("Every labeled positive in y_pu must be positive in y_true.")
    return true.astype(int, copy=False)


def _feature_statistics(
    X: np.ndarray | sparse.csr_matrix,
    low_variance_threshold: float,
) -> dict[str, Any]:
    values = X.data if sparse.issparse(X) else X
    missing_count = int(np.isnan(values).sum())
    infinite_count = int(np.isinf(values).sum())
    finite = missing_count == 0 and infinite_count == 0

    constant_features: list[int] = []
    low_variance_features: list[int] = []
    if finite:
        if sparse.issparse(X):
            _, variances = mean_variance_axis(X, axis=0)
        else:
            variances = np.var(X, axis=0)
        constant_features = np.flatnonzero(variances == 0.0).astype(int).tolist()
        low_variance_features = (
            np.flatnonzero((variances > 0.0) & (variances <= low_variance_threshold))
            .astype(int)
            .tolist()
        )
    return {
        "missing_value_count": missing_count,
        "infinite_value_count": infinite_count,
        "constant_feature_indices": constant_features,
        "low_variance_feature_indices": low_variance_features,
        "low_variance_threshold": float(low_variance_threshold),
    }


def _inconclusive_selection(message: str, threshold: float) -> dict[str, Any]:
    return {
        "separability_auc": float("nan"),
        "is_observed_dependence_absent": None,
        "status": "inconclusive",
        "message": message,
        "threshold": float(threshold),
        "evidence": "not_evaluated",
        "is_identifying": False,
        "n_samples_evaluated": 0,
        "n_splits": 0,
    }


def profile_pu_data(
    X: Any,
    y_pu: np.ndarray,
    *,
    y_true: np.ndarray | None = None,
    class_prior: float | None = None,
    min_labeled_positives: int = 30,
    max_unlabeled_to_positive: float = 100.0,
    low_variance_threshold: float = 1e-12,
    scar_auc_threshold: float = 0.65,
    cv: int = 5,
    random_state: int | None = 42,
) -> PUDataProfile:
    """Build a structured quality and SCAR/SAR assumption profile.

    Parameters
    ----------
    X : array-like or sparse matrix of shape (n_samples, n_features)
        Numeric feature matrix. Missing and infinite values are reported as
        errors rather than silently imputed.
    y_pu : array-like of shape (n_samples,)
        PU labels accepted by :func:`normalize_pu_labels`.
    y_true : array-like of shape (n_samples,), optional
        Audited binary truth. When provided, the SCAR diagnostic conditions on
        true positives and becomes identifying for feature-dependent selection.
    class_prior : float, optional
        Assumed or independently estimated ``P(Y=1)``. It is used only for a
        labeling-frequency consistency check; it is never estimated from
        hidden truth.
    min_labeled_positives : int, default=30
        Warning threshold for labeled-positive sample size.
    max_unlabeled_to_positive : float, default=100
        Warning threshold for the unlabeled-to-labeled-positive ratio.
    low_variance_threshold : float, default=1e-12
        Features with variance in ``(0, threshold]`` are flagged.
    scar_auc_threshold : float, default=0.65
        Cross-validated selection AUC threshold used by
        :func:`scar_diagnostic`.
    cv : int, default=5
        Maximum stratified folds for the selection diagnostic.
    random_state : int or None, default=42
        Reproducibility control for the diagnostic.

    Returns
    -------
    PUDataProfile
        Structured summary, feature statistics, selection evidence, issues,
        and assumption notes. Use ``format_text()`` for a readable report or
        ``to_dict()`` for JSON serialization.

    Notes
    -----
    Without audited ``y_true``, SCAR versus SAR is not identifiable from
    observed ``(X, S)`` alone. The returned observed-mixture AUC is therefore
    a screening signal, not a statistical proof of SAR.
    """
    _validate_thresholds(
        min_labeled_positives,
        max_unlabeled_to_positive,
        low_variance_threshold,
    )
    X_array = _validate_x(X)
    y = normalize_pu_labels(np.asarray(y_pu))
    if y.ndim != 1:
        raise ValueError(f"y_pu must be 1-D; got ndim={y.ndim}.")
    if X_array.shape[0] != len(y):
        raise ValueError(f"X has {X_array.shape[0]} samples but y_pu has {len(y)}.")
    prior = _validate_class_prior(class_prior)
    audited_truth = _validate_audit_labels(y_true, y)

    summary = pu_data_summary(X_array, y)
    summary["class_prior"] = prior
    summary["implied_label_frequency"] = (
        summary["positive_fraction"] / prior if prior is not None else None
    )
    feature_statistics = _feature_statistics(X_array, low_variance_threshold)
    issues: list[ProfileIssue] = []

    if summary["n_positives"] == 0:
        issues.append(
            ProfileIssue(
                "no_labeled_positives",
                "error",
                "The dataset contains no labeled positives.",
                "Verify label encoding or collect at least one trusted positive.",
            )
        )
    elif summary["n_positives"] < min_labeled_positives:
        issues.append(
            ProfileIssue(
                "few_labeled_positives",
                "warning",
                f"Only {summary['n_positives']} labeled positives are available.",
                "Use repeated validation and report uncertainty; collect more "
                "positives when possible.",
            )
        )
    if summary["n_unlabeled"] == 0:
        issues.append(
            ProfileIssue(
                "no_unlabeled_samples",
                "error",
                "The dataset contains no unlabeled samples.",
                "Use a supervised workflow or provide the unlabeled population.",
            )
        )
    elif summary["pu_ratio"] > max_unlabeled_to_positive:
        issues.append(
            ProfileIssue(
                "extreme_pu_imbalance",
                "warning",
                f"The unlabeled-to-positive ratio is {summary['pu_ratio']:.1f}:1.",
                "Use PU-aware stratification and metrics robust to imbalance.",
            )
        )

    missing_count = feature_statistics["missing_value_count"]
    infinite_count = feature_statistics["infinite_value_count"]
    if missing_count:
        issues.append(
            ProfileIssue(
                "missing_features",
                "error",
                f"X contains {missing_count} missing feature values.",
                "Impute missing values inside a leakage-safe training pipeline.",
            )
        )
    if infinite_count:
        issues.append(
            ProfileIssue(
                "infinite_features",
                "error",
                f"X contains {infinite_count} infinite feature values.",
                "Replace or remove infinite values before fitting.",
            )
        )
    constant = feature_statistics["constant_feature_indices"]
    low_variance = feature_statistics["low_variance_feature_indices"]
    if constant:
        issues.append(
            ProfileIssue(
                "constant_features",
                "warning",
                f"Detected {len(constant)} constant features.",
                "Remove constant columns within the training pipeline.",
            )
        )
    if low_variance:
        issues.append(
            ProfileIssue(
                "low_variance_features",
                "info",
                f"Detected {len(low_variance)} low-variance features.",
                "Review feature scaling and whether these columns carry useful signal.",
            )
        )
    if summary["n_features"] >= summary["n_samples"]:
        issues.append(
            ProfileIssue(
                "high_dimensional_data",
                "warning",
                "The number of features is at least the number of samples.",
                "Prefer regularized estimators and fit preprocessing inside each validation fold.",
            )
        )
    if prior is not None and summary["implied_label_frequency"] > 1.0:
        issues.append(
            ProfileIssue(
                "inconsistent_class_prior",
                "warning",
                "The labeled-positive fraction exceeds the supplied class prior.",
                "Recheck the prior estimate, sampling frame, and label encoding.",
            )
        )

    has_nonfinite = bool(missing_count or infinite_count)
    if has_nonfinite:
        selection = _inconclusive_selection(
            "Selection diagnostic was skipped because X contains non-finite values.",
            scar_auc_threshold,
        )
    else:
        selection = scar_diagnostic(
            X_array,
            y,
            y_true=audited_truth,
            cv=cv,
            threshold=scar_auc_threshold,
            random_state=random_state,
        )

    if selection["status"] == "at_risk":
        if selection["is_identifying"]:
            issues.append(
                ProfileIssue(
                    "sar_signal",
                    "warning",
                    "Audited positives show feature-dependent labeling inconsistent with SCAR.",
                    "Use a bias-aware method such as PUSB/LBE and run sensitivity analysis.",
                )
            )
        else:
            issues.append(
                ProfileIssue(
                    "observed_selection_signal",
                    "info",
                    "Labeled and unlabeled samples are feature-separable, but "
                    "the cause is not identifiable.",
                    "Audit true positives or collect labeling-policy information "
                    "before asserting SAR.",
                )
            )
    elif selection["status"] == "inconclusive":
        issues.append(
            ProfileIssue(
                "selection_diagnostic_inconclusive",
                "info",
                selection["message"],
                "Increase both selection groups or provide audited true-positive labels.",
            )
        )

    hints = [
        "SCAR and SAR are not identifiable from observed PU labels alone "
        "without additional assumptions or audit data."
    ]
    if selection["is_identifying"]:
        if selection["status"] == "plausible":
            hints.append(
                "The audited check found no strong feature dependence; this "
                "supports but does not prove SCAR."
            )
        elif selection["status"] == "at_risk":
            hints.append(
                "The audited check detected selection dependence; SCAR-only "
                "estimators may be biased."
            )
    else:
        hints.append(
            "The observed-mixture AUC may reflect class separation rather than selection bias."
        )
    if prior is None:
        hints.append(
            "Methods that require a class prior need an independently justified "
            "estimate and sensitivity analysis."
        )
    else:
        hints.append(
            "The supplied class prior implies a labeling frequency of "
            f"{summary['implied_label_frequency']:.3f}."
        )

    return PUDataProfile(
        summary=summary,
        feature_statistics=feature_statistics,
        selection_diagnostic=selection,
        issues=tuple(issues),
        assumption_hints=tuple(hints),
    )
