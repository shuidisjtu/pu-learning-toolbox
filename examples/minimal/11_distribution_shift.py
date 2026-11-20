# ruff: noqa: N806

"""Audit PU distribution shift without training or adaptation."""
# API reference: docs/user/reference/api/tools.md#distribution-shift-api

from pu_toolbox.core.random import check_random_state
from pu_toolbox.diagnostics import analyze_pu_shift
from pu_toolbox.preprocessing import GaussianSourceConfig, PULabelingConfig, simulate_pu_dataset


def main() -> None:
    rng = check_random_state(10)
    result = simulate_pu_dataset(
        GaussianSourceConfig(500, 5, 0.5, 2.0, rng, row_order="class_blocked"),
        PULabelingConfig("scar", "fixed_count", rng, c=0.5),
    )
    X_source, y_source = result.observations.features, result.observations.labels
    rng = check_random_state(20)
    result = simulate_pu_dataset(
        GaussianSourceConfig(240, 5, 0.5, 2.0, rng, row_order="class_blocked"),
        PULabelingConfig("scar", "fixed_count", rng, c=0.5),
    )
    X_target, y_target = result.observations.features, result.observations.labels
    # A location shift exercises the workflow but need not preserve P(Y|X).
    # Any adaptation guarantee still depends on the covariate-shift assumptions.
    X_target = X_target + 0.25

    report = analyze_pu_shift(
        X_source, y_source, X_target, y_target_pu=y_target, cv=3, random_state=42
    )
    print(report.domain_auc, report.severity)
    print(report.weight_summary["effective_sample_fraction"])


if __name__ == "__main__":
    main()
