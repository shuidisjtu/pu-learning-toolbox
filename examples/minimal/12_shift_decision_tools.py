# ruff: noqa: N806

"""Monitor shift and create a human-review queue for a directly fitted model."""
# API reference: docs/user/reference/api/tools.md#distribution-shift-api

from pu_toolbox.core.random import check_random_state
from pu_toolbox.diagnostics import (
    PUShiftMonitor,
    analyze_domain_assumptions,
    analyze_pu_uncertainty,
)
from pu_toolbox.estimators.classic import ElkanNotoClassifier
from pu_toolbox.preprocessing import GaussianSourceConfig, PULabelingConfig, simulate_pu_dataset


def main() -> None:
    rng = check_random_state(10)
    result = simulate_pu_dataset(
        GaussianSourceConfig(320, 4, 0.5, 1.5, rng, row_order="class_blocked"),
        PULabelingConfig("scar", "fixed_count", rng, c=0.5),
    )
    X_source, y_source = result.observations.features, result.observations.labels
    rng = check_random_state(20)
    result = simulate_pu_dataset(
        GaussianSourceConfig(200, 4, 0.5, 1.5, rng, row_order="class_blocked"),
        PULabelingConfig("scar", "fixed_count", rng, c=0.5),
    )
    X_target, y_target = result.observations.features, result.observations.labels
    y_true_target = result.audit.true_labels
    # A location shift exercises the workflow but need not preserve P(Y|X).
    # This is not evidence that covariate-shift assumptions hold.
    X_target += 0.2

    model = ElkanNotoClassifier().fit(X_source, y_source)

    monitor = PUShiftMonitor(X_source, y_source, cv=3)
    window, _ = monitor.update(X_target, y_window_pu=y_target, window_id="current")
    print("monitor alert:", window.alert_level, window.alert_codes)

    assumptions = analyze_domain_assumptions(
        X_source,
        y_source,
        X_target,
        y_target,
        source_class_prior=0.5,
        target_class_prior=0.5,
    )
    print("assumption difference:", assumptions.conclusion)

    review = analyze_pu_uncertainty(
        model,
        X_target,
        y_pu=y_target,
        y_true=y_true_target,
        min_confidence=0.5,
        query_budget=10,
        query_strategy="diverse_uncertainty",
    )
    print("coverage:", review.summary["coverage"])
    print("review rows:", review.query_indices.tolist())


if __name__ == "__main__":
    main()
