# ruff: noqa: N806

"""Monitor shift and create a human-review queue for a directly fitted model."""
# API reference: docs/user/reference/api/tools.md#distribution-shift-api

from pu_toolbox.diagnostics import (
    PUShiftMonitor,
    analyze_domain_assumptions,
    analyze_pu_uncertainty,
)
from pu_toolbox.estimators.classic import ElkanNotoClassifier
from pu_toolbox.preprocessing import make_scar_dataset


def main() -> None:
    X_source, y_source, y_true_source = make_scar_dataset(
        n=160, c=0.5, n_features=4, separation=1.5, random_state=10
    )
    X_target, y_target, y_true_target = make_scar_dataset(
        n=100, c=0.5, n_features=4, separation=1.5, random_state=20
    )
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
