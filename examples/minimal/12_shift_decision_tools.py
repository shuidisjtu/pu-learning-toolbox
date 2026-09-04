# ruff: noqa: N806

"""Compare adaptation, monitor windows, and create a human-review queue."""
# API 参考（签名/参数/返回契约）：docs/user/reference/api.md

from pu_toolbox.diagnostics import (
    PUShiftMonitor,
    analyze_domain_assumptions,
    analyze_pu_uncertainty,
)
from pu_toolbox.preprocessing import make_scar_dataset
from pu_toolbox.workflows import ShiftAwarePUPipeline


def main() -> None:
    X_source, y_source, y_true_source = make_scar_dataset(
        n=160, c=0.5, n_features=4, separation=1.5, random_state=10
    )
    X_target, y_target, y_true_target = make_scar_dataset(
        n=100, c=0.5, n_features=4, separation=1.5, random_state=20
    )
    X_target += 0.2

    workflow = ShiftAwarePUPipeline(classifier="elkan_noto", cv=3, shift_cv=3)
    comparison = workflow.compare(
        X_source,
        y_source,
        X_target,
        y_target_pu=y_target,
        y_true_source=y_true_source,
        y_true_target=y_true_target,
        class_prior=0.5,
        target_class_prior=0.5,
    )
    print("adaptation decision:", comparison.recommendation)

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

    chosen = comparison.weighted or comparison.baseline
    review = analyze_pu_uncertainty(
        chosen.final_model,
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
