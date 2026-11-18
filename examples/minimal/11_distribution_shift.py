# ruff: noqa: N806

"""Audit PU distribution shift without training or adaptation."""
# API reference: docs/user/reference/api/tools.md#distribution-shift-api

from pu_toolbox.diagnostics import analyze_pu_shift
from pu_toolbox.preprocessing import make_scar_dataset


def main() -> None:
    X_source, y_source, y_true_source = make_scar_dataset(
        n=250,
        c=0.5,
        n_features=5,
        separation=2.0,
        random_state=10,
    )
    X_target, y_target, y_true_target = make_scar_dataset(
        n=120,
        c=0.5,
        n_features=5,
        separation=2.0,
        random_state=20,
    )
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
