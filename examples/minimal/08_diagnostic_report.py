# ruff: noqa: N806

"""Build a diagnostic report for a fitted PUSB classifier."""
# API reference: docs/user/reference/api/tools.md#build_diagnostic_report

from sklearn.model_selection import train_test_split

from pu_toolbox.core.random import check_random_state
from pu_toolbox.diagnostics import build_diagnostic_report
from pu_toolbox.estimators.bias_aware import PUSBClassifier
from pu_toolbox.preprocessing import GaussianSourceConfig, PULabelingConfig, simulate_pu_dataset


def main() -> None:
    rng = check_random_state(42)
    result = simulate_pu_dataset(
        GaussianSourceConfig(1200, 6, 0.3, 2.0, rng),
        PULabelingConfig(
            "sar",
            "bernoulli",
            rng,
            c=0.4,
            propensity_rule="linear",
            strength=1.5,
            ensure_labeled=True,
        ),
    )
    X, y_pu = result.observations.features, result.observations.labels
    y_true = result.audit.true_labels
    X_train, X_test, y_train, y_test, _, truth_test = train_test_split(
        X,
        y_pu,
        y_true,
        test_size=0.3,
        stratify=y_pu,
        random_state=42,
    )

    # PUSB is an observed-P-vs-U logistic baseline, not a latent-class posterior model.
    # Its class_prior is metadata; SAR can invalidate SCAR-based PU metric interpretations.
    classifier = PUSBClassifier().fit(X_train, y_train)
    report = build_diagnostic_report(
        X_test,
        y_test,
        estimator=classifier,
        y_true=truth_test,
        evaluation_prior=0.3,
        random_state=42,
    )

    print(report.to_markdown())


if __name__ == "__main__":
    main()
