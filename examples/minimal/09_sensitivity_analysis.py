# ruff: noqa: N806

"""Audit class-prior and labeling-propensity assumptions for fixed outputs."""
# API reference: docs/user/reference/api/tools.md#analyze_pu_sensitivity

from sklearn.model_selection import train_test_split

from pu_toolbox.core.random import check_random_state
from pu_toolbox.diagnostics import analyze_pu_sensitivity
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
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y_pu,
        test_size=0.3,
        stratify=y_pu,
        random_state=42,
    )

    # PUSB fits observed P-vs-U labels without a prior.
    # Under SAR these SCAR-based curves are assumption sensitivity, not corrected truth.
    classifier = PUSBClassifier().fit(X_train, y_train)
    analysis = analyze_pu_sensitivity(
        y_valid,
        classifier.predict(X_valid),
        scores=classifier.decision_function(X_valid),
        population_priors=[0.2, 0.25, 0.3, 0.35, 0.4],
        label_propensities=[0.2, 0.3, 0.4, 0.5, 0.8],
    )

    print(analysis.to_markdown())


if __name__ == "__main__":
    main()
