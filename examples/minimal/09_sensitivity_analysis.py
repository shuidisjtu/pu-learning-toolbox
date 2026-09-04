# ruff: noqa: N806

"""Audit class-prior and labeling-propensity assumptions for fixed outputs."""
# API 参考（签名/参数/返回契约）：docs/user/reference/api.md

from sklearn.model_selection import train_test_split

from pu_toolbox.diagnostics import analyze_pu_sensitivity
from pu_toolbox.estimators.bias_aware import PUSBClassifier
from pu_toolbox.preprocessing import make_sar_dataset


def main() -> None:
    X, y_pu, _, _ = make_sar_dataset(
        n_samples=1200,
        n_features=6,
        class_prior=0.3,
        mechanism="linear",
        label_frequency=0.4,
        strength=1.5,
        random_state=42,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y_pu,
        test_size=0.3,
        stratify=y_pu,
        random_state=42,
    )

    classifier = PUSBClassifier().fit(X_train, y_train, class_prior=0.3)
    analysis = analyze_pu_sensitivity(
        y_valid,
        classifier.predict(X_valid),
        scores=classifier.decision_function(X_valid),
        class_priors=[0.2, 0.25, 0.3, 0.35, 0.4],
        label_propensities=[0.2, 0.3, 0.4, 0.5, 0.8],
    )

    print(analysis.to_markdown())


if __name__ == "__main__":
    main()
