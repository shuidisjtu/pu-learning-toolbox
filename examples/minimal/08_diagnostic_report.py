# ruff: noqa: N806

"""Build a diagnostic report for a fitted PUSB classifier."""
# API 参考（签名/参数/返回契约）：docs/user/reference/api.md

from sklearn.model_selection import train_test_split

from pu_toolbox.diagnostics import build_diagnostic_report
from pu_toolbox.estimators.bias_aware import PUSBClassifier
from pu_toolbox.preprocessing import make_sar_dataset


def main() -> None:
    X, y_pu, y_true, _ = make_sar_dataset(
        n_samples=1200,
        n_features=6,
        class_prior=0.3,
        separation=2.0,
        mechanism="linear",
        label_frequency=0.4,
        strength=1.5,
        random_state=42,
    )
    X_train, X_test, y_train, y_test, _, truth_test = train_test_split(
        X,
        y_pu,
        y_true,
        test_size=0.3,
        stratify=y_pu,
        random_state=42,
    )

    classifier = PUSBClassifier().fit(X_train, y_train, class_prior=0.3)
    report = build_diagnostic_report(
        X_test,
        y_test,
        estimator=classifier,
        y_true=truth_test,
        class_prior=0.3,
        random_state=42,
    )

    print(report.to_markdown())


if __name__ == "__main__":
    main()
