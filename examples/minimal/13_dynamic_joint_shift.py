# ruff: noqa: N806

"""Train the research dynamic joint-shift PU objective on two domains."""

from pu_toolbox.estimators.research import DynamicJointShiftPUClassifier
from pu_toolbox.preprocessing import make_scar_dataset


def main() -> None:
    X_source, y_source, _ = make_scar_dataset(
        n=100, c=0.5, n_features=4, separation=1.5, random_state=10
    )
    X_target, y_target, _ = make_scar_dataset(
        n=60, c=0.5, n_features=4, separation=1.5, random_state=20
    )
    X_target += 0.25

    model = DynamicJointShiftPUClassifier(
        alpha=0.1,
        beta=0.5,
        hidden_dim=32,
        feature_dim=16,
        max_epochs=10,
        random_state=42,
        device="cpu",
    ).fit(
        X_source,
        y_source,
        X_target=X_target,
        y_target_pu=y_target,
        class_prior=0.5,
        target_class_prior=0.5,
    )
    print(model.get_pu_metadata())
    print(model.predict_proba(X_target[:5]))


if __name__ == "__main__":
    main()
