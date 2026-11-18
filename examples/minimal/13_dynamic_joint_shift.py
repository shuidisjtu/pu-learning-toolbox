# ruff: noqa: N806

"""Train the research dynamic joint-shift PU objective on two domains."""
# API reference: docs/user/reference/api/tools.md#dynamicjointshiftpuclassifier-research

from pu_toolbox.estimators.research import DynamicJointShiftPUClassifier
from pu_toolbox.preprocessing import make_gaussian_case_control_data


def main() -> None:
    # This solver uses label-0 rows as its marginal-risk pool. Draw independent
    # P and population-mixture U so the supplied population priors apply to U.
    X_source, y_source, _, source = make_gaussian_case_control_data(
        n_p=50, n_u=150, class_prior=0.5, n_features=4, separation=1.5, random_state=10
    )
    X_target, y_target, _, target = make_gaussian_case_control_data(
        n_p=30, n_u=90, class_prior=0.4, n_features=4, separation=1.5, random_state=20
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
        class_prior=source["population_class_prior"],
        target_class_prior=target["population_class_prior"],
    )
    print("Input sampling: independent TS; synthetic research smoke, not paper reproduction.")
    print(model.get_pu_metadata())
    print(model.predict_proba(X_target[:5]))


if __name__ == "__main__":
    main()
