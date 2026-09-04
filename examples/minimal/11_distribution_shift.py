# ruff: noqa: N806

"""Audit PU distribution shift and run guarded covariate adaptation."""
# API 参考（签名/参数/返回契约）：docs/user/reference/api.md

from pu_toolbox.preprocessing import make_scar_dataset
from pu_toolbox.workflows import ShiftAwarePUPipeline


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
    X_target = X_target + 0.25

    report = ShiftAwarePUPipeline(
        classifier="elkan_noto",
        cv=3,
        shift_cv=3,
        random_state=42,
    ).fit_evaluate(
        X_source,
        y_source,
        X_target,
        y_target_pu=y_target,
        y_true_source=y_true_source,
        y_true_target=y_true_target,
        class_prior=0.5,
        target_class_prior=0.5,
    )

    print(report.to_markdown())


if __name__ == "__main__":
    main()
