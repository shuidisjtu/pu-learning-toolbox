# ruff: noqa: N806

"""Profile PU data quality and inspect SCAR/SAR evidence."""
# API reference: docs/user/reference/api/tools.md#profile_pu_data

from pu_toolbox.core.random import check_random_state
from pu_toolbox.preprocessing import (
    GaussianSourceConfig,
    PULabelingConfig,
    profile_pu_data,
    simulate_pu_dataset,
)


def main() -> None:
    for mechanism in ("scar", "linear"):
        rng = check_random_state(42)
        result = simulate_pu_dataset(
            GaussianSourceConfig(1000, 5, 0.3, 2.0, rng),
            PULabelingConfig(
                "scar" if mechanism == "scar" else "sar",
                "bernoulli",
                rng,
                c=0.4,
                propensity_rule=mechanism if mechanism != "scar" else None,
                strength=2.0 if mechanism != "scar" else 1.0,
                ensure_labeled=True,
            ),
        )
        X, y_pu = result.observations.features, result.observations.labels
        y_true = result.audit.true_labels

        observed = profile_pu_data(X, y_pu, evaluation_prior=0.3)
        audited = profile_pu_data(X, y_pu, y_true=y_true, evaluation_prior=0.3)

        print(f"\n=== {mechanism.upper()} ===")
        print(
            "Observed-only: ",
            observed.selection_diagnostic["status"],
            f"identifying={observed.selection_diagnostic['is_identifying']}",
        )
        print(audited.format_text())


if __name__ == "__main__":
    main()
