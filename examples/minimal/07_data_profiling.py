# ruff: noqa: N806

"""Profile PU data quality and inspect SCAR/SAR evidence."""

from pu_toolbox.preprocessing import make_sar_dataset, profile_pu_data


def main() -> None:
    for mechanism in ("scar", "linear"):
        X, y_pu, y_true, _ = make_sar_dataset(
            n_samples=1000,
            n_features=5,
            class_prior=0.3,
            separation=2.0,
            mechanism=mechanism,
            label_frequency=0.4,
            strength=2.0,
            random_state=42,
        )

        observed = profile_pu_data(X, y_pu, class_prior=0.3)
        audited = profile_pu_data(X, y_pu, y_true=y_true, class_prior=0.3)

        print(f"\n=== {mechanism.upper()} ===")
        print(
            "Observed-only: ",
            observed.selection_diagnostic["status"],
            f"identifying={observed.selection_diagnostic['is_identifying']}",
        )
        print(audited.format_text())


if __name__ == "__main__":
    main()
