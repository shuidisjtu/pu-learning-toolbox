"""Generate SCAR/SAR PU data and inspect the hidden labeling mechanism."""

# Example variables mirror the public estimator API.
# ruff: noqa: N806

import numpy as np

from pu_toolbox.preprocessing import make_sar_dataset


def main():
    for mechanism in ("scar", "linear", "nonlinear"):
        X, y_pu, y_true, propensity = make_sar_dataset(
            n_samples=1000,
            n_features=5,
            class_prior=0.3,
            mechanism=mechanism,
            label_frequency=0.4,
            strength=1.5,
            random_state=42,
        )
        positive = y_true == 1
        print(
            f"{mechanism:9s}",
            f"shape={X.shape}",
            f"labeled={y_pu.sum()}",
            f"realized_frequency={y_pu.sum() / positive.sum():.3f}",
            f"target_propensity={propensity[positive].mean():.3f}",
            f"propensity_std={propensity[positive].std():.3f}",
        )
        assert np.all(y_true[y_pu == 1] == 1)


if __name__ == "__main__":
    main()
