"""Generate SCAR/SAR PU data and inspect the hidden labeling mechanism."""
# API reference: docs/user/reference/api/tools.md#data-generation


# Example variables mirror the public estimator API.
# ruff: noqa: N806

import numpy as np

from pu_toolbox.core.random import check_random_state
from pu_toolbox.preprocessing import GaussianSourceConfig, PULabelingConfig, simulate_pu_dataset


def main():
    for mechanism in ("scar", "linear", "nonlinear"):
        rng = check_random_state(42)
        result = simulate_pu_dataset(
            GaussianSourceConfig(1000, 5, 0.3, 2.0, rng),
            PULabelingConfig(
                "scar" if mechanism == "scar" else "sar",
                "bernoulli",
                rng,
                c=0.4,
                propensity_rule=mechanism if mechanism != "scar" else None,
                strength=1.5 if mechanism != "scar" else 1.0,
                ensure_labeled=True,
            ),
        )
        X, y_pu = result.observations.features, result.observations.labels
        y_true = result.audit.true_labels
        propensity = result.audit.propensity
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
