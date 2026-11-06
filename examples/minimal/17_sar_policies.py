"""Compare Bernoulli SAR and named fixed-count LBE policies on one clean pool."""

from __future__ import annotations

import argparse
import json

import numpy as np

from pu_toolbox.experiment import SARLBEAGenerator, SARLBEBGenerator
from pu_toolbox.preprocessing import make_sar_labels
from pu_toolbox.utils.serialization import canonical_hash


def make_pool(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Create one OS source pool, shared by every labeling policy."""
    rng = np.random.RandomState(seed)
    truth = rng.permutation(np.r_[np.ones(160, int), np.zeros(240, int)])
    features = rng.normal(size=(400, 4)) + truth[:, None]
    return features, truth


def compare_policies(
    features: np.ndarray, truth: np.ndarray, *, label_frequency: float = 0.4, seed: int = 7
) -> list[tuple[np.ndarray, dict]]:
    """Return each label view and its mechanism, count policy and scoring source."""
    records = []
    n_positive = int(np.sum(truth == 1))
    for mechanism in ("linear", "nonlinear"):
        labels, propensity = make_sar_labels(
            features,
            truth,
            mechanism=mechanism,
            label_frequency=label_frequency,
            strength=1.5,
            ensure_labeled=False,
            return_propensity=True,
            random_state=seed,
        )
        n_labeled = int(labels.sum())
        records.append(
            (
                labels,
                {
                    "mechanism": mechanism,
                    "input_sampling": "os",
                    "sampling_policy": "independent_bernoulli",
                    "parameters": {"strength": 1.5, "feature_weights": "equal"},
                    "ensure_labeled": False,
                    "c_requested": label_frequency,
                    "c_realized": n_labeled / n_positive,
                    "n_positive": n_positive,
                    "n_labeled_expected": label_frequency * n_positive,
                    "n_labeled": n_labeled,
                    "propensity_mean_positive": float(propensity[truth == 1].mean()),
                    "propensity_std_positive": float(propensity[truth == 1].std()),
                    "score_source": "feature projection calibrated within this clean pool",
                    "source": "pu_toolbox.preprocessing.make_sar_propensity",
                    "generation_seed": seed,
                    "label_view_sha256": canonical_hash({"y_pu": labels.tolist()}),
                },
            )
        )
    for generator in (SARLBEAGenerator(), SARLBEBGenerator()):
        labels, metadata = generator.generate(features, truth, label_frequency, seed)
        records.append(
            (
                labels,
                {
                    **metadata,
                    "input_sampling": "os",
                    "sampling_policy": "weighted_fixed_count_without_replacement",
                    "count_policy": "round_to_even_minimum_one_available_positive",
                    "score_source": "LogisticRegression fitted on this clean pool's real labels",
                    "source": "PU-Bench 2d95a19 / toolbox named LBE policy",
                },
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label-frequency", type=float, default=0.4)
    args = parser.parse_args()
    features, truth = make_pool(args.seed)
    records = compare_policies(
        features, truth, label_frequency=args.label_frequency, seed=args.seed
    )
    print(json.dumps([metadata for _, metadata in records], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
