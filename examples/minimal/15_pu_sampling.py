"""Compare OS labeling and independent TS sampling from one Gaussian population."""

from __future__ import annotations

import argparse
import json

import numpy as np

from pu_toolbox.preprocessing import (
    PULabelingConfig,
    SourcePartition,
    construct_pu_dataset,
    make_gaussian_case_control_data,
)


def compare_sampling(
    *, seed: int = 7, class_prior: float = 0.4, n_p: int = 80, n_u: int = 800, c: float = 0.5
) -> dict:
    """Use the TS population draw as the clean source pool of the OS condition."""
    features, ts_labels, ts_truth, ts_metadata = make_gaussian_case_control_data(
        n_p=n_p, n_u=n_u, class_prior=class_prior, n_features=2, random_state=seed
    )
    population = ts_labels == 0
    population_truth = ts_truth[population]
    result = construct_pu_dataset(
        SourcePartition(
            features[population],
            population_truth,
            f"gaussian-{seed}",
            "population",
            np.asarray(ts_metadata["sample_ids"])[population],
        ),
        PULabelingConfig("scar", "fixed_count", seed, c=c),
        positive_classes=(1,),
        negative_classes=(0,),
    )
    os_labels = result.observations.labels
    os_unlabeled = population_truth[os_labels == 0]
    return {
        "seed": seed,
        "population_class_prior": class_prior,
        "os": {
            "n_labeled": int(np.count_nonzero(os_labels)),
            "n_unlabeled": int(len(os_unlabeled)),
            "label_frequency_requested": c,
            "label_frequency_realized": float(os_labels.sum() / population_truth.sum())
            if population_truth.sum()
            else None,
            "labeled_fraction": float(os_labels.mean()),
            "population_empirical_prior": float(population_truth.mean()),
            "unlabeled_empirical_prior": float(os_unlabeled.mean()) if len(os_unlabeled) else None,
            "sampling_policy": "fixed_count_without_replacement",
        },
        "ts": {
            "n_labeled": n_p,
            "n_unlabeled": n_u,
            "unlabeled_empirical_prior": ts_metadata["unlabeled_class_prior"],
            "training_empirical_prior": ts_metadata["training_class_prior"],
            "n_unique_sample_ids": len(set(ts_metadata["sample_ids"])),
            "sampling_policy": ts_metadata["sampling_policy"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(compare_sampling(seed=args.seed), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
