"""Known and estimated priors through OS/TS training and reporting."""

import numpy as np

from pu_toolbox.workflows import PUPipeline


def main() -> None:
    for sampling, positive_u in (("os", 10), ("ts", 16)):
        # Disjoint supports make U's mixture weight identifiable.
        features = np.vstack([np.ones((10 + positive_u, 1)), -np.ones((40 - positive_u, 1))])
        y_pu = np.r_[np.ones(10, dtype=int), np.zeros(40, dtype=int)]
        for source in ("known", "estimated"):
            pipeline = PUPipeline(
                classifier="upu",
                classifier_params={"fit_intercept": False},
                prior_estimator="pen_l1",
                prior_params={"sigma": 0.5},
                input_sampling=sampling,
                cv=2,
            )
            report = pipeline.fit_evaluate(
                features, y_pu, class_prior=0.4 if source == "known" else None
            )
            prior = report.prior
            consumed = prior.training_consumption[-1]
            print(
                f"{sampling.upper()} {source}: input={prior.input_value:.3f} "
                f"({prior.input_target}), population={prior.value:.3f}, "
                f"training={consumed['value']:.3f} ({consumed['target']})"
            )


if __name__ == "__main__":
    main()
