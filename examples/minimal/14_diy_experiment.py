"""Run four-role experiments with built-in and custom strategies on CPU."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pu_toolbox.estimators.risk import UPUClassifier
from pu_toolbox.experiment import (
    DatasetBundle,
    DatasetPart,
    ExperimentRunner,
    FitTrainer,
    ProtocolOA,
    ProtocolPA,
    SCARGenerator,
)
from pu_toolbox.experiment.protocols import Generator, SelectionProtocol, Trainer
from pu_toolbox.experiment.tracking import RunResult


class AuditedSCARGenerator(Generator):
    """Keep SCAR sampling and attach an application-specific policy identifier."""

    output_view = "pu"

    def generate(self, X, y_true, c, seed=None):  # noqa: N803
        labels, metadata = SCARGenerator().generate(X, y_true, c, seed)
        return labels, {**metadata, "policy_id": "diy-scar-v1"}


class ScaledFitTrainer(Trainer):
    """Fit preprocessing on train and keep it with the fitted predictor."""

    trains_on_real_labels = False

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):  # noqa: N803
        scaler = StandardScaler()
        train_features = scaler.fit_transform(X)
        trajectory = FitTrainer().fit(
            estimator, train_features, y, class_prior=class_prior, os_or_ts=os_or_ts
        )
        trajectory.model = Pipeline([("scale", scaler), ("classifier", trajectory.model)])
        return trajectory


class FixedThresholdPA(SelectionProtocol):
    """Choose a candidate by PU proxy accuracy at one validation-normalized threshold."""

    name = "PA-fixed"

    def select(self, trajectories, val_part, threshold_candidates=None, *, class_prior=None):
        artifact = ProtocolPA().select(
            trajectories, val_part, np.array([0.5]), class_prior=class_prior
        )
        return replace(artifact, protocol=self.name)


def make_bundle(seed: int = 7) -> DatasetBundle:
    """Create four disjoint splits from a synthetic population with prior 0.5."""
    rng = np.random.RandomState(seed)
    labels = np.tile([0, 1], 300)
    features = (rng.normal(size=(600, 5)) + labels[:, None]) * np.array([1, 10, 0.1, 2, 3])
    shuffled = rng.permutation(len(labels))
    parts = []
    for start, stop in ((0, 300), (300, 400), (400, 500), (500, 600)):
        indices = shuffled[start:stop]
        parts.append(
            DatasetPart(
                X=features[indices],
                labels=labels[indices],
                view="clean",
                indices=indices,
                for_selection=start != 500,
            )
        )
    return DatasetBundle(*parts)


def run_experiment(
    bundle: DatasetBundle,
    *,
    custom: bool = False,
    seed: int = 7,
    manifest_path: str | None = None,
) -> RunResult:
    """Train three candidates and select independently on PU and clean validation."""
    runner = ExperimentRunner(
        seed=seed,
        generator=AuditedSCARGenerator() if custom else SCARGenerator(),
        protocols=[FixedThresholdPA(), ProtocolOA()] if custom else [ProtocolPA(), ProtocolOA()],
        class_prior=0.5,
        threshold_candidates=np.linspace(0, 1, 11),
        manifest_path=manifest_path,
        config={
            "c": 0.4,
            "trainer": ScaledFitTrainer() if custom else FitTrainer(),
            "candidates": [{"reg_lambda": value} for value in (0.001, 0.01, 0.1)],
            "split_ref": {"dataset": "synthetic_diy", "split_seed": seed},
        },
    )
    return runner.fit(
        UPUClassifier(class_prior=0.5, fit_intercept=False, random_state=seed),
        bundle.train,
        bundle.pu_val,
        bundle.clean_val,
        bundle.test,
    )


def predict_selected(result: RunResult, protocol: str, features: np.ndarray) -> np.ndarray:
    """Apply the selected threshold with its stored validation-side score transform."""
    model = result.selected_models[protocol]
    selection = result.selections[protocol]
    if selection.threshold is None:
        return model.predict(features)
    scores = model.decision_function(features)
    minimum = selection.metrics.get("val_score_min")
    scale = selection.metrics.get("val_score_scale")
    if minimum is not None and scale:
        scores = (scores - minimum) / scale
    return (scores >= selection.threshold).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True)
    bundle = make_bundle(args.seed)
    for name, custom in (("basic", False), ("custom", True)):
        manifest = str(args.output_dir / f"{name}.json") if args.output_dir else None
        result = run_experiment(bundle, custom=custom, seed=args.seed, manifest_path=manifest)
        for protocol, selection in result.selections.items():
            candidate = result.manifest["selection"][protocol]["candidate_index"]
            print(
                f"{name}/{protocol}: candidate={candidate}, threshold={selection.threshold:.2f}, "
                f"test_accuracy={result.test_metrics[protocol]['accuracy']:.3f}"
            )


if __name__ == "__main__":
    main()
