"""ExperimentRunner — Template Method orchestration of the four-way protocol.

Design notes: fixed skeleton; every variation point is an injected
strategy (Generator/Trainer/SelectionProtocol) per implementation_plan.md
§1.4. Does NOT subclass — users inject their own strategy objects.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.base import clone

from .bundle import DatasetBundle, DatasetPart, validate_bundle
from .manifest import write_manifest
from .strategies import DeepFitTrainer, ProtocolOA, ProtocolPA, SCARGenerator
from .tracking import RunResult


class ExperimentRunner:
    """Configured runner for a four-way PU experiment (§2.4).

    Best-epoch semantics: every ``SelectionArtifact.epoch`` mirrors
    ``RunTrajectory.best_epoch`` — a 1-based *position* inside the
    selected run's ``epochs`` list (``epochs[epoch - 1]`` holds the
    selected record). It is unrelated to the estimator's own epoch
    labels (``EpochRecord.epoch``); the runner never assumes equality.
    Callers who restore the checkpoint for a re-run must index
    ``trajectory.epochs[artifact.epoch - 1]``.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        generator=None,
        protocols=None,
        class_prior: float | None = None,
        threshold_candidates=None,
        manifest_path: str | None = None,
        config: dict | None = None,
    ) -> None:
        self.seed = seed
        self.generator = generator or SCARGenerator()
        self.protocols = protocols or [ProtocolPA(), ProtocolOA()]
        self.class_prior = class_prior
        self.threshold_candidates = threshold_candidates
        self.manifest_path = manifest_path
        self.config = config or {}

    def fit(
        self,
        model,
        train: DatasetPart,
        pu_val: DatasetPart,
        clean_val: DatasetPart,
        test: DatasetPart,
    ) -> RunResult:
        """Run the full pipeline; see bundle contract (views must be clean).

        Steps: validate → generate PU views → train the candidate pool →
        offline selection per protocol → independent test evaluation →
        manifest + ``RunResult``.

        Raises
        ------
        ValueError
            If the generated pu-val PU view ends up with no labeled
            positive (the pu-val split has no real positive or the
            labeling rate ``c`` is too low).  Fail loudly instead of
            silently degrading PA's separation metric into its -1.0
            sentinel.
        """
        t0 = time.perf_counter()
        bundle = DatasetBundle(train=train, pu_val=pu_val, clean_val=clean_val, test=test)
        validate_bundle(bundle)

        # 2. generate PU views (SCAR / SAR via the injected generator)
        c = self.config.get("c", 0.1)
        y_pu_train, meta_train = self.generator.generate(train.X, train.labels, c, self.seed)
        y_pu_val, meta_val = self.generator.generate(pu_val.X, pu_val.labels, c, self.seed)
        train_pu = DatasetPart(X=train.X, labels=y_pu_train, view="pu", indices=train.indices)
        pu_val_pu = DatasetPart(X=pu_val.X, labels=y_pu_val, view="pu", indices=pu_val.indices)

        if int(np.sum(pu_val_pu.labels == 1)) == 0:
            raise ValueError(
                "generated pu_val PU view has no labeled positive; "
                "ensure pu_val contains real positives (and/or raise c)."
            )

        # 3. train the candidate pool (clone + set_params per config)
        candidates = self.config.get("candidates", [{}])
        trajectories = []
        for params in candidates:
            est = clone(model)
            if params:
                est.set_params(**params)
            trajectories.append(self._train(est, train_pu, pu_val_pu))

        # 4. offline selection: PA on the PU view, OA on the clean view,
        #    any other protocol gets the PU view (never clean labels).
        selections = {}
        for proto in self.protocols:
            if isinstance(proto, ProtocolPA):
                art = proto.select(trajectories, pu_val_pu, self.threshold_candidates)
            elif isinstance(proto, ProtocolOA):
                art = proto.select(trajectories, clean_val, self.threshold_candidates)
            else:
                art = proto.select(trajectories, pu_val_pu, self.threshold_candidates)
            selections[art.protocol] = art

        # 5. independent test evaluation (test never entered selection/training)
        test_metrics = {}
        for name, art in selections.items():
            est = trajectories[art.run_index].model
            scores = est.decision_function(test.X)
            if art.threshold is not None and np.ptp(scores) > 0:
                # Same min-max convention ProtocolOA used on the val view,
                # so the [0, 1] threshold grid stays in range.
                scores = (scores - scores.min()) / np.ptp(scores)
                pred = (scores >= art.threshold).astype(int)
            else:
                pred = est.predict(test.X)
            test_metrics[name] = {
                "accuracy": float(np.mean(pred == test.labels)),
                "auc": _auc(est, test),
            }

        manifest = {
            "seed": self.seed,
            "split_ref": self.config.get("split_ref", {}),
            "generation": {"train": meta_train, "pu_val": meta_val},
            "selection": {k: asdict(v) for k, v in selections.items()},
            "test_results": test_metrics,
            "elapsed": time.perf_counter() - t0,
            "failures": [],
        }
        if self.manifest_path:
            path = Path(self.manifest_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_manifest(path, manifest)
        return RunResult(selections=selections, test_metrics=test_metrics, manifest=manifest)

    def _train(self, est, train_pu: DatasetPart, pu_val_view: DatasetPart):
        trainer = self.config.get("trainer", DeepFitTrainer())
        val_pu = (pu_val_view.X, pu_val_view.labels) if pu_val_view is not None else None
        return trainer.fit(
            est,
            train_pu.X,
            train_pu.labels,
            class_prior=self.class_prior,
            val_pu=val_pu,
        )


def _auc(est, test: DatasetPart) -> float:
    """AUROC on real test labels; NaN (never a crash) when degenerate."""
    from pu_toolbox.metrics import pu_auc_roc

    try:
        return float(pu_auc_roc(test.labels, est.decision_function(test.X)))
    except Exception:
        return float("nan")
