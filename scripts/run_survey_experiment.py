"""Official PU-survey example script (protocol §2.4 item 9).

Runs the complete four-way PU experiment for one method from user-prepared
splits: load the four clean partitions -> validate the bundle -> run the
ExperimentRunner (SCAR generation, candidate pool, PA/OA offline selection,
independent test evaluation) -> write manifests -> print a summary.

Input directory layout (one .npz per role, produced by the user's split
step; see protocol §2.4 (二)3 — the toolbox does not split raw data):

    train.npz     X, y (true binary {0, 1}), indices
    pu_val.npz    X, y (true binary {0, 1}), indices
    clean_val.npz X, y (true binary {0, 1}), indices
    test.npz      X, y (true binary {0, 1}), indices

``indices`` are globally unique sample ids from the user's split manifest.
The runner generates the PU label views itself (SCAR contract), so the
four files always carry REAL labels.

Every manifest records a ``split_ref`` pointing back at the split it ran on
(protocol §2.4 item 11): by default the ``split_manifest.json`` sitting next
to the four .npz files, referenced by path, role sizes and index digest;
``--split-ref`` overrides it.

The script reads ``pu_toolbox/experiment/method_ledger.json`` as the
programmatic truth source: ``--class-prior`` is enforced for entries whose
prior_semantics is the population prior, and the method's ledger entry is
copied next to the manifests so results can be labelled (runtime method,
native-sampling assumption, adaptation level).

``--oracle`` switches to the PN oracle path (§2.4 item 10): the same runner
over the same underlying train partition, but the real labels are passed
through (``CleanLabelGenerator``), OA is the only selection protocol, and no
class prior is applied; the estimator defaults to a supervised MLP.  Its
result does not depend on ``c``, so a batch only needs one run per
(dataset, seed) — see pn_oracle_integration.md §5 (D-C).

Usage::

    uv run python scripts/run_survey_experiment.py path/to/my/splits/ \\
        --method upu --c 0.1,0.3 --seeds 0,1,2 --class-prior 0.1 \\
        --model-params '{"loss": "double_hinge", "class_prior": 0.1}' \\
        --out-dir results/survey/upu

    uv run python scripts/run_survey_experiment.py path/to/my/splits/ \\
        --oracle --seeds 0,1,2 --out-dir results/survey/pn_oracle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart, validate_bundle
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import (
    CleanLabelGenerator,
    ProtocolOA,
    SCARGenerator,
    SupervisedTrainer,
)

LEDGER_PATH = Path(__file__).resolve().parent.parent / "pu_toolbox/experiment/method_ledger.json"

_ROLE_FILES: tuple[str, ...] = ("train", "pu_val", "clean_val", "test")


class OracleMLP(MLPClassifier):
    """Supervised MLP satisfying the toolbox classifier contract.

    ``MLPClassifier`` exposes ``predict_proba`` but no ``decision_function``,
    which the experiment layer requires (trajectory validation and OA
    thresholding both score through it); the positive-class log-odds is the
    natural score.  An MLP also keeps the tabular/text backbone of protocol
    §2.5, so the oracle stays comparable to the PU rows on those datasets.
    """

    def decision_function(self, X):  # noqa: N803 - mirrors sklearn's estimator API
        # MLPClassifier returns float32 probabilities: clipping in that dtype
        # collapses 1 - 1e-12 back to 1.0 and divides by zero. Promote first.
        positive = np.clip(
            np.asarray(self.predict_proba(X)[:, 1], dtype=np.float64),
            1e-12,
            1.0 - 1e-12,
        )
        return np.log(positive / (1.0 - positive))


def load_ledger(path: Path) -> dict[str, Any]:
    """Load the method ledger JSON (programmatic truth source, protocol §4)."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("methods"), dict):
        raise ValueError(f"{path} is not a valid method ledger file.")
    return value


def load_split_parts(data_dir: Path) -> tuple[DatasetPart, ...]:
    """Load the four clean partitions and validate the four-way contract."""
    base = Path(data_dir)
    missing = [name for name in _ROLE_FILES if not (base / f"{name}.npz").is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing split files in {base}: {', '.join(missing)}. "
            "Provide user-prepared train/pu_val/clean_val/test partitions (protocol §2.4 (二)3)."
        )
    parts: tuple[DatasetPart, ...] = tuple(
        DatasetPart(
            X=np.load(base / f"{name}.npz")["X"],
            labels=np.load(base / f"{name}.npz")["y"],
            view="clean",
            indices=np.load(base / f"{name}.npz")["indices"],
            for_selection=(name != "test"),
        )
        for name in _ROLE_FILES
    )
    validate_bundle(DatasetBundle(*parts))
    return parts


def resolve_class_prior(
    ledger: dict[str, Any], method: str, provided: float | None
) -> float | None:
    """Enforce the required class prior per ledger prior_semantics."""
    entry = ledger["methods"].get(method)
    if entry and "population" in entry.get("prior_semantics", "") and provided is None:
        raise ValueError(
            f"method '{method}' needs the population class prior "
            f"(prior_semantics={entry['prior_semantics']!r}); pass --class-prior."
        )
    return provided


def run_one(
    model,
    parts: tuple[DatasetPart, ...],
    *,
    generator,
    protocols: list | None = None,
    seed: int,
    c: float | None,
    class_prior: float | None,
    config: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    """Run one (method, c, seed) experiment and return the test metrics."""
    runner_config = {**config}
    if c is not None:
        runner_config["c"] = c
    runner = ExperimentRunner(
        seed=seed,
        generator=generator,
        protocols=protocols,
        class_prior=class_prior,
        manifest_path=str(manifest_path),
        config=runner_config,
    )
    result = runner.fit(model, *parts)
    return result.test_metrics


def _load_json_maybe(path: str | None, *, default: Any) -> Any:
    if path is None:
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_split_ref(data_dir: Path, explicit: str | None) -> dict[str, Any]:
    """Reference the split a run used (protocol §2.4 item 11).

    An explicit ``--split-ref`` wins.  Otherwise the split directory's own
    ``split_manifest.json`` — which holds the per-role sample ids and their
    digest — is referenced by path, role sizes and index digest, so a run
    manifest says which split it used without inlining thousands of ids
    beside every run.
    """
    if explicit is not None:
        return json.loads(Path(explicit).read_text(encoding="utf-8"))
    manifest_path = Path(data_dir) / "split_manifest.json"
    if not manifest_path.is_file():
        return {}
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "manifest_path": str(manifest_path.resolve()),
        "dataset": split_manifest.get("dataset"),
        "seed": split_manifest.get("seed"),
        "role_sizes": split_manifest.get("role_sizes"),
        "indices_sha256": split_manifest.get("indices_sha256"),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one PU-survey method on user-prepared four-way splits."
    )
    parser.add_argument("data_dir", help="directory with train/pu_val/clean_val/test .npz files")
    parser.add_argument("--method", default=None, help="registered algorithm name (default: upu)")
    parser.add_argument(
        "--oracle",
        action="store_true",
        help=(
            "run the PN oracle path (§2.4 item 10) instead of a PU method: real "
            "labels, OA-only selection, no class prior; estimator is a supervised MLP"
        ),
    )
    parser.add_argument(
        "--model-params",
        default="{}",
        help="JSON string of classifier constructor parameters (class_prior, loss, ...)",
    )
    parser.add_argument(
        "--c", default="0.1", help="comma-separated label frequencies (0.1,0.3,0.5)"
    )
    parser.add_argument("--seeds", default="0", help="comma-separated seeds")
    parser.add_argument("--class-prior", type=float, default=None, help="population class prior π")
    parser.add_argument(
        "--candidates",
        default=None,
        help="JSON file with the candidate hyper-parameter pool (default: [{}])",
    )
    parser.add_argument(
        "--split-ref",
        default=None,
        help=(
            "JSON file with the split reference recorded in every manifest "
            "(default: reference the data_dir's own split_manifest.json)"
        ),
    )
    parser.add_argument(
        "--out-dir", default=None, help="results root (default: results/survey/<method>)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)

    try:
        parts = load_split_parts(data_dir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        ledger = load_ledger(LEDGER_PATH)
        candidates = _load_json_maybe(args.candidates, default=[{}])
        split_ref = resolve_split_ref(data_dir, args.split_ref)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # The oracle path (§2.4 item 10) and the PU path differ in the estimator,
    # the label view and the selection protocol, so resolve them together.
    if args.oracle:
        if args.class_prior is not None:
            print(
                "error: --oracle trains on real labels and needs no class prior; "
                "drop --class-prior.",
                file=sys.stderr,
            )
            return 1
        if args.method is not None:
            print(
                "error: --oracle runs the PN path and does not use --method; drop --method.",
                file=sys.stderr,
            )
            return 1
        try:
            model = OracleMLP(**json.loads(args.model_params))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: cannot create the oracle estimator: {exc}", file=sys.stderr)
            return 1
        generator = CleanLabelGenerator()
        protocols: list | None = [ProtocolOA()]
        class_prior = None
        method = "pn_oracle"
        ledger_entry: dict[str, Any] | None = None
        config_extra: dict[str, Any] = {"trainer": SupervisedTrainer()}
    else:
        method = args.method or "upu"
        try:
            class_prior = resolve_class_prior(ledger, method, args.class_prior)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            from pu_toolbox.core.exceptions import RegistryError
            from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

            register_all_builtin_methods()  # idempotent; standalone scripts need the registry
            model = get_algorithm(method)(**json.loads(args.model_params))
        except (ImportError, RegistryError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"error: cannot create method '{method}': {exc}", file=sys.stderr)
            return 1
        generator = SCARGenerator()
        protocols = None
        ledger_entry = ledger["methods"].get(method)
        config_extra = {}

    config: dict[str, Any] = {"candidates": candidates, "split_ref": split_ref, **config_extra}
    out_root = Path(args.out_dir) if args.out_dir else Path("results") / "survey" / method

    c_values = [float(value) for value in args.c.split(",") if value.strip()]
    seed_values = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.oracle:
        # PN oracle uses real labels and is independent of the PU labeling rate.
        # Keep one physical artifact per (dataset, seed); aggregation broadcasts
        # it to the requested c columns.
        run_specs = [(None, seed) for seed in seed_values]
        config["c_independent"] = True
        config["broadcast_c_values"] = c_values
    else:
        run_specs = [(c, seed) for c in c_values for seed in seed_values]
    completed_runs = 0
    for c, seed in run_specs:
        run_dir = (
            out_root / "c_independent" / f"seed_{seed}"
            if args.oracle
            else out_root / f"c_{c:.1f}" / f"seed_{seed}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        if ledger_entry is not None:
            (run_dir / "method_ledger_entry.json").write_text(
                json.dumps(ledger_entry, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        manifest_path = run_dir / "manifest.json"
        try:
            metrics = run_one(
                model,
                parts,
                generator=generator,
                protocols=protocols,
                seed=seed,
                c=c,
                class_prior=class_prior,
                config=config,
                manifest_path=manifest_path,
            )
        except Exception as exc:  # noqa: BLE001 - user-facing example script boundary
            c_label = "c_independent" if c is None else f"c={c}"
            print(
                f"error: run failed (method={method}, {c_label}, seed={seed}): {exc}",
                file=sys.stderr,
            )
            return 1
        completed_runs += 1
        c_label = "c_independent" if c is None else f"c={c}"
        print(f"[{method}] {c_label} seed={seed} -> {manifest_path}")
        for protocol, metrics_value in metrics.items():
            print(f"  {protocol}: {metrics_value}")

    if args.oracle and completed_runs:
        # Written only after every planned run succeeded: a calibration file
        # sitting next to no results would let a failed output directory pass as
        # an oracle row. What it records (see pn_oracle_integration.md §3.1) is
        # what makes these numbers comparable to — and distinguishable from —
        # the PU method rows.
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "oracle_integration.json").write_text(
            json.dumps(
                {
                    "method": "pn_oracle",
                    "estimator": type(model).__name__,
                    "generator": "CleanLabelGenerator",
                    "trainer": "SupervisedTrainer",
                    "protocols": ["OA"],
                    "selection_metric": "clean_val_accuracy",
                    "class_prior_applied": False,
                    "c_independent": True,
                    "broadcast_c_values": c_values,
                    "runs_completed": completed_runs,
                    "note": (
                        "PN oracle per protocol §2.4 item 10; the result is "
                        "c-independent, so one run per (dataset, seed) suffices."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
