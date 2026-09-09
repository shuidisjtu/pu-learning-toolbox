"""Official PU-survey example script (protocol §2.4 item 9).

Runs the complete four-way PU experiment for one method from user-prepared
splits: load the four clean partitions -> validate the bundle -> run the
ExperimentRunner (SCAR generation, candidate pool, PA/OA offline selection,
independent test evaluation) -> write manifests -> print a summary.

Input directory layout (one .npz per role, produced by the user's split
step; see protocol §2.4 (二)3 — the toolbox does not split raw data):

    train.npz     X, y (true binary {+1, 0}), indices
    pu_val.npz    X, y (true binary {+1, 0}), indices
    clean_val.npz X, y (true binary {+1, 0}), indices
    test.npz      X, y (true binary {+1, 0}), indices

``indices`` are globally unique sample ids from the user's split manifest.
The runner generates the PU label views itself (SCAR contract), so the
four files always carry REAL labels.

The script reads ``pu_toolbox/experiment/method_ledger.json`` as the
programmatic truth source: ``--class-prior`` is enforced for entries whose
prior_semantics is the population prior, and the method's ledger entry is
copied next to the manifests so results can be labelled (runtime method,
native-sampling assumption, adaptation level).  The PN oracle (§2.4 item
10) uses SupervisedTrainer via the same runner and is wired in with the
P2 pilot batch; this script ships the PU path.

Usage::

    uv run python scripts/run_survey_experiment.py path/to/my/splits/ \\
        --method upu --c 0.1,0.3 --seeds 0,1,2 --class-prior 0.1 \\
        --model-params '{"loss": "double_hinge", "class_prior": 0.1}' \\
        --out-dir results/survey/upu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart, validate_bundle
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import SCARGenerator

LEDGER_PATH = Path(__file__).resolve().parent.parent / "pu_toolbox/experiment/method_ledger.json"

_ROLE_FILES: tuple[str, ...] = ("train", "pu_val", "clean_val", "test")


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
    seed: int,
    c: float,
    class_prior: float | None,
    config: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    """Run one (method, c, seed) experiment and return the test metrics."""
    runner = ExperimentRunner(
        seed=seed,
        generator=SCARGenerator(),
        class_prior=class_prior,
        manifest_path=str(manifest_path),
        config={**config, "c": c},
    )
    result = runner.fit(model, *parts)
    return result.test_metrics


def _load_json_maybe(path: str | None, *, default: Any) -> Any:
    if path is None:
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one PU-survey method on user-prepared four-way splits."
    )
    parser.add_argument("data_dir", help="directory with train/pu_val/clean_val/test .npz files")
    parser.add_argument("--method", default="upu", help="registered algorithm name (default: upu)")
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
        help="JSON file with the split manifest reference (recorded in manifests)",
    )
    parser.add_argument(
        "--out-dir", default=None, help="results root (default: results/survey/<method>)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        parts = load_split_parts(Path(args.data_dir))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        ledger = load_ledger(LEDGER_PATH)
        class_prior = resolve_class_prior(ledger, args.method, args.class_prior)
        candidates = _load_json_maybe(args.candidates, default=[{}])
        split_ref = _load_json_maybe(args.split_ref, default={})
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        from pu_toolbox.core.exceptions import RegistryError
        from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

        register_all_builtin_methods()  # idempotent; standalone scripts need the registry
        model = get_algorithm(args.method)(**json.loads(args.model_params))
    except (ImportError, RegistryError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: cannot create method '{args.method}': {exc}", file=sys.stderr)
        return 1

    config: dict[str, Any] = {"candidates": candidates, "split_ref": split_ref}
    out_root = Path(args.out_dir) if args.out_dir else Path("results") / "survey" / args.method
    ledger_entry = ledger["methods"].get(args.method)

    c_values = [float(value) for value in args.c.split(",") if value.strip()]
    seed_values = [int(value) for value in args.seeds.split(",") if value.strip()]
    for c in c_values:
        for seed in seed_values:
            run_dir = out_root / f"c_{c:.1f}" / f"seed_{seed}"
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
                    seed=seed,
                    c=c,
                    class_prior=class_prior,
                    config=config,
                    manifest_path=manifest_path,
                )
            except Exception as exc:  # noqa: BLE001 - user-facing example script boundary
                print(
                    f"error: run failed (method={args.method}, c={c}, seed={seed}): {exc}",
                    file=sys.stderr,
                )
                return 1
            print(f"[{args.method}] c={c} seed={seed} -> {manifest_path}")
            for protocol, metrics_value in metrics.items():
                print(f"  {protocol}: {metrics_value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
