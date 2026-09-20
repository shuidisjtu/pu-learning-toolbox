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

Two gates run before any training.  (1) ``--method`` must be a row of
``pu_toolbox/experiment/method_ledger.json``: methods outside the survey
scope fail loud instead of producing an unlabelled run.  (2)
``--class-prior`` is mandatory when the registry entry carries
``requires_class_prior`` — the estimator class attribute, mirrored into the
registry — so the requirement follows the implementation rather than the
prose.  The ledger entry itself only annotates the result: it is copied next
to the manifests so runs can be labelled (runtime method, native-sampling
assumption, adaptation level, prior semantics).

``--labeling-mechanism`` selects how the PU label view is *generated* — SCAR
(protocol §2.1) or one of the SAR LBE variants (protocol §2.3 pressure test) —
and is orthogonal to ``--method``: the mechanism is the experiment's
independent variable, so ``sar_lbe_a`` does not imply ``--method lbe`` and any
survey row can be run under either mechanism.  SAR is OA-only (protocol §2.3:
under SAR the PA protocol only logs diagnostics, official model selection and
conclusions use OA), so the SAR path injects ``[ProtocolOA()]`` explicitly and
nests its runs under ``<mechanism>/c_<token>/seed_<seed>``.  The SAR label
frequencies are restricted to the protocol values ``{0.05, 0.5}`` (PU-Bench
vary-e, see implementation_plan.md §2.2); a token outside that set fails before
any data is read.  Note that this CLI's ``--labeling-mechanism`` value space is
*not* the ``labeling_mechanism`` column of ``benchmarks/assigned_methods``
(``{scar, linear, nonlinear}``, the propensity-model shape inside that
benchmark runner): here the name picks one of the survey's generator
strategies.

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
        --method lbe --labeling-mechanism sar_lbe_a --c 0.05,0.5 \\
        --seeds 0,1,2 --out-dir results/survey/sar_lbe_a

    uv run python scripts/run_survey_experiment.py path/to/my/splits/ \\
        --oracle --seeds 0,1,2 --out-dir results/survey/pn_oracle
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart, validate_bundle
from pu_toolbox.experiment.manifest import load_manifest, write_manifest
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import (
    CleanLabelGenerator,
    ProtocolOA,
    SARLBEAGenerator,
    SARLBEBGenerator,
    SCARGenerator,
    SupervisedTrainer,
)

LEDGER_PATH = Path(__file__).resolve().parent.parent / "pu_toolbox/experiment/method_ledger.json"

_ROLE_FILES: tuple[str, ...] = ("train", "pu_val", "clean_val", "test")

# Label-view mechanisms this CLI can run (``--labeling-mechanism``): SCAR, the
# survey's main row, plus the two SAR LBE variants of the pressure test.
LABELING_MECHANISMS: tuple[str, ...] = ("scar", "sar_lbe_a", "sar_lbe_b")

# Mechanisms whose official selection protocol is OA only (protocol §2.3).
SAR_MECHANISMS: frozenset[str] = frozenset({"sar_lbe_a", "sar_lbe_b"})

# SAR label frequencies (PU-Bench vary-e, implementation_plan.md §2.2).
SAR_C_TOKENS: frozenset[str] = frozenset({"0.05", "0.5"})

_LABELING_GENERATORS: dict[str, type] = {
    "scar": SCARGenerator,
    "sar_lbe_a": SARLBEAGenerator,
    "sar_lbe_b": SARLBEBGenerator,
}


@dataclass(frozen=True)
class CValue:
    """One requested label frequency: numeric value plus the spelling used.

    Both parts are kept because they answer different questions.  ``value`` is
    what the estimator and the generator run on; ``token`` is what the user
    typed, and it names the output directory — ``0.05`` must not be
    renormalised into the SCAR grid's ``c_0.1``.
    """

    value: float
    token: str


def _parse_c_values(raw: str) -> list[CValue]:
    """Parse a ``--c`` argument into (value, token) pairs, rejecting bad input.

    Per token: strip the whitespace, then require a non-empty, finite number
    inside ``(0, 1]``.  A final pass rejects two tokens naming the same value
    (``0.05`` and ``5e-2``): they would select one output directory and
    silently reduce a two-point scan to a single overwritten run.
    """
    parsed: list[CValue] = []
    seen: dict[float, str] = {}
    for raw_token in raw.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"empty label frequency in --c {raw!r}")
        try:
            value = float(token)
        except ValueError:
            raise ValueError(f"label frequency {token!r} in --c {raw!r} is not a number") from None
        if not math.isfinite(value):
            raise ValueError(f"label frequency {token!r} in --c {raw!r} is not finite")
        if not 0.0 < value <= 1.0:
            raise ValueError(f"label frequency {token!r} in --c {raw!r} is outside (0, 1]")
        if value in seen:
            raise ValueError(
                f"duplicate label frequency: tokens {seen[value]!r} and {token!r} both "
                f"mean c={value}; keep one spelling per value so one request maps to "
                "one output directory"
            )
        seen[value] = token
        parsed.append(CValue(value=value, token=token))
    return parsed


def _resolve_labeling_generator(mechanism: str):
    """Map a ``--labeling-mechanism`` name to a fresh generator strategy."""
    if mechanism not in _LABELING_GENERATORS:
        raise ValueError(
            f"unknown labeling mechanism {mechanism!r}; expected one of "
            f"{', '.join(LABELING_MECHANISMS)}"
        )
    return _LABELING_GENERATORS[mechanism]()


def _resolve_labeling_protocols(mechanism: str) -> list | None:
    """Selection protocols for a mechanism; ``None`` means the runner default.

    SAR is OA-only (protocol §2.3), and the SAR branch must return a non-empty
    list: the runner interprets ``protocols or [ProtocolPA(), ProtocolOA()]``,
    so an empty list would silently reinstate the PA protocol SAR forbids.
    SCAR keeps the runner default of PA + OA.
    """
    if mechanism in SAR_MECHANISMS:
        return [ProtocolOA()]
    return None


def _validate_labeling_request(*, mechanism: str, is_oracle: bool, c_values: list[CValue]) -> None:
    """Gate the mechanism/oracle combination and the SAR label frequencies.

    Both gates run before any split is read, directory created or estimator
    built, so an impossible request costs nothing but the error message.
    """
    if is_oracle and mechanism != "scar":
        raise ValueError(
            "--oracle trains on real labels and generates no PU label view, so "
            f"--labeling-mechanism {mechanism!r} does not apply; drop --labeling-mechanism."
        )
    if mechanism in SAR_MECHANISMS:
        allowed = ", ".join(sorted(SAR_C_TOKENS))
        invalid = [item.token for item in c_values if item.token not in SAR_C_TOKENS]
        if invalid:
            raise ValueError(
                f"labeling mechanism {mechanism!r} runs the protocol's SAR label "
                f"frequencies only (PU-Bench vary-e: c in {{{allowed}}}); got "
                f"{', '.join(invalid)}. Re-run with --c {','.join(sorted(SAR_C_TOKENS))}."
            )


def _run_directory(
    out_root: Path, *, mechanism: str, c_value: CValue | None, seed: int, is_oracle: bool
) -> Path:
    """Output directory for one run, keyed by the requested c token."""
    if is_oracle:
        return Path(out_root) / "c_independent" / f"seed_{seed}"
    if mechanism in SAR_MECHANISMS:
        return Path(out_root) / mechanism / f"c_{c_value.token}" / f"seed_{seed}"
    return Path(out_root) / f"c_{c_value.token}" / f"seed_{seed}"


def _record_c_token(manifest_path: Path, token: str) -> None:
    """Add the requested c spelling to a manifest of a run that succeeded.

    The runner's manifest schema is a fixed whitelist and never serialises
    arbitrary config, so the CLI's own spelling of ``c`` — the one that names
    the output directory — is written back here rather than smuggled through
    the runner.
    """
    payload = load_manifest(manifest_path)
    payload["c_requested_token"] = token
    write_manifest(manifest_path, payload)


class OracleMLP(MLPClassifier):
    """Supervised MLP satisfying the toolbox classifier contract.

    ``MLPClassifier`` exposes ``predict_proba`` but no ``decision_function``,
    which the experiment layer requires (trajectory validation and OA
    thresholding both score through it); the positive-class log-odds is the
    natural score.  An MLP also keeps the tabular/text backbone of protocol
    §2.5, so the oracle stays comparable to the PU rows on those datasets.
    """

    label_semantics = "pn"

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
    ledger: dict[str, Any],
    method: str,
    provided: float | None,
    *,
    requires_class_prior: bool,
) -> float | None:
    """Gate a PU run on the survey ledger scope and the registry prior flag.

    Two checks run before any estimator is built:

    1. ``method`` must be a row of the survey ledger.  A method that is
       registered in the toolbox but sits outside the survey scope fails
       loud instead of producing a run that carries no survey annotation.
    2. ``--class-prior`` is mandatory when the registry says so.  The flag
       is the estimator class attribute (``requires_class_prior``), mirrored
       into the registry through ``_SYNC_FIELDS``; the ledger's
       ``prior_semantics`` text only annotates the error and the result.

    The oracle path never reaches this function: it trains on real labels
    and applies no class prior.
    """
    entry = ledger["methods"].get(method)
    if entry is None:
        raise ValueError(
            f"method '{method}' is not in the survey ledger "
            f"({LEDGER_PATH.name}, {len(ledger['methods'])} methods: "
            f"{', '.join(sorted(ledger['methods']))}); add its ledger entry "
            "before running it as a survey row."
        )
    if requires_class_prior and provided is None:
        raise ValueError(
            f"method '{method}' needs the population class prior "
            f"(registry requires_class_prior=True; ledger prior_semantics="
            f"{entry.get('prior_semantics', '')!r}); pass --class-prior."
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

    ``class_prior`` rides along because §3.1 defines pi as data-generation
    metadata that only the split artifact can record: PA's selection criterion is
    weighted by it, so without this the runner would have no prior to select
    with.  The addition is inert for split identity — nothing decides which split
    ran by reading ``split_ref``.
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
        "class_prior": split_manifest.get("class_prior"),
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
        "--labeling-mechanism",
        choices=LABELING_MECHANISMS,
        default="scar",
        help=(
            "PU label-view mechanism (default: scar), orthogonal to --method: "
            "'scar' keeps PA+OA selection, 'sar_lbe_a'/'sar_lbe_b' are the SAR "
            "pressure test (OA only, c in {0.05, 0.5})"
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
    parser.add_argument(
        "--protocol",
        default=None,
        help="survey-v1.2 (v1/v1.1 aliases) or matrix JSON; otherwise technical smoke",
    )
    parser.add_argument("--dataset", choices=("spambase", "imdb", "cifar10"), default=None)
    parser.add_argument(
        "--training-path",
        choices=("native_2d", "native_cnn", "cnn_feature_adapter"),
        default=None,
        help="select a matrix row; CIFAR oracle requires an explicit path",
    )
    parser.add_argument("--device", default="cpu", help="execution device (default: cpu)")
    parser.add_argument("--adapter-cache", default="data/cache/survey_adapter")
    parser.add_argument("--extraction-batch-size", type=int, default=64)
    return parser.parse_args(argv)


def _versioned_main(args, c_values, seed_values) -> int:
    """Execute a bound pilot unit; no mutation of the generic smoke path."""
    from pu_toolbox.experiment.survey_comparison import (
        load_comparison_protocol,
        validate_comparison_coverage,
    )
    from pu_toolbox.experiment.survey_execution import (
        SourceSpaceGenerator,
        assemble_model,
        cached_adapter,
        prepare_image_bundle,
    )
    from pu_toolbox.experiment.survey_protocol import (
        PROTOCOL_PATH,
        digest,
        load_protocol,
        resolve_unit,
        unit_checkpoint_bytes,
        validate_parameters,
    )
    from pu_toolbox.registry import get_metadata, register_all_builtin_methods

    method = "pn_oracle" if args.oracle else args.method or "upu"
    protocol_path = (
        PROTOCOL_PATH
        if args.protocol in ("survey-v1", "survey-v1.1", "survey-v1.2")
        else Path(args.protocol).resolve()
    )
    # These gates precede loading splits, building encoders and creating output directories.
    try:
        if args.dataset is None:
            raise ValueError("--protocol requires --dataset")
        if not seed_values or len(set(seed_values)) != len(seed_values):
            raise ValueError("survey seeds must be non-empty and unique")
        if args.oracle and (args.class_prior is not None or args.method is not None):
            raise ValueError("--oracle does not use --method or --class-prior")
        if args.extraction_batch_size < 1:
            raise ValueError("extraction batch size must be positive")
        protocol = load_protocol(protocol_path)
        if digest(protocol) == digest(load_protocol()):
            # Prove the pre-registered pair covers every unit it must, here
            # rather than per run.  A ghost mapping, a unit covered twice and a
            # mapping that covers nothing are properties of the two files, and
            # the last is visible only when both directions are checked; per run
            # they would surface as a unit that quietly carries no comparison
            # entry, long after the runs that could have revealed it were paid
            # for.  Keyed on the matrix contents rather than on its path, so a
            # copy of the shipped file is still checked: a run's comparison
            # block is resolved against the shipped comparison either way, and
            # nothing about the copy makes that resolution safer.  A genuinely
            # different matrix is not checked -- the comparison file is bound to
            # the shipped protocol by digest, so there is nothing to prove such
            # a run against, and its units resolve, or do not, one by one.
            validate_comparison_coverage(protocol, load_comparison_protocol(survey=protocol))
        row = resolve_unit(protocol, args.dataset, method, args.training_path)
        profile = protocol["method_profiles"][method]
        params = json.loads(args.model_params)
        validate_parameters(params, profile)
        candidates = _load_json_maybe(args.candidates, default=protocol["candidate_pool"])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("survey candidates must be a non-empty list")
        for candidate in candidates:
            validate_parameters(candidate, profile)
        ledger = load_ledger(LEDGER_PATH)
        register_all_builtin_methods()
        prior = (
            None
            if args.oracle
            else resolve_class_prior(
                ledger,
                method,
                args.class_prior,
                requires_class_prior=get_metadata(method).requires_class_prior,
            )
        )
        if "class_prior" in params and params["class_prior"] != prior:
            raise ValueError("constructor class_prior disagrees with --class-prior")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_root = (
        Path(args.out_dir)
        if args.out_dir
        else Path("results") / "survey" / args.dataset / method / row["training_path"]
    )
    for seed in seed_values:
        # A template or a dataset root selects the actual per-seed split;
        # never relabel the same split as five different protocol seeds.
        data_dir = Path(args.data_dir.replace("{seed}", str(seed)))
        if (data_dir / f"split_{seed}").is_dir():
            data_dir = data_dir / f"split_{seed}"
        try:
            parts = load_split_parts(data_dir)
            split_ref = resolve_split_ref(data_dir, args.split_ref)
            if split_ref.get("dataset") != args.dataset:
                raise ValueError("split manifest dataset does not match --dataset")
            if split_ref.get("seed") != seed:
                raise ValueError("split manifest seed does not match the requested run seed")
            source = DatasetBundle(*parts)
            image_manifest = adapter_manifest = encoder = None
            bundle = source
            if args.dataset == "imdb" and any(
                getattr(source, role).X.shape[1] != 384 for role in _ROLE_FILES
            ):
                raise ValueError("IMDB protocol requires SBERT 384-dimensional inputs")
            if args.dataset == "cifar10":
                bundle, encoder, image_manifest = prepare_image_bundle(source, protocol, seed)
                if row["training_path"] == "cnn_feature_adapter":
                    bundle, adapter_manifest = cached_adapter(
                        bundle,
                        encoder,
                        image_manifest,
                        cache_dir=Path(args.adapter_cache),
                        batch_size=args.extraction_batch_size,
                        device=args.device,
                    )
                    encoder = None
            model = assemble_model(
                protocol,
                row,
                bundle.train.X.shape[1],
                seed=seed,
                params=params,
                class_prior=prior,
                device=args.device,
                encoder=encoder,
            )
        except (OSError, KeyError, TypeError, ValueError, ImportError) as exc:
            print(f"error: cannot assemble survey unit: {exc}", file=sys.stderr)
            return 1
        for c_value in [None] if args.oracle else c_values:
            generator = (
                CleanLabelGenerator()
                if args.oracle
                else _resolve_labeling_generator(args.labeling_mechanism)
            )
            if row["training_path"] == "cnn_feature_adapter":
                generator = SourceSpaceGenerator(generator, source, bundle)
            config = {
                "candidates": candidates,
                "split_ref": split_ref,
                "architecture": "cnn" if row["training_path"] == "native_cnn" else "mlp",
                "survey_protocol": {
                    "path": str(protocol_path),
                    "dataset": args.dataset,
                    "method": method,
                    "training_path": row["training_path"],
                    "mechanism": args.labeling_mechanism,
                    "seeds": seed_values,
                    "c_tokens": [value.token for value in c_values],
                    "split_ref_overridden": args.split_ref is not None,
                    "constructor_overrides": params,
                },
                "adapter_manifest": adapter_manifest,
                "image_manifest": image_manifest,
                # Pre-run disk guard input. The networks are built inside fit,
                # so the size has to be derived from the locked protocol here
                # rather than introspected from an unfitted estimator.
                "checkpoint_bytes_per_component": unit_checkpoint_bytes(
                    protocol, row, input_dim=bundle.train.X.shape[1]
                ),
            }
            if args.oracle:
                config.update(
                    {
                        "trainer": SupervisedTrainer(),
                        "c_independent": True,
                        "broadcast_c_values": [value.value for value in c_values],
                    }
                )
            else:
                config["c_requested_token"] = c_value.token
            run_dir = _run_directory(
                out_root,
                mechanism=args.labeling_mechanism,
                c_value=c_value,
                seed=seed,
                is_oracle=args.oracle,
            )
            try:
                metrics = run_one(
                    model,
                    tuple(getattr(bundle, role) for role in _ROLE_FILES),
                    generator=generator,
                    protocols=[ProtocolOA()]
                    if args.oracle
                    else _resolve_labeling_protocols(args.labeling_mechanism),
                    seed=seed,
                    c=None if c_value is None else c_value.value,
                    class_prior=prior,
                    config=config,
                    manifest_path=run_dir / "manifest.json",
                )
                if c_value is not None:
                    _record_c_token(run_dir / "manifest.json", c_value.token)
                if not args.oracle:
                    (run_dir / "method_ledger_entry.json").write_text(
                        json.dumps(ledger["methods"][method], indent=2), encoding="utf-8"
                    )
            except Exception as exc:  # noqa: BLE001 - user-facing run boundary
                print(f"error: survey run failed: {exc}", file=sys.stderr)
                return 1
            print(f"[{method}] seed={seed} -> {run_dir / 'manifest.json'}\n  {metrics}")
    print("Versioned pilot evidence only: check manifest formal_blockers before aggregation.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)

    # Request gates first: a bad --c spelling or an impossible (mechanism,
    # oracle) combination must fail before any split is read, any directory
    # created and any estimator built.
    try:
        c_values = _parse_c_values(args.c)
        seed_values = [int(value) for value in args.seeds.split(",") if value.strip()]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        _validate_labeling_request(
            mechanism=args.labeling_mechanism, is_oracle=args.oracle, c_values=c_values
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.protocol is not None:
        return _versioned_main(args, c_values, seed_values)
    if args.training_path is not None or args.dataset is not None:
        print("error: --dataset/--training-path require --protocol", file=sys.stderr)
        return 1

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
            from pu_toolbox.core.exceptions import RegistryError
            from pu_toolbox.registry import (
                get_algorithm,
                get_metadata,
                register_all_builtin_methods,
            )

            register_all_builtin_methods()  # idempotent; standalone scripts need the registry
            class_prior = resolve_class_prior(
                ledger,
                method,
                args.class_prior,
                requires_class_prior=get_metadata(method).requires_class_prior,
            )
        except (ImportError, RegistryError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            model = get_algorithm(method)(**json.loads(args.model_params))
        except (RegistryError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"error: cannot create method '{method}': {exc}", file=sys.stderr)
            return 1
        generator = _resolve_labeling_generator(args.labeling_mechanism)
        protocols = _resolve_labeling_protocols(args.labeling_mechanism)
        ledger_entry = ledger["methods"].get(method)
        config_extra = {}

    config: dict[str, Any] = {"candidates": candidates, "split_ref": split_ref, **config_extra}
    out_root = Path(args.out_dir) if args.out_dir else Path("results") / "survey" / method

    if args.oracle:
        # PN oracle uses real labels and is independent of the PU labeling rate.
        # Keep one physical artifact per (dataset, seed); aggregation broadcasts
        # it to the requested c columns.
        run_specs: list[tuple[CValue | None, int]] = [(None, seed) for seed in seed_values]
        config["c_independent"] = True
        config["broadcast_c_values"] = [item.value for item in c_values]
    else:
        run_specs = [(item, seed) for item in c_values for seed in seed_values]
    completed_runs = 0
    for c_value, seed in run_specs:
        c = None if c_value is None else c_value.value
        run_dir = _run_directory(
            out_root,
            mechanism=args.labeling_mechanism,
            c_value=c_value,
            seed=seed,
            is_oracle=args.oracle,
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
        if c_value is not None:
            # Only a completed run's manifest earns the annotation.
            _record_c_token(manifest_path, c_value.token)
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
                    "broadcast_c_values": [item.value for item in c_values],
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
