"""Read-only engineering receipt checks; never approve recipes or resources."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs/research/pu_survey/data/candidate_extended_cnn_storage_probe_20261007.json"
spec = importlib.util.spec_from_file_location(
    "extended_cnn_driver", ROOT / "scripts/profile_survey_extended_cnn_storage.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_report(report, *, verify_sources=True, root=ROOT):
    """Check sweep/roles/cost/bytes and optionally current raw source identity."""
    require(report["schema_version"] == 2, "bad schema")
    require(
        report["status"]
        == "isolated_synthetic_extended_cnn_resource_evidence_not_formal_admission",
        "scope drift",
    )
    require(
        report["formal_budget_approved"] is False
        and report["candidate_protocol_ref"] is None
        and report["owner_signatures"] == [],
        "formal admission overclaim",
    )
    require(
        report["profile_order"] == "recipe_then_seed_sequential_fresh_process",
        "isolation scope drift",
    )
    config = report["measurement_spec"]
    recipes, seeds = config["recipes"], config["seeds"]
    require(
        bool(recipes)
        and len(set(recipes)) == len(recipes)
        and all(r in driver.RECIPES for r in recipes),
        "bad recipes",
    )
    require(
        bool(seeds)
        and len(set(seeds)) == len(seeds)
        and all(type(s) is int and 0 <= s < 2**32 for s in seeds),
        "bad seeds",
    )
    rows = report["profiles"]
    require(
        [(r["probe_recipe"], r["seed"]) for r in rows] == [(r, s) for r in recipes for s in seeds],
        "sweep missing, duplicate or out of order",
    )
    checked = {}

    def source(reference, digest):
        require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "bad source digest",
        )
        path = (root / reference).resolve()
        require(
            not Path(reference).is_absolute()
            and path.is_relative_to(root.resolve())
            and path.is_file(),
            "unsafe or missing source",
        )
        if reference in checked:
            require(checked[reference] == digest, "inconsistent source identity")
        if verify_sources:
            require(
                hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                f"current source mismatch: {reference}",
            )
        checked[reference] = digest

    require(
        set(report["driver_source_files_sha256"])
        == {
            "scripts/profile_survey_cnn_storage.py",
            "scripts/profile_survey_extended_cnn_storage.py",
        },
        "driver source set drift",
    )
    for reference, digest in report["driver_source_files_sha256"].items():
        source(reference, digest)
    for row in rows:
        options = driver.RECIPES[row["probe_recipe"]]
        method = options["method"]
        require(
            row["schema_version"] == 2
            and row["status"] == "synthetic_cnn_storage_probe_not_formal_admission",
            "child schema/scope drift",
        )
        require(
            type(row["seed"]) is int
            and row["method"] == method
            and row["device"] == config["device"],
            "child identity mismatch",
        )
        require(
            row["input_shape"] == [24, 3, config["image_size"], config["image_size"]],
            "image geometry mismatch",
        )
        require(
            type(config["image_size"]) is int
            and config["image_size"] >= 8
            and type(config["base_channels"]) is int
            and config["base_channels"] > 0,
            "invalid geometry",
        )
        require(
            row["encoder"]["base_channels"] == config["base_channels"]
            and row["encoder"]["matches_toolbox_default_width"] is (config["base_channels"] == 64),
            "backbone width overclaim",
        )
        require(
            row["single_profile_isolated_process"] is True
            and type(row["process_id"]) is int
            and row["process_id"] > 0,
            "missing isolation evidence",
        )
        require(
            row["formal_budget_approved"] is False
            and row["trusted_pickle_roundtrip_verified"] is True,
            "missing recovery or admission overclaim",
        )
        require(re.fullmatch(r"[0-9a-f]{40}", row["code_commit"]) is not None, "bad code commit")
        require(
            type(row["estimator_pickle_bytes"]) is int and row["estimator_pickle_bytes"] > 0,
            "missing pickle bytes",
        )
        for name, value in options.items():
            if name != "method":
                require(row["parameters"][name] == value, "recipe mismatch")
        expected_view = "ts" if method == "genpu" else "os"
        require(
            row["training_view"] == expected_view
            and row["calibration_applied"] is (expected_view == "ts"),
            "calibration mismatch",
        )
        stage = row["stage_optimizer_steps"]
        require(
            all(type(n) is int and n >= 0 for n in stage.values())
            and sum(stage.values()) == row["optimizer_steps"],
            "stage/full cost mismatch",
        )
        snapshots, contexts = row["epoch_weights_bytes"], row["checkpoint_training_contexts"]
        if method == "pulns":
            support = row["clean_support"]
            require(
                snapshots is None
                and contexts == []
                and row["epoch_checkpoint_status"] == "unavailable_no_epoch_callback",
                "fabricated PULNS snapshots",
            )
            require(
                support["train_ids"] == list(range(24))
                and support["support_ids"] == list(range(24, 32))
                and support["rows"] == 8
                and support["labels"] == "clean_pn"
                and support["disjoint_ids"] is True,
                "support identity drift",
            )
            require(
                support["formal_support_budget_approved"] is False
                and support["selection_test_labels_provided"] is False,
                "support label budget leak",
            )
        else:
            require(row["clean_support"] is None, "unexpected clean support")
            count = 2 if method == "genpu" else 4
            require(
                snapshots["count"] == len(contexts) == count
                and snapshots["all_snapshots_replayed"] is True
                and snapshots["final_scores_match"] is True,
                "snapshot replay mismatch",
            )
            require(
                0 < snapshots["min"] <= snapshots["max"]
                and count * snapshots["min"] <= snapshots["total"] <= count * snapshots["max"],
                "invalid measured bytes",
            )
            require(
                all(
                    set(c) == {"stage", "stage_epoch", "round_index", "optimizer_steps"}
                    for c in contexts
                ),
                "missing stage provenance",
            )
            require(all(c["round_index"] is None for c in contexts), "unexpected round")
            expected = (
                [("synthetic_pn", 1), ("synthetic_pn", 2)]
                if method == "genpu"
                else [("warmup", 1), ("warmup", 2), ("warmup", 3), ("pseudo_pn", 1)]
            )
            require(
                [(c["stage"], c["stage_epoch"]) for c in contexts] == expected,
                "stage provenance mismatch",
            )
            costs = [c["optimizer_steps"] for c in contexts]
            require(
                all(a < b for a, b in zip(costs[:-1], costs[1:], strict=True))
                and costs[-1] == row["optimizer_steps"],
                "paid cost rolled back",
            )
        if method == "holistic_pu":
            terminal = row["holistic_terminal_selection"]
            require(
                terminal["executed_warmup_epochs_"] == 3
                and terminal["pseudo_pn_optimizer_reset_"]
                is (options["pseudo_pn_initialization"] == "reinitialize"),
                "terminal recipe mismatch",
            )
            lzo = options["warmup_selection"] == "lzo_positive_loss"
            require(
                terminal["selected_warmup_epoch_"] in ((2, 3) if lzo else (3,)),
                "bad warmup terminal",
            )
            require(
                terminal["lzo_evaluated_rows_"] == (15 if lzo else 0)
                and terminal["lzo_forward_batches_"] == (3 if lzo else 0),
                "LZO evaluation cost mismatch",
            )
            require(
                terminal["discarded_warmup_optimizer_steps_"]
                == (3 - terminal["selected_warmup_epoch_"]) * 2,
                "discarded cost mismatch",
            )
        if config["device"] == "cpu":
            require(
                row["cuda_peak"] is None
                and row["cuda_peak_including_verification"] is None
                and row["cuda_identity"] is None,
                "CPU receipt overclaims CUDA",
            )
        else:
            require(
                config["device"] == "cuda"
                and row["cuda_peak"] is not None
                and row["cuda_identity"] is not None,
                "missing CUDA evidence",
            )
        require(
            row["host_resident_memory"]["precise_upper_bound"] is False, "RSS upper bound overclaim"
        )
        for phase in ("before_fit", "after_fit", "after_verification"):
            memory = row["host_resident_memory"][phase]
            require(
                all(
                    value is None or (type(value) is int and value > 0) for value in memory.values()
                ),
                "invalid OS memory measurement",
            )
        require(
            row["retained_module_storage"]["unique_parameter_count"]
            >= row["inference_model_parameter_count"]
            > 0,
            "retained model accounting mismatch",
        )
        expected_sources = {
            "scripts/profile_survey_cnn_storage.py",
            "pu_toolbox/estimators/deep/vision.py",
            "pu_toolbox/experiment/checkpoints.py",
            "pu_toolbox/utils/serialization.py",
            "pu_toolbox/core/training_views.py",
            "pu_toolbox/core/device.py",
            "pu_toolbox/estimators/deep/_validation.py",
            f"pu_toolbox/estimators/deep/{'gen_pu' if method == 'genpu' else method}.py",
        }
        require(set(row["source_files_sha256"]) == expected_sources, "source set mismatch")
        for reference, digest in row["source_files_sha256"].items():
            source(reference, digest)
    # Unavailable OS RSS is null, not NaN, zero, or an inferred capacity.
    json.dumps(report, allow_nan=False)
    return {
        "ok": True,
        "profiles_checked": len(rows),
        "source_files_checked": sorted(checked) if verify_sources else [],
        "formal_admission": False,
        "training_peak_upper_bound_proven": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT)
    args = parser.parse_args()
    try:
        result = validate_report(json.loads(args.record.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
