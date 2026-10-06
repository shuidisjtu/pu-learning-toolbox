"""Seven new CNN engineering recipes, each measured in a fresh subprocess.

Sequential isolated processes prevent previous recipes contaminating RSS
highwater. No formal admission, selection candidates or GPU fallback/retry.
Only the underlying probe's temporary artifacts are created and removed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/profile_survey_cnn_storage.py"
RECIPES = {
    "pulns_clean_reward": {"method": "pulns"},
    "genpu_identity": {"method": "genpu", "generator_output": "identity"},
    "genpu_tanh": {"method": "genpu", "generator_output": "tanh"},
    "holistic_fixed_continue": {
        "method": "holistic_pu",
        "warmup_selection": "fixed",
        "pseudo_pn_initialization": "continue",
    },
    "holistic_fixed_reinitialize": {
        "method": "holistic_pu",
        "pseudo_pn_initialization": "reinitialize",
        "warmup_selection": "fixed",
    },
    "holistic_lzo_continue": {
        "method": "holistic_pu",
        "warmup_selection": "lzo_positive_loss",
        "pseudo_pn_initialization": "continue",
    },
    "holistic_lzo_reinitialize": {
        "method": "holistic_pu",
        "warmup_selection": "lzo_positive_loss",
        "pseudo_pn_initialization": "reinitialize",
    },
}


def source_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_recipes(
    *, recipes=None, seeds=(0, 1, 2), base_channels=64, image_size=32, device="cpu"
):
    """Strict child receipts; failed/timeout children stop, never silently retry."""
    spec = importlib.util.spec_from_file_location("cnn_storage_arguments", PROBE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    recipes = tuple(RECIPES) if recipes is None else tuple(recipes)
    seeds = tuple(seeds)
    if (
        not recipes
        or len(set(recipes)) != len(recipes)
        or any(recipe not in RECIPES for recipe in recipes)
    ):
        raise ValueError("recipes must be nonempty unique known names")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    common = dict(base_channels=base_channels, image_size=image_size, device=device)
    for recipe in recipes:
        for seed in seeds:
            probe.validate_probe_arguments(**RECIPES[recipe], seed=seed, **common)
    sources = {str(path.relative_to(ROOT)): source_digest(path) for path in (Path(__file__), PROBE)}
    profiles = []
    for recipe in recipes:
        options = RECIPES[recipe]
        for seed in seeds:
            command = [
                sys.executable,
                str(PROBE),
                "--method",
                options["method"],
                "--seeds",
                str(seed),
                "--base-channels",
                str(base_channels),
                "--image-size",
                str(image_size),
                "--device",
                device,
            ]
            for name, value in options.items():
                if name != "method":
                    command.extend(["--" + name.replace("_", "-"), value])
            result = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=300
            )
            if result.returncode:
                raise RuntimeError(
                    f"probe failed recipe={recipe} seed={seed}; no retry: {result.stderr[-2000:]}"
                )
            child = json.loads(result.stdout)
            if child.get("schema_version") != 2 or len(child.get("profiles", [])) != 1:
                raise ValueError("child must return one v2 profile")
            row = child["profiles"][0]
            if (row["method"], row["seed"], row["device"], row["input_shape"]) != (
                options["method"],
                seed,
                device,
                [24, 3, image_size, image_size],
            ):
                raise ValueError("child receipt identity mismatch")
            if (
                not row["trusted_pickle_roundtrip_verified"]
                or row["formal_budget_approved"] is not False
            ):
                raise ValueError("child receipt overclaims admission or lacks trusted recovery")
            if (
                row["source_files_sha256"].get(str(PROBE.relative_to(ROOT)))
                != sources[str(PROBE.relative_to(ROOT))]
            ):
                raise ValueError("child receipt source mismatch")
            if row["encoder"]["base_channels"] != base_channels:
                raise ValueError("child backbone width mismatch")
            for name, value in options.items():
                if name != "method" and row["parameters"].get(name) != value:
                    raise ValueError("child recipe parameter mismatch")
            row["probe_recipe"] = recipe
            row["single_profile_isolated_process"] = True
            profiles.append(row)
    if sources != {
        str(path.relative_to(ROOT)): source_digest(path) for path in (Path(__file__), PROBE)
    }:
        raise RuntimeError("probe driver/source changed during measurement")
    # Reject non-finite JSON, including unexpected output from a child.
    json.dumps(profiles, allow_nan=False)
    return {
        "schema_version": 2,
        "status": "isolated_synthetic_extended_cnn_resource_evidence_not_formal_admission",
        "profile_order": "recipe_then_seed_sequential_fresh_process",
        "measurement_spec": {"recipes": list(recipes), "seeds": list(seeds), **common},
        "driver_source_files_sha256": sources,
        "formal_budget_approved": False,
        "candidate_protocol_ref": None,
        "owner_signatures": [],
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", nargs="+", choices=list(RECIPES), default=list(RECIPES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    try:
        report = profile_recipes(**vars(args))
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
