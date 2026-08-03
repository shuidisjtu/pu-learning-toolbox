"""Run an official-data deep-PU smoke or paper-protocol benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .official_data import (
    environment_preflight,
    load_official_data_config,
    run_official_data_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_official_data_config(args.config)
    preflight = environment_preflight(config, data_root=args.data_root, download=args.download)
    if args.preflight_only:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "preflight.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    trials, summary = run_official_data_benchmark(
        config,
        args.output,
        data_root=args.data_root,
        download=args.download,
        resume=args.resume,
    )
    print(f"Wrote {len(trials)} trials and {len(summary)} summary rows to {args.output}")


if __name__ == "__main__":
    main()
